import os
from pathlib import Path

from datasets import Dataset, interleave_datasets, load_dataset
from transformers import AutoTokenizer
import random
import numpy as np

from openrlhf.utils import DeepspeedStrategy
import torch
from transformers import set_seed, StoppingCriteria
import re

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

def check_tokenizer_chat_template(tokenizer):
    """
    Check if tokenizer has non none chat_template attribute.
    """
    if hasattr(tokenizer, "chat_template"):
        if tokenizer.chat_template is not None:
            return True
    return False

def get_tokenizer(pretrain, model, padding_side="left", strategy=None, use_fast=True, truncation_side = 'left'):
    tokenizer = AutoTokenizer.from_pretrained(pretrain, trust_remote_code=True, use_fast=use_fast)
    tokenizer.padding_side = padding_side
    tokenizer.truncation_side = truncation_side
    # NOTE: When enable vLLM, do not resize_token_embeddings, or the vocab size will mismatch with vLLM.
    # https://github.com/facebookresearch/llama-recipes/pull/196
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
    if strategy:
        strategy.print(f'Tokenizer padding side: {tokenizer.padding_side}')
        strategy.print(f'Tokenizer truncation side: {tokenizer.truncation_side}')
    return tokenizer


def get_strategy(args):
    strategy = DeepspeedStrategy(
        seed=getattr(args, "seed", 42),
        max_norm=getattr(args, "max_norm", 1.0),
        micro_train_batch_size=getattr(args, "micro_train_batch_size", 1),
        train_batch_size=getattr(args, "train_batch_size", 128),
        zero_stage=args.zero_stage,
        bf16=getattr(args, "bf16", True),
        args=args,
    )
    return strategy


def blending_datasets(
    datasets,
    probabilities,
    strategy=None,
    seed=42,
    max_count=5000000,
    return_eval=True,
    stopping_strategy="first_exhausted",
    train_split="train",
    eval_split="test",
):
    datasets = datasets.split(",")
    probabilities = list(map(float, probabilities.split(",")))
    assert len(probabilities) == len(datasets)

    train_data_list = []
    eval_data_list = []
    for i, dataset in enumerate(datasets):
        dataset = dataset.strip()
        strategy.print(f"dataset: {dataset}")

        data_dir = dataset.split("@")[1].strip() if "@" in dataset else None
        dataset = dataset.split("@")[0].strip()
        dataset_basename = os.path.basename(dataset)

        ext = os.path.splitext(dataset)[-1]
        # local python script
        if ext == ".py" or (
            os.path.isdir(dataset) and os.path.exists(os.path.join(dataset, f"{dataset_basename}.py"))
        ):
            data = load_dataset(dataset, trust_remote_code=True)
            strategy.print(f"loaded {dataset} with python script")
        # local text file
        elif ext in [".json", ".jsonl", ".csv"]:
            ext = ext.lower().strip(".")
            if ext == "jsonl":
                ext = "json"
            data = load_dataset(ext, data_files=dataset)
        # remote/local folder or common file
        else:
            data = load_dataset(dataset, data_dir=data_dir)
            strategy.print(f"loaded {dataset} from files")

        if train_split and train_split in data:
            train_data = data[train_split].select(range(min(max_count, len(data[train_split]))))
        else:
            train_data = data.select(range(min(max_count, len(data))))
        train_data_list.append(train_data)

        if return_eval:
            if eval_split and eval_split in data:
                eval_data = data[eval_split].select(range(min(max_count, len(data[eval_split]))))
            # train will contains eval? TODO
            else:
                eval_data = train_data.select(range(min(max_count, int(len(train_data) * 0.03))))
            eval_data_list.append(eval_data)

    # merge datasets
    if strategy.is_rank_0():
        print(train_data_list)

    train_dataset = interleave_datasets(
        train_data_list,
        probabilities=probabilities,
        seed=seed,
        stopping_strategy=stopping_strategy,
    )
    if return_eval:
        eval_dataset = interleave_datasets(
            eval_data_list,
            probabilities=probabilities,
            seed=seed,
            stopping_strategy=stopping_strategy,
        )
        return train_dataset, eval_dataset
    else:
        return train_dataset

# I have no idea why the cluster has a OSError: input/output error when loading all dataset at once
# sequential loading on each gpu is a workaround, but this is not a good solution
# change back to blending dataset if you do not have this issue
def sequential_data_loading(rank, num_processes, args, strategy, return_eval=True, stopping_strategy="first_exhausted", pretrain_data = False):
    for current_rank in range(num_processes):
        if rank == current_rank:
            print(f"Rank {rank} is loading data.")
            if return_eval:
                train_data, eval_data = blending_datasets(
                    args.dataset,
                    args.dataset_probs,
                    strategy,
                    args.seed,
                    max_count=args.max_samples,
                    stopping_strategy=stopping_strategy,
                    train_split=args.train_split,
                    eval_split=args.eval_split,
                    return_eval=return_eval,
                )
            else:
                if pretrain_data:
                    train_data = blending_datasets(
                        args.pretrain_data,
                        args.pretrain_data_probs,
                        strategy,
                        args.seed,
                        return_eval=return_eval,
                        train_split=args.pretrain_split,
                    )
                # PPO Prompt data
                else:
                    train_data = blending_datasets(
                        args.prompt_data,
                        args.prompt_data_probs,
                        strategy,
                        args.seed,
                        max_count=args.max_samples,
                        stopping_strategy=stopping_strategy,
                        train_split=args.prompt_split,
                        return_eval=return_eval,
                    )
        torch.distributed.barrier()  
    if return_eval:
        return train_data, eval_data
    else:
        return train_data


# some tokenizers (tulu and zephyr) tokenize "10" as "1" and "0"
# therefore we cannot use Confidence: 1 as the stopping criteria, since it might still generate "0" to form "10"
# therefore, we should continue generating if the confidence is followed by 1 and stop otherwise
class MyStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        sequences_should_be_stopped = []
        for ids in input_ids: 
            generated_text = self.tokenizer.decode(ids, skip_special_tokens=True)
            match = re.search(r"Confidence: (\d+)", generated_text)
            if match:
                number = match.group(1)
                # if number is '1', according to tulu, it might still generate '0' to form '10'
                if number == '1':
                    sequences_should_be_stopped.append(False)
                else:
                    sequences_should_be_stopped.append(True)
            else:
                sequences_should_be_stopped.append(False)
        return all(sequences_should_be_stopped)


# noqa
# def get_unsupervised_data(
#     args,
#     rank,
#     num_processes,
#     tokenizer,
#     strategy = None,
# ):
#     unsupervised_raw_datasets = sequential_data_loading(rank, num_processes, args, strategy, return_eval=False, train_split=args.pretrain_split, pretrain_data = True)
#     column_names = unsupervised_raw_datasets["train"].column_names
#     text_column_name = "text" if "text" in column_names else column_names[0]
#     def tokenize_function(examples):
#         return tokenizer(examples[text_column_name])
    
#     tokenized_datasets = unsupervised_raw_datasets.map(
#         tokenize_function,
#         batched=True,
#         num_proc=args.preprocessing_num_workers,
#         remove_columns=column_names,
#         load_from_cache_file=True,
#         desc="Running tokenizer on dataset",
#     )
#     block_size = args.max_prompt_seq_len + args.max_answer_seq_len

#     def group_texts(examples):
#         # Concatenate all texts.
#         concatenated_examples = {
#             k: list(chain(*examples[k]))
#             for k in examples.keys()
#         }
#         total_length = len(concatenated_examples[list(examples.keys())[0]])
#         # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
#         # customize this part to your needs.
#         if total_length >= block_size:
#             total_length = (total_length // block_size) * block_size
#         # Split by chunks of max_len.
#         result = {
#             k:
#             [t[i:i + block_size] for i in range(0, total_length, block_size)]
#             for k, t in concatenated_examples.items()
#         }
#         result["labels"] = result["input_ids"].copy()
#         return result

#     lm_datasets = tokenized_datasets.map(
#         group_texts,
#         batched=True,
#         num_proc=args.preprocessing_num_workers,
#         load_from_cache_file=True,
#         desc=f"Grouping texts in chunks of {block_size}",
#     )

#     train_dataset = lm_datasets["train"]

#     return train_dataset


def return_value(input, op='None'):
    if torch.is_tensor(input):
        if op == 'mean':
            return input.mean().item()
        else:
            return input.item()
    return input

                
        