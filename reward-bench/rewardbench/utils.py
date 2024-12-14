# Copyright 2023 AllenAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import logging
import os
from typing import Any, Dict, List, Union

import pandas as pd
import torch
from datasets import Dataset, DatasetDict, Value, concatenate_datasets, load_dataset
from fastchat.conversation import Conversation
from huggingface_hub import HfApi
from transformers import PreTrainedTokenizer

from rewardbench.models import REWARD_MODEL_CONFIG
import copy
import random

# HuggingFace Hub locations
CORE_EVAL_SET = "allenai/reward-bench"
EXTRA_PREF_SETS = "allenai/pref-test-sets"
BON_CANDIDATES = "ai2-adapt-dev/HERM_BoN_candidates"  # private until officially supported
EVAL_REPO = "allenai/reward-bench-results"  # data repo to upload results

# get token from HF_TOKEN env variable, but if it doesn't exist pass none
HF_TOKEN = os.getenv("HF_TOKEN", None)
api = HfApi(token=HF_TOKEN)
#PROMPT = 'Please answer the following question and rate your confidence on a scale from 0 to 10.\n\nQuestion: {question}'


# Lets just prepend this system prompt and append a randon confidence score on the fly, instead of creating a new dataset
# some randomness should be ok
PROMPT = (
    "For the following question, provide your best response first, followed by your confidence in the accuracy or helpfulness of your response. Rate your confidence on a scale from 0 to 10.\n"
    "```Example Format:\n"
    f"<Your responses>\n"
    "Confidence: <Insert your numerical confidence level from 0 to 10, reflecting how certain you are that your answer is accurate or helpful.>```\n\n"
    "Ensure that your response strictly adheres to this format. Explicitly include the word 'Confidence:' in your response."
).strip()

def torch_dtype_mapping(dtype_str):
    """
    Helper function for argparse to map string to torch dtype.
    """
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "float64": torch.float64,
    }
    if dtype_str not in dtype_map:
        raise argparse.ArgumentTypeError(f"Invalid torch dtype: {dtype_str}")
    return dtype_map[dtype_str]


def calculate_scores_per_section(example_counts, subset_mapping, metrics):
    """
    Helper function for immediately logging RewardBench scores.
    """
    section_scores = {}
    for section, tests in subset_mapping.items():
        total_weighted_score = 0
        total_examples = 0
        for test in tests:
            if test in metrics:
                total_weighted_score += metrics[test] * example_counts[test]
                total_examples += example_counts[test]
        if total_examples > 0:
            section_scores[section] = total_weighted_score / total_examples
        else:
            section_scores[section] = 0
    return section_scores


def check_tokenizer_chat_template(tokenizer):
    """
    Check if tokenizer has non none chat_template attribute.
    """
    if hasattr(tokenizer, "chat_template"):
        if tokenizer.chat_template is not None:
            return True
    return False


def save_to_hub(
    results_dict: Union[Dict, List],
    model_name: str,
    target_path: str,
    debug: bool = False,
    local_only: bool = False,
    save_metrics_for_beaker: bool = False,
):
    """
    Utility for saving results in dict to the hub in programatic organization.

    Args:
        results_dict: dictionary of results to save.
        model_name: name of the model (including organization).
        target_path: path to save the results in the hub. Usually set in script (e.g. eval-set/, eval-set-scores/).
        debug: if True, save to debug repo on HF.
        local_only: if True, do not save to HF (for most non-AI2 users).
        save_metrics_for_beaker: if True, save metrics for AI2 beaker visualization.

    Returns:
        scores_url: URL to the saved scores (optional).
    """
    scores_path = f"./results/{target_path}{model_name}.json"

    if save_metrics_for_beaker:
        # ai2 internal visualization, not needed externally, global path intentional.
        dirname = os.path.dirname("/output/metrics.json")
        os.makedirs(dirname, exist_ok=True)  # redundant in Beaker code
        with open("/output/metrics.json", "w+") as f:  # save format for AI2 beaker to show results
            json.dump(results_dict, f)

    dirname = os.path.dirname(scores_path)
    os.makedirs(dirname, exist_ok=True)

    # remove old data
    if os.path.isfile(scores_path):
        os.remove(scores_path)

    with open(scores_path, "w") as f:
        if isinstance(results_dict, Dict):
            dumped = json.dumps(results_dict, indent=4, sort_keys=True)  # nol removed , default=str
            f.write(dumped)
        # else, dump each row in list
        else:
            for record in results_dict:
                dumped = json.dumps(record, indent=4, sort_keys=True) + "\n"
                f.write(dumped)

    if not local_only:
        scores_url = api.upload_file(
            path_or_fileobj=scores_path,
            path_in_repo=target_path + f"{model_name}.json",
            repo_id=EVAL_REPO if not debug else "ai2-adapt-dev/herm-debug",  # push to correct results repo
            repo_type="dataset",
            commit_message=f"Add chosen-rejected text with scores for  model {model_name}",
        )
        return scores_url
    else:
        return None


def map_conversations_testsets(example):
    prompt = example["prompt"]
    example["text_chosen"] = prompt + [{"role": "assistant", "content": example["chosen"]}]
    example["text_rejected"] = prompt + [{"role": "assistant", "content": example["rejected"]}]
    return example


def load_preference_dataset(
    dataset_name: str,
    split: str = "train",
    json: bool = False,
    conv: Conversation = None,
    tokenizer: PreTrainedTokenizer = None,
    logger: logging.Logger = None,
) -> Dataset:
    """
    Load a preference dataset from the datasets library.

    Expects the data the following schema.
    - prompt (string): question
    - chosen (list): all turns of the conversation (including the prompt), chosen answer
    - rejected (list): all turns of the conversation (including the prompt), rejected answer

    Removes all excess columns, only returns scores over the provided data in order.

    Args:
        dataset_name (str): The name of the dataset to load (HuggingFace or local directory)
        split (str): The split of the dataset to load (train, validation, test, ...)

    Returns:
        dataset (Dataset): The loaded dataset with prompt, text_chosen, and text_rejected columns.
            text_ indicates a full conversation ending with that turn
    """
    if json:
        dataset = load_dataset("json", data_files=dataset_name)
    else:
        dataset = load_dataset(dataset_name, split=split)

    # if datasetdict, flatten all splits
    if isinstance(dataset, DatasetDict):
        available_splits = list(dataset.keys())
        datasets_to_combine = [dataset[split] for split in available_splits]
        dataset = concatenate_datasets(datasets_to_combine)

    # if has column question without prompt, rename question column to prompt
    if "question" in dataset.column_names:
        assert "prompt" not in dataset.column_names, "Both prompt and question columns found"
        dataset = dataset.rename_column("question", "prompt")
    if "input" in dataset.column_names:
        assert "prompt" not in dataset.column_names, "Both prompt and question columns found"
        dataset = dataset.rename_column("input", "prompt")

    # switch to format used for data utils
    # e.g. for evaluating this data https://huggingface.co/datasets/allenai/preference-test-sets
    # python -m rewardbench/rewardbench.py --dataset-name allenai/preference-test-sets --split shp
    features = dataset.features

    def switch_format(example):
        # chosen/rejected append {"role": "assistnat", "content": chosen}
        example["prompt"] = example["chosen"][:-1]
        example["chosen"] = example["chosen"][-1]["content"]
        example["rejected"] = example["rejected"][-1]["content"]
        return example

    # NOTE: We do NOT want to support every schema. These are the main three to start with
    # 1. Prompt is in a list of previous turns, chosen and rejected are final message from assistant
    # 2. Prompt is a string, chosen and rejected are full conversations with different final turns
    # 3. Prompt is not existent, chosen and rejected are full conversations with different final turns
    # TODO implement system prompts correctly (though, often doesn't work for Reward Models)

    # if prompt isn't a column,
    if "prompt" not in dataset.column_names:
        dataset = dataset.map(
            switch_format,
            num_proc=8,
            load_from_cache_file=False,
        )
    # elif prompt is a list and not a str, same function works
    elif not isinstance(features["prompt"], list):
        dataset = dataset.map(
            switch_format,
            num_proc=8,
            load_from_cache_file=False,
        )

    # update features
    features = dataset.features

    # assert the correct types
    assert features["chosen"].dtype == "string", f"chosen is wrong type (should be string): {features['chosen']}"
    assert features["rejected"].dtype == "string", f"rejected is wrong type (should be string): {features['rejected']}"

    # tokenize the data
    usable_tokenizer = check_tokenizer_chat_template(tokenizer)

    # assert either conv is passed or tokenizer has chat_template
    assert conv is not None or usable_tokenizer

    if usable_tokenizer:
        if logger is not None:
            logger.info("*** Preparing dataset with HF Transformers ***")
        # docs https://huggingface.co/docs/transformers/main/en/chat_templating
        dataset = dataset.map(
            prepare_dialogue_from_tokenizer,
            fn_kwargs={"tokenizer": tokenizer},
            num_proc=8,
            load_from_cache_file=False,
        )

    # else use FastChat to get chat template
    else:
        if logger is not None:
            logger.info("*** Preparing dataset with FastChat ***")
        dataset = dataset.map(
            prepare_dialogue,
            fn_kwargs={"dialogue_template": conv},
            num_proc=8,
            load_from_cache_file=False,
        )

    # remove excess data
    keep_columns = ["prompt", "text_chosen", "text_rejected"]
    all_cols = dataset.column_names
    dataset = dataset.remove_columns([c for c in all_cols if c not in keep_columns])
    return dataset


def load_eval_dataset(
    core_set: bool = True,
    custom_dialogue_formatting: bool = False,
    conv: Conversation = None,
    tokenizer: PreTrainedTokenizer = None,
    logger: logging.Logger = None,
    keep_columns: List[str] = ["text_chosen", "text_rejected", "id"],
    max_turns: int = None,
    mode = 'chosen',
    add_probabilities = False,
    format_prompt = False,
    pure_number = False,
    probabilities_side = 'right',
) -> tuple[Dataset, list[str]]:
    """
    Loads either the core eval set for HERM or the existing preference data test sets.

    Args:
        core_set: if True, load the core eval set for HERM.
        custom_dialogue_formatting: if True, format the dialogue as needed for custom models (e.g. SHP and PairRM).
        conv: fastchat conversation template.
                If None (default) the passed tokenizer needs to have a usable chat template.
        tokenizer: HuggingFace tokenizer to use. The tokenizer's chat template, if available, has precedence over conv.
        logger: logger to use for logging. If None (default), no logging is done.
        keep_columns: list of columns to keep in the dataset.
        max_turns: maximum number of turns in the dialogue (usually even). If None (default), no filtering is done.

    Returns:
        dataset: loaded dataset with required properties.
        subsets: list of subsets for the corresponding samples in the dataset.
    """
    if core_set:
        raw_dataset = load_dataset(CORE_EVAL_SET, split="filtered")
    else:
        raw_dataset = load_dataset(EXTRA_PREF_SETS)
        modified_datasets = []

        # Iterate over each subset in the DatasetDict
        for subset_name, subdataset in raw_dataset.items():
            # if subset column exists, move to subsubset (for pref sets)
            if "subset" in subdataset.column_names:
                subdataset = subdataset.rename_column("subset", "subsubset")

            # Add a new column 'subset' to the dataset with the subset name
            subdataset = subdataset.add_column("subset", [subset_name] * len(subdataset))

            # Append the modified dataset to the list
            # remove pku_safer and pku_better from the dict, no longer part of the benchmark
            if subset_name not in ["pku_safer", "pku_better"]:
                modified_datasets.append(subdataset)

        # Concatenate all the modified datasets into one dataset
        raw_dataset = concatenate_datasets(modified_datasets)

    # Apply chat template
    if not custom_dialogue_formatting:
        usable_tokenizer = check_tokenizer_chat_template(tokenizer)

        # assert either conv is passed or tokenizer has chat_template
        assert conv is not None or usable_tokenizer

        if usable_tokenizer:
            if logger is not None:
                logger.info("*** Preparing dataset with HF Transformers ***")
            # docs https://huggingface.co/docs/transformers/main/en/chat_templating
            dataset = raw_dataset.map(
                prepare_dialogue_from_tokenizer,
                fn_kwargs={"tokenizer": tokenizer, "mode": mode, "add_probabilities": add_probabilities, 'format_prompt': format_prompt, 'pure_number': pure_number, 'probabilities_side':probabilities_side},
                num_proc=8,
                load_from_cache_file=False,
            )

        # else use FastChat to get chat template
        else:
            if logger is not None:
                logger.info("*** Preparing dataset with FastChat ***")
            dataset = raw_dataset.map(
                prepare_dialogue,
                fn_kwargs={"dialogue_template": conv, "mode": mode, "add_probabilities": add_probabilities, 'format_prompt': format_prompt, 'pure_number': pure_number, "probabilities_side": probabilities_side},
                num_proc=8,  # using >1 process causes issues with re-assigning prompt in example
                load_from_cache_file=False,
            )
    else:
        if logger is not None:
            logger.info("*** Preparing dataset with custom formatting ***")

        def map_conversations(example, core_set=True):
            if mode == 'chosen':
                example['rejected'] = copy.deepcopy(example['chosen'])
            elif mode == 'rejected':
                example['chosen'] = copy.deepcopy(example['rejected'])
            if add_probabilities:
                low_probability = random.randint(0, 3)
                if mode == 'both_high':
                    low_probability = random.randint(7, 10)
                high_probability = random.randint(7, 10)
                if mode == 'both_low':
                    high_probability = random.randint(0, 3)
                if mode == 'random':
                    low_probability = random.randint(0, 10)
                    high_probability = random.randint(0, 10)
                if probabilities_side == 'right':
                    if pure_number:
                        example['chosen'] = example['chosen'].rstrip() + f"\n{low_probability}."
                        example['rejected'] = example['rejected'].rstrip() + f"\n{high_probability}."
                    else:
                        # chosen with low and rejected with high
                        example['chosen'] = example['chosen'].rstrip() + f"\nConfidence: {low_probability}."
                        example['rejected'] = example['rejected'].rstrip() + f"\nConfidence: {high_probability}."

                else:
                    if pure_number:
                        example['chosen'] = f"{low_probability}.\n" + example['chosen'].strip() 
                        example['rejected'] = f"{high_probability}.\n" + example['rejected'].strip()
                    else:
                        # chosen with low and rejected with high
                        example['chosen'] =  f"Confidence: {low_probability}.\n" + example['chosen'].strip()
                        example['rejected'] = f"Confidence: {high_probability}.\n" + example['rejected'].strip()
            
            # if format_prompt:
            #     example['prompt'] = PROMPT.format(question = example['prompt'])

            if core_set:
                if format_prompt:
                    example["text_chosen"] = [
                        {"role": "system", "content": PROMPT},
                        {"role": "user", "content": f'{example["prompt"]}'},
                        {"role": "assistant", "content": example["chosen"]},
                    ]
                    example["text_rejected"] = [
                        {"role": "system", "content": PROMPT},
                        {"role": "user", "content": f'{example["prompt"]}'},
                        {"role": "assistant", "content": example["rejected"]},
                    ]
                else:
                    example["text_chosen"] = [
                        {"role": "user", "content": example["prompt"]},
                        {"role": "assistant", "content": example["chosen"]},
                    ]
                    example["text_rejected"] = [
                        {"role": "user", "content": example["prompt"]},
                        {"role": "assistant", "content": example["rejected"]},
                    ]
            else:
                if format_prompt:
                    prompt = example["prompt"]
                    example["text_chosen"] = [{"role": "system", "content": PROMPT}] + [{"role": "system", "user": prompt}] + [{"role": "assistant", "content": example["chosen"]}]
                    example["text_rejected"] = [{"role": "system", "content": PROMPT}] + [{"role": "system", "user": prompt}] + [{"role": "assistant", "content": example["rejected"]}]
                else:
                    prompt = example["prompt"]
                    example["text_chosen"] = prompt + [{"role": "assistant", "content": example["chosen"]}]
                    example["text_rejected"] = prompt + [{"role": "assistant", "content": example["rejected"]}]
            return example

        dataset = raw_dataset.map(
            map_conversations,
            fn_kwargs={"core_set": core_set},
            num_proc=8,
        )

    if max_turns is not None:
        assert max_turns > 0, "max_turns must be greater than 0"

        # filter long answers (MT Bench prompt as 1 or 2 turn examples)
        def filter_long_turns(batch):
            return len(batch["text_chosen"]) <= max_turns

        dataset = dataset.filter(filter_long_turns)

    # take column subset from dataset
    subsets = dataset["subset"]

    # remove columns if set and not custom_dialogue_formatting
    all_cols = dataset.column_names
    dataset = dataset.remove_columns([c for c in all_cols if c not in keep_columns])

    return dataset, subsets


def load_bon_dataset(
    best_of: int = 16,
    custom_dialogue_formatting: bool = False,
    conv: Conversation = None,
    tokenizer: PreTrainedTokenizer = None,
    logger: logging.Logger = None,
    remove_columns: List[str] = None,
):
    """
    Loads the BON candidates dataset.
    """

    alpaca_eval = load_dataset("ai2-adapt-dev/HERM_BoN_candidates", "alpaca_eval")
    mt_bench = load_dataset("ai2-adapt-dev/HERM_BoN_candidates", "mt_bench")
    merged_alpaca_eval = concatenate_datasets([alpaca_eval["zephyr"], alpaca_eval["tulu"]])
    merged_mt_bench = concatenate_datasets([mt_bench["zephyr"], mt_bench["tulu"]])

    # add column "subset" alpaca_eval
    merged_alpaca_eval = merged_alpaca_eval.add_column(
        "subset", ["alpaca_eval" for i in range(len(merged_alpaca_eval))]
    )
    # rename column dataset to dataset_details
    merged_alpaca_eval = merged_alpaca_eval.rename_column("dataset", "dataset_details")
    merged_mt_bench = merged_mt_bench.rename_column("category", "dataset_details")
    # convert alpaca eval id to int
    merged_alpaca_eval = merged_alpaca_eval.cast_column("id", Value(dtype="int64", id=None))

    # rename generator to model
    merged_alpaca_eval = merged_alpaca_eval.rename_column("generator", "model")
    merged_mt_bench = merged_mt_bench.rename_column("generator", "model")

    # rename instruction to prompt
    merged_alpaca_eval = merged_alpaca_eval.rename_column("instruction", "prompt")
    merged_mt_bench = merged_mt_bench.rename_column("instruction", "prompt")

    # add column "subset" mt_bench
    merged_mt_bench = merged_mt_bench.add_column("subset", ["mt_bench" for i in range(len(merged_mt_bench))])

    # remove question_id
    merged_mt_bench = merged_mt_bench.remove_columns("question_id")

    # remove model_id
    merged_mt_bench = merged_mt_bench.remove_columns("model_id")

    raw_dataset = concatenate_datasets([merged_alpaca_eval, merged_mt_bench])

    # unroll every row in ['output'] to a new row, all other columns are copied,
    # index is changed to tuple (index, output_index)
    def unroll_output(row, n):
        rows = []
        outputs = row["output"]
        id = row["id"]

        for i, output in enumerate(outputs[:n]):
            new_row = row.copy()
            new_row["output_new"] = output
            new_row["index"] = [id, i]
            del new_row["output"]
            del new_row["id"]
            rows.append(new_row)
        return rows

    new_dataset = []
    for row in raw_dataset:
        new_dataset.extend([r for r in unroll_output(row, n=best_of)])

    # create huggingface dataset through pandas
    unrolled_dataset = Dataset.from_pandas(pd.DataFrame(data=new_dataset))
    # rename output_new to text
    unrolled_dataset = unrolled_dataset.rename_column("output_new", "input")
    unrolled_dataset = unrolled_dataset.rename_column("index", "id")

    # Apply chat template
    if not custom_dialogue_formatting:
        usable_tokenizer = check_tokenizer_chat_template(tokenizer)

        # assert either conv is passed or tokenizer has chat_template
        assert conv is not None or usable_tokenizer

        if usable_tokenizer:
            if logger is not None:
                logger.info("*** Preparing dataset with HF Transformers ***")
            # docs https://huggingface.co/docs/transformers/main/en/chat_templating
            dataset = unrolled_dataset.map(
                prepare_dialogue_from_tokenizer,
                fn_kwargs={"tokenizer": tokenizer, "ift": True},
            )

        # else use FastChat to get chat template
        else:
            if logger is not None:
                logger.info("*** Preparing dataset with FastChat ***")
            dataset = unrolled_dataset.map(
                prepare_dialogue,
                fn_kwargs={"dialogue_template": conv, "ift": True},
                num_proc=8,
            )
    else:
        if logger is not None:
            logger.info("*** Preparing dataset with custom formatting ***")

        def map_conversations_ift(example):
            example["text"] = [
                {"role": "user", "content": example["prompt"]},
                {"role": "assistant", "content": example["input"]},
            ]
            return example

        dataset = unrolled_dataset.map(
            map_conversations_ift,
            # fn_kwargs={"core_set": core_set},
            num_proc=8,
        )

    # remove column input
    dataset = dataset.remove_columns(remove_columns)

    return dataset


def prepare_dialogue_from_tokenizer(
    example: Dict[str, Any],
    tokenizer: PreTrainedTokenizer,
    ift: bool = False,
    mode = 'chosen',
    add_probabilities = False,
    format_prompt = False,
    pure_number = False,
    probabilities_side = 'right',
) -> Dict[str, Any]:
    if all(k in example.keys() for k in ("chosen", "rejected")):
        # multi turn
        if isinstance(example["prompt"], list) and len(example["prompt"]) > 0:
            # iterate through prompt messages, alternate user and assistant, end with example["chosen"]/rejected
            messages = []
            for i, (line) in enumerate(example["prompt"]):
                p = line["content"]
                _ = line["role"]
                if (i + 1) % 2 == 1:
                    messages.append({"role": "user", "content": p})
                else:
                    messages.append({"role": "assistant", "content": p})
            # assert that the last message before this is user
            assert messages[-1]["role"] == "user"
            # multi-turn only in prior set
            # so it is not a big deal (since these part are the same, and only last sentence is chosen/rejected, we format prompt only for the last user msg)
            if format_prompt:
                if 'user' in tokenizer.chat_template and 'system' not in tokenizer.chat_template:
                    messages[-1]['content'] = f"{PROMPT}\n\n{messages[-1]['content']}"  
                else:
                    messages = [{"role": "system", "content": PROMPT}] + messages

            # required for DPO code only, otherwise discarded
            temp_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
            )

            if mode == 'chosen':
                example['rejected'] = copy.deepcopy(example['chosen'])
            elif mode == 'rejected':
                example['chosen'] = copy.deepcopy(example['rejected'])

            if add_probabilities:
                low_probability = random.randint(0, 3)
                if mode == 'both_high':
                    low_probability = random.randint(7, 10)
                high_probability = random.randint(7, 10)
                if mode == 'both_low':
                    high_probability = random.randint(0, 3)
                if mode == 'random':
                    low_probability = random.randint(0, 10)
                    high_probability = random.randint(0, 10)
                if probabilities_side == 'right':
                    if pure_number:
                        example['chosen'] = example['chosen'].rstrip() + f"\n{low_probability}."
                        example['rejected'] = example['rejected'].rstrip() + f"\n{high_probability}."
                    else:
                        # chosen with low and rejected with high
                        example['chosen'] = example['chosen'].rstrip() + f"\nConfidence: {low_probability}."
                        example['rejected'] = example['rejected'].rstrip() + f"\nConfidence: {high_probability}."
                else:
                    if pure_number:
                        example['chosen'] = f"{low_probability}.\n" + example['chosen'].strip() 
                        example['rejected'] = f"{high_probability}.\n" + example['rejected'].strip()
                    else:
                        # chosen with low and rejected with high
                        example['chosen'] =  f"Confidence: {low_probability}.\n" + example['chosen'].strip()
                        example['rejected'] = f"Confidence: {high_probability}.\n" + example['rejected'].strip()


            # end with chosen/rejected
            messages.append({"role": "assistant", "content": example["chosen"]})
            example["text_chosen"] = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
            )

            messages[-1] = {"role": "assistant", "content": example["rejected"]}
            example["text_rejected"] = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
            )
            example['text_chosen'] = example['text_chosen'].rstrip()
            example['text_rejected'] = example['text_rejected'].rstrip()
            example["prompt"] = temp_prompt

        # single turn
        else:
            # needed for DPO
            if format_prompt:
                if 'user' in tokenizer.chat_template and 'system' not in tokenizer.chat_template:
                    messages = [
                        {"role": "user", "content": f'{PROMPT}\n\n{example["prompt"]}'},
                    ]
                else:
                    messages = [
                        {"role": "system", "content": PROMPT},
                        {"role": "user", "content": f'{example["prompt"]}'},
                    ]
            else:
                messages = [
                    {"role": "user", "content": example["prompt"]},
                ]
            temp_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
            )

            if mode == 'chosen':
                example['rejected'] = copy.deepcopy(example['chosen'])
            elif mode == 'rejected':
                example['chosen'] = copy.deepcopy(example['rejected'])
            if add_probabilities:
                low_probability = random.randint(0, 3)
                if mode == 'both_high':
                    low_probability = random.randint(7, 10)
                high_probability = random.randint(7, 10)
                if mode == 'both_low':
                    high_probability = random.randint(0, 3)
                if probabilities_side == 'right':
                    if pure_number:
                        example['chosen'] = example['chosen'].rstrip() + f"\n{low_probability}."
                        example['rejected'] = example['rejected'].rstrip() + f"\n{high_probability}."
                    else:
                        # chosen with low and rejected with high
                        example['chosen'] = example['chosen'].rstrip() + f"\nConfidence: {low_probability}."
                        example['rejected'] = example['rejected'].rstrip() + f"\nConfidence: {high_probability}."
                else:
                    if pure_number:
                        example['chosen'] = f"{low_probability}.\n" + example['chosen'].strip() 
                        example['rejected'] = f"{high_probability}.\n" + example['rejected'].strip()
                    else:
                        # chosen with low and rejected with high
                        example['chosen'] =  f"Confidence: {low_probability}.\n" + example['chosen'].strip()
                        example['rejected'] = f"Confidence: {high_probability}.\n" + example['rejected'].strip()
            
            if format_prompt:
                if 'user' in tokenizer.chat_template and 'system' not in tokenizer.chat_template:
                    messages = [
                        {"role": "user", "content": f'{PROMPT}\n\n{example["prompt"]}'},
                        {"role": "assistant", "content": example["chosen"]},
                    ]
                else:
                    messages = [
                        {"role": "system", "content": PROMPT},
                        {"role": "user", "content": f'{example["prompt"]}'},
                        {"role": "assistant", "content": example["chosen"]},
                    ]
            else:
                messages = [
                    {"role": "user", "content": example["prompt"]},
                    {"role": "assistant", "content": example["chosen"]},
                ]
            example["text_chosen"] = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
            )
            if format_prompt:
                if 'user' in tokenizer.chat_template and 'system' not in tokenizer.chat_template:
                    messages = [
                        {"role": "user", "content": f'{PROMPT}\n\n{example["prompt"]}'},
                        {"role": "assistant", "content": example["rejected"]},
                    ]
                else:
                    messages = [
                        {"role": "system", "content": PROMPT},
                        {"role": "user", "content": f'{example["prompt"]}'},
                        {"role": "assistant", "content": example["rejected"]},
                    ]
            else:
                messages = [
                    {"role": "user", "content": example["prompt"]},
                    {"role": "assistant", "content": example["rejected"]},
                ]
            example["text_rejected"] = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
            )
            # I do not why but llama3-8b-rm-mixture add a '\n' after eos
            example['text_chosen'] = example['text_chosen'].rstrip()
            example['text_rejected'] = example['text_rejected'].rstrip()
            example["prompt"] = temp_prompt
    elif ift:
        # TODO adapt this for DPO models with tokenize_row function
        messages = [
            {"role": "user", "content": example["prompt"]},
            {"role": "assistant", "content": example["input"]},
        ]
        example["text"] = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
        )
    else:
        raise ValueError(
            "Could not format example as dialogue for `rm` task!"
            f"Require `[chosen, rejected]` keys but found {list(example.keys())}"
        )
    return example

# fastchat template
# this is tricky, since some model has system_message="BEGINNING OF CONVERSATION:",
# just prepend the confidence query to the user prompt
def prepare_dialogue(
    example: Dict[str, Any],
    dialogue_template: Conversation,
    ift: bool = False,
    mode = 'chosen',
    add_probabilities = False,
    format_prompt = False,
    pure_number = False,
    probabilities_side = 'right',
) -> Dict[str, Any]:
    """Format example to single- or multi-turn dialogue."""
    if all(k in example.keys() for k in ("chosen", "rejected")):
        # multi turn
        if isinstance(example["prompt"], list) and len(example["prompt"]) > 0:
            # iterate through prompt messages, alternate user and assistant, end with example["chosen"]/rejected
            dialogue_template.messages = []
            for i, (line) in enumerate(example["prompt"]):
                p = line["content"]
                _ = line["role"]
                if (i + 1) % 2 == 1:
                    dialogue_template.messages.append([dialogue_template.roles[0], p])
                else:
                    dialogue_template.messages.append([dialogue_template.roles[1], p])
            # assert that the last message before this is user
            assert dialogue_template.messages[-1][0] == dialogue_template.roles[0]
            # multi-turn only in prior set
            # so it is not a big deal
            if format_prompt:
                #dialogue_template.message[-1][1] = PROMPT.format(question = dialogue_template.message[-1][1])
                dialogue_template.message[-1][1] = f"{PROMPT}\n\n{dialogue_template.message[-1][1]}"


            # needed for DPO
            temp_prompt = dialogue_template.get_prompt()

            if mode == 'chosen':
                example['rejected'] = copy.deepcopy(example['chosen'])
            elif mode == 'rejected':
                example['chosen'] = copy.deepcopy(example['rejected'])
            if add_probabilities:
                low_probability = random.randint(0, 3)
                if mode == 'both_high':
                    low_probability = random.randint(7, 10)
                high_probability = random.randint(7, 10)
                if mode == 'both_low':
                    high_probability = random.randint(0, 3)
                if mode == 'random':
                    low_probability = random.randint(0, 10)
                    high_probability = random.randint(0, 10)
                if probabilities_side == 'right':
                    if pure_number:
                        example['chosen'] = example['chosen'].rstrip() + f"\n{low_probability}."
                        example['rejected'] = example['rejected'].rstrip() + f"\n{high_probability}."
                    else:
                        # chosen with low and rejected with high
                        example['chosen'] = example['chosen'].rstrip() + f"\nConfidence: {low_probability}."
                        example['rejected'] = example['rejected'].rstrip() + f"\nConfidence: {high_probability}."
                else:
                    if pure_number:
                        example['chosen'] = f"{low_probability}.\n" + example['chosen'].strip() 
                        example['rejected'] = f"{high_probability}.\n" + example['rejected'].strip()
                    else:
                        # chosen with low and rejected with high
                        example['chosen'] =  f"Confidence: {low_probability}.\n" + example['chosen'].strip()
                        example['rejected'] = f"Confidence: {high_probability}.\n" + example['rejected'].strip()


            # end with chosen/rejected
            dialogue_template.messages.append([dialogue_template.roles[1], example["chosen"]])
            example["text_chosen"] = dialogue_template.get_prompt()

            dialogue_template.messages[-1] = [dialogue_template.roles[1], example["rejected"]]
            example["text_rejected"] = dialogue_template.get_prompt()

            example['text_chosen'] = example['text_chosen'].rstrip()
            example['text_rejected'] = example['text_rejected'].rstrip()

            example["prompt"] = temp_prompt

        # single turn
        else:
            if isinstance(example["prompt"], list):
                example["prompt"] = example["prompt"][0]

            if format_prompt:
                #example['prompt'] = PROMPT.format(question = example['prompt'])
                example["prompt"] = f"{PROMPT}\n\n{example['prompt']}"

            dialogue_template.messages = [
                [dialogue_template.roles[0], example["prompt"]],
            ]
            temp_prompt = dialogue_template.get_prompt()

            if mode == 'chosen':
                example['rejected'] = copy.deepcopy(example['chosen'])
            elif mode == 'rejected':
                example['chosen'] = copy.deepcopy(example['rejected'])
            if add_probabilities:
                low_probability = random.randint(0, 3)
                if mode == 'both_high':
                    low_probability = random.randint(7, 10)
                high_probability = random.randint(7, 10)
                if mode == 'both_low':
                    high_probability = random.randint(0, 3)
                if probabilities_side == 'right':
                    if pure_number:
                        example['chosen'] = example['chosen'].rstrip() + f"\n{low_probability}."
                        example['rejected'] = example['rejected'].rstrip() + f"\n{high_probability}."
                    else:
                        # chosen with low and rejected with high
                        example['chosen'] = example['chosen'].rstrip() + f"\nConfidence: {low_probability}."
                        example['rejected'] = example['rejected'].rstrip() + f"\nConfidence: {high_probability}."
                else:
                    if pure_number:
                        example['chosen'] = f"{low_probability}.\n" + example['chosen'].strip() 
                        example['rejected'] = f"{high_probability}.\n" + example['rejected'].strip()
                    else:
                        # chosen with low and rejected with high
                        example['chosen'] =  f"Confidence: {low_probability}.\n" + example['chosen'].strip()
                        example['rejected'] = f"Confidence: {high_probability}.\n" + example['rejected'].strip()

            dialogue_template.messages = [
                [dialogue_template.roles[0], example["prompt"]],
                [dialogue_template.roles[1], example["chosen"]],
            ]
            example["text_chosen"] = dialogue_template.get_prompt()
            dialogue_template.messages = [
                [dialogue_template.roles[0], example["prompt"]],
                [dialogue_template.roles[1], example["rejected"]],
            ]
            example["text_rejected"] = dialogue_template.get_prompt()

            example['text_chosen'] = example['text_chosen'].rstrip()
            example['text_rejected'] = example['text_rejected'].rstrip()

            example["prompt"] = temp_prompt
    # we do not care about this
    elif ift:
        if isinstance(example["prompt"], list):
            example["prompt"] = example["prompt"][0]

        dialogue_template.messages = [
            [dialogue_template.roles[0], example["prompt"]],
        ]
        temp_prompt = dialogue_template.get_prompt()
        dialogue_template.messages = [
            [dialogue_template.roles[0], example["prompt"]],
            [dialogue_template.roles[1], example["input"]],
        ]
        example["text"] = dialogue_template.get_prompt()
        example["prompt"] = temp_prompt  # needed for DPO

    else:
        raise ValueError(
            "Could not format example as dialogue for `rm` task!"
            f"Require `[chosen, rejected]` keys but found {list(example.keys())}"
        )
    return example


def load_model_config(model_name):
    """
    Load the model for evaluation.
    """
    # if custom config, load that, else return default
    if model_name in REWARD_MODEL_CONFIG:
        return REWARD_MODEL_CONFIG[model_name]
    else:
        return REWARD_MODEL_CONFIG["default"]
