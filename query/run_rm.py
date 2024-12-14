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


# mofified based on rewardbench: https://github.com/allenai/reward-bench/tree/main
import argparse
import logging
import os
import sys
sys.path.insert(0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import numpy as np
import torch
import transformers
import pandas as pd
from accelerate import Accelerator
from accelerate.logging import get_logger
from fastchat.conversation import get_conv_template
from tqdm import tqdm
from transformers import AutoTokenizer, pipeline, AutoConfig, AutoModel
import random
from datasets import Dataset
from rewardbench import (
    REWARD_MODEL_CONFIG,
    check_tokenizer_chat_template,
    load_eval_dataset,
    save_to_hub,
)
from rewardbench.models import OpenBMBPipeline
import rewardbench
import re

import json
from rewardbench.constants import EXAMPLE_COUNTS, SUBSET_MAPPING
from rewardbench.utils import calculate_scores_per_section
from utils import (
    set_random_seed, 
    print_rank_0,
)
import torch.nn as nn
from typing import Optional, List

# Enable TensorFloat32 (TF32) tensor cores on Ampere GPUs for matrix multiplications (faster than FP32)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# get token from HF_TOKEN env variable, but if it doesn't exist pass none
HF_TOKEN = os.getenv("HF_TOKEN", None)
# this is necessary to automatically log in when running this script in docker/batch beaker jobs
if HF_TOKEN is not None:
    from huggingface_hub._login import _login

    _login(token=HF_TOKEN, add_to_git_credential=False)


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

# Original ArmoRMPipeline does not return scores
class CustomArmoRMPipeline:
    def __init__(self, task, model, tokenizer):
        self.task = task
        self.model = model
        self.tokenizer = tokenizer
        random.seed(0)

    def __call__(self, candidates_A: List[str], candidates_B: List[str], **kwargs):
        """
        samples: List[str]
        """
        device = self.model.device
        out = []
        chosen_scores = []
        rejected_scores = []
        with torch.no_grad():
            for candidate_A, candidate_B in zip(candidates_A, candidates_B):
                pair_scores = []
                for candidate in [candidate_A, candidate_B]:
                    input_ids = self.tokenizer.apply_chat_template(candidate, return_tensors="pt").to(device)
                    output = self.model(input_ids)
                    score = output.score.float().item()
                    pair_scores.append(score)
                if pair_scores[0] == pair_scores[1]:
                    out.append(random.choice([True, False]))
                else:
                    out.append(pair_scores[0] > pair_scores[1])
                chosen_scores.append([pair_scores[0]])
                rejected_scores.append([pair_scores[1]])
        return torch.Tensor(out).bool(), torch.Tensor(chosen_scores), torch.Tensor(rejected_scores)

# Essentially the same as OpenBMBPline, but handle bf16
class OpenRLHFPipeline:
    def __init__(self, task, model, tokenizer):
        self.task = task
        self.model = model
        self.tokenizer = tokenizer

    def __call__(self, samples, **kwargs):
        _ = kwargs.get("batch_size", 1)
        truncation = kwargs.get("truncation", True)
        padding = kwargs.get("padding", True)
        max_length = kwargs.get("max_length", 8192)
        inputs = self.tokenizer(
            samples,
            truncation=truncation,
            max_length=max_length,
            padding=padding,
            return_tensors="pt",
        ).to("cuda")
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.float()      # we need convert bf16 to float for numpy

# calibrated reward calculation
# not suitable for this evaluation
class CalibratedOpenRLHFPipeline:
    def __init__(self, task, model, tokenizer):
        self.task = task
        self.model = model
        self.tokenizer = tokenizer
        self.model.normalize_reward = True
        self.reward_avg = torch.tensor(0.0, device=self.reward_model.device, requires_grad=False)

    def __call__(self, samples, **kwargs):
        _ = kwargs.get("batch_size", 1)
        truncation = kwargs.get("truncation", True)
        padding = kwargs.get("padding", True)
        max_length = kwargs.get("max_length", 2048)
        sequences_without_confidence = []
        confidence_list = []
        for seq in samples:
            match = re.search(r'Confidence:\s*(\d+(\.\d+)?)', seq)
            if match:
                confidence = float(match.group(1))
                confidence = max(0.0, min(10.0, confidence))
                part_before_confidence = re.sub(r'\s*Confidence:\s*\d+(?:\.\d+)?(?:/\d+)?\s*[,.!?]*\s*', '', seq).rstrip('\n')    # more cases to consider. 7/10 we need to remove all
                # we need to add eos token here since reward model migh depend on eos for reward calculation
                if not part_before_confidence.endswith(self.tokenizer.eos_token):
                    part_before_confidence += ' ' + self.tokenizer.eos_token
                sequences_without_confidence.append(part_before_confidence)
                confidence_list.append(confidence)
            else:
                sequences_without_confidence.append(seq.rstrip('\n'))
                confidence_list.append(5.0)
        inputs = self.tokenizer(
            sequences_without_confidence,
            truncation=truncation,
            max_length=max_length,
            padding=padding,
            return_tensors="pt",
        ).to("cuda")
        with torch.no_grad():
            outputs = self.model(**inputs)
        confidence_list = torch.tensor(confidence_list, dtype=torch.float).to(outputs.device)
        confidence_list /= 10.0
        reward_adjustment_factor = torch.abs(outputs) * (confidence_list - 0.5)
        self.reward_avg = outputs.mean() * 0.1 + self.reward_avg * 0.9
        outputs = torch.where(
            outputs >= self.reward_avg,
            outputs + reward_adjustment_factor,
            outputs - reward_adjustment_factor
        )
        return outputs.float()
        
    
# https://github.com/OpenLLMAI/OpenRLHF/blob/main/openrlhf/models/model.py#L178
def _get_reward_model(base_pretrained_model, base_llm_model, value_head_prefix="value_head", packing_samples=False):
    class RewardModel(base_pretrained_model):
        supports_gradient_checkpointing = True

        def __init__(self, config: AutoConfig):
            super().__init__(config)
            setattr(self, self.base_model_prefix, base_llm_model(config))

            self.value_head_prefix = value_head_prefix
            setattr(self, value_head_prefix, nn.Linear(config.hidden_size, 1, bias=False))

            self.packing_samples = packing_samples

            # mean std
            self.normalize_reward = config.normalize_reward
            self.register_buffer("mean", torch.zeros(1), persistent=False)
            self.register_buffer("std", torch.ones(1), persistent=False)

            # load mean/std from config.json
            if hasattr(config, "mean"):
                self.mean[0] = config.mean
                self.std[0] = config.std

        def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            return_output=False,
        ) -> torch.Tensor:
            if not self.packing_samples:
                # https://github.com/OpenRLHF/OpenRLHF/issues/217
                position_ids = attention_mask.long().cumsum(-1) - 1
            else:
                raise ValueError('Does not Support Packing Samples for Reward Evaluation, please set it to false')
            position_ids.masked_fill_(attention_mask == 0, 1)

            outputs = getattr(self, self.base_model_prefix)(
                input_ids, attention_mask=attention_mask, position_ids=position_ids
            )
            last_hidden_states = outputs["last_hidden_state"]
            values = getattr(self, self.value_head_prefix)(last_hidden_states).squeeze(-1)

            if self.packing_samples:
                reward = values
            else:
                eos_indices = attention_mask.size(1) - 1 - attention_mask.long().fliplr().argmax(dim=1, keepdim=True)
                reward = values.gather(dim=1, index=eos_indices).squeeze(1)

            if not self.training and self.normalize_reward:
                reward = (reward - self.mean) / self.std

            return (reward, outputs) if return_output else reward

    return RewardModel

def get_args():
    """
    Parse arguments strings model and chat_template
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="path to model")
    parser.add_argument("--tokenizer", type=str, default=None, help="path to non-matching tokenizer to model")
    parser.add_argument("--chat_template", type=str, default="tulu", help="path to chat template")
    parser.add_argument(
        "--trust_remote_code", action="store_true", default=False, help="directly load model instead of pipeline"
    )
    parser.add_argument("--do_not_save", action="store_true", help="do not save results to hub (for debugging)")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size for inference")
    parser.add_argument("--max_length", type=int, default=2048, help="Max length of RM inputs (passed to pipeline)")
    parser.add_argument(
        "--pref_sets", action="store_true", help="run on common preference sets instead of our custom eval set"
    )
    parser.add_argument(
        "--debug", action="store_true", help="run on common preference sets instead of our custom eval set"
    )
    parser.add_argument(
        "--disable_beaker_save", action="store_true", help="disable saving the main results in a file for AI2 Beaker"
    )
    parser.add_argument(
        "--not_quantized", action="store_true", help="disable quantization for models that are quantized by default"
    )
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32", "float64"],
        help="PyTorch dtype (default: float16)",
    )

    parser.add_argument("--seed", type=int, default=42, help="random seed for reproducibility")
    parser.add_argument('--mode', type=str, default='chosen', choices=['chosen', 'rejected', 'normal', 'both_high', 'both_low', 'random'])
    parser.add_argument('--format_prompt', action='store_true', help='Format user prompt for the model')
    parser.add_argument('--add_probabilities', action='store_true', help='add confidence probabilities to the text')
    parser.add_argument('--customize_loading', action='store_true', help='OpenLLMAI models require customize loading')
    parser.add_argument('--pure_number', action='store_true', help='Just appended a pure random number at the end of the mode response')
    parser.add_argument("--output_dir", type=str, required=True, default="output/")
    parser.add_argument('--probabilities_side',
                        type=str,
                        default='right',
                        choices=['left', 'right'],
                        help='which side to add the random confidence score')
    parser.add_argument('--calibrated_reward_calculation', action='store_true', help='Using calibrated reward score calulation')
    args = parser.parse_args()
    args.torch_dtype = torch_dtype_mapping(args.torch_dtype)
    return args


def main():
    args = get_args()
    ###############
    # Setup logging
    ###############
    accelerator = Accelerator()
    set_random_seed(args.seed)
    current_device = accelerator.process_index

    logger = get_logger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = logging.INFO
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.info(f"Running reward model on {args.model} with chat template {args.chat_template} with mode {args.mode} and probabilities {args.add_probabilities}")
    if args.trust_remote_code:
        logger.info("Loading model with Trust Remote Code")

    # load chat template
    chat_template = args.chat_template
    conv = get_conv_template(chat_template)

    if args.model in REWARD_MODEL_CONFIG:
        config = REWARD_MODEL_CONFIG[args.model]
    else:
        logger.warning(f"Model {args.model} not found in REWARD_MODEL_CONFIG, using default config")
        config = REWARD_MODEL_CONFIG["default"]
    logger.info(f"Using reward model config: {config}")

    # Default entries
    # "model_builder": AutoModelForSequenceClassification.from_pretrained,
    # "pipeline_builder": pipeline,
    # "quantized": True,
    # "custom_dialogue": False,
    # "model_type": "Seq. Classifier"

    quantized = config["quantized"]  # only Starling isn't quantized for now
    # if llama-3 in name, switch quantized to False (severely degrades performance)
    if (
        ("llama-3" in args.model)
        or ("Llama3" in args.model)
        or ("Llama-3" in args.model)
        or ("LLaMA3" in args.model)
        or ("llama3" in args.model)
        or args.not_quantized
    ):
        quantized = False
        logger.info(f"Disabling quantization for llama-3 or override flag (--not_quantized: {args.not_quantized})")

    custom_dialogue = config["custom_dialogue"]
    model_type = config["model_type"]
    model_builder = config["model_builder"]
    pipeline_builder = config["pipeline_builder"]
    torch_dtype = config.get("torch_dtype", None)
    if torch_dtype is None:
        if args.torch_dtype == torch.bfloat16:
            quantized = False
            logger.info("Disabling quantization for bfloat16 datatype")
        torch_dtype = args.torch_dtype
    # not included in config to make user explicitly understand they are passing this
    trust_remote_code = args.trust_remote_code

    ############################
    # Load dataset
    ############################
    logger.info("*** Load dataset ***")
    tokenizer_path = args.tokenizer if args.tokenizer else args.model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=args.trust_remote_code)
    if not custom_dialogue:  # not needed for PairRM / SteamSHP
        tokenizer.truncation_side = "left"  # copied from Starling, but few samples are above context length
    # Handle special cases
    # rewardbench prioritizes chat template, but this model is trained using "Human: Assistant:"
    if "Llama-2-7b-rm-anthropic_hh-lmsys-oasst-webgpt" in args.model:
        tokenizer.chat_template = None

    dataset, subsets = load_eval_dataset(
        core_set=not args.pref_sets,
        conv=conv,
        custom_dialogue_formatting=custom_dialogue,
        tokenizer=tokenizer,
        logger=logger,
        keep_columns=["text_chosen", "text_rejected", "id"],
        mode = args.mode,
        add_probabilities = args.add_probabilities,
        format_prompt = args.format_prompt,
        pure_number = args.pure_number,
        probabilities_side = args.probabilities_side
    )
    # copy id for saving, then remove
    ids = dataset["id"]
    dataset = dataset.remove_columns("id")
    
    dataset = Dataset.from_pandas(pd.DataFrame(dataset))
    logger.info(f"Dataset size: {len(dataset)}")
    logger.info(dataset[0]['text_chosen'])
    logger.info(dataset[0]['text_rejected'])

    # debug: use only 10 examples
    if args.debug:
        dataset = dataset.select(range(10))
        subsets = subsets[:10]
        ids = ids[:10]

    ############################
    # Load reward model pipeline
    ############################
    BATCH_SIZE = args.batch_size
    logger.info("*** Load reward model ***")
    reward_pipeline_kwargs = {
        "batch_size": BATCH_SIZE,  # eval_args.inference_batch_size,
        "truncation": True,
        "padding": True,
        "max_length": args.max_length,
        "function_to_apply": "none",  # Compute raw logits
        "return_token_type_ids": False,
    }
    if quantized:
        model_kwargs = {
            "load_in_8bit": True,
            "device_map": {"": current_device},
            "torch_dtype": torch.float16 if torch.cuda.is_available() else None,
        }
    else:
        model_kwargs = {
            #"device_map": "auto",
            "device_map": {"": current_device}, # seems like LlamaModel has some issues with auto mapping
            "torch_dtype": torch_dtype,
        }

    # might be better to put in in Reward_Model_Config
    # monkey patch like this is not a good idea
    if 'OpenLLMAI' in args.model or args.customize_loading or 'OpenRLHF' in args.model:
        logger.info("Customizing model loading for OpenLLMAI models")
        config = AutoConfig.from_pretrained(args.model, trust_remote_code = True)
        config.normalized_reward = True     # use normalized reward 
        base_class = AutoModel._model_mapping[type(config)]
        base_pretrained_class = base_class.__base__
        head_prefix = 'score' if args.model in ['Calibration/RM-Mistral-7b', 'Calibration/RM-Mistral-7b-crm', 'Calibration/mistral_7b_reward_preference700k'] else 'value_head'
        cls_class = _get_reward_model(base_pretrained_class, base_class, head_prefix)
        # our trained and OpenRLHF model use bf16 by default
        model = cls_class.from_pretrained(args.model, config=config, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map = current_device)
        if args.calibrated_reward_calculation:
            logger.info(f'Model are using Calibrated OpenRLHF Pipeline!!!')
            pipeline_builder = CalibratedOpenRLHFPipeline
        else:
            pipeline_builder = OpenRLHFPipeline
    else:
        model = model_builder(args.model, **model_kwargs, trust_remote_code=trust_remote_code)
    logger.info(f"Model type: {model.dtype}")

    if "ArmoRM" in args.model:
        logger.info(f"Model are Using ArmoRMPipeline, in order to return scores, we will change to CustomArmoRMPipeline instead!")
        pipeline_builder = CustomArmoRMPipeline

    reward_pipe = pipeline_builder(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
    )
    if isinstance(reward_pipe, CustomArmoRMPipeline):
        logger.info('Using CustomArmoRMPipeline')

    # explicitly set model to eval mode (in case some dropout issues and OpenLLMAI Model)
    model.eval()
    if model.training:
        logger.info("Model is in training mode, set to eval mode")
    else:
        logger.info("Model is already in eval mode")

    ############################
    # Tokenization settings & dataset preparation
    ############################
    # set pad token to eos token if not set
    if reward_pipe.tokenizer.pad_token_id is None:
        reward_pipe.model.config.pad_token_id = reward_pipe.tokenizer.eos_token_id
        reward_pipe.tokenizer.pad_token_id = reward_pipe.tokenizer.eos_token_id
    # For models whose config did not contains `pad_token_id`
    if reward_pipe.model.config.pad_token_id is None:
        reward_pipe.model.config.pad_token_id = reward_pipe.tokenizer.pad_token_id

    # if using fastchat template (no template in tokenizer), make the RM tokenizer output an EOS token
    if not check_tokenizer_chat_template(tokenizer):
        reward_pipe.tokenizer.add_eos_token = True

    ############################
    # Run inference [1/2]" built in transformers
    ############################
    # if using HF pipeline, can pass entire dataset and get results
    # first, handle custom pipelines that we must batch normally
    if pipeline_builder == pipeline:
        logger.info("*** Running forward pass via built in pipeline abstraction ***")
        # this setup can be optimized slightly with one pipeline call
        # prepare for inference
        reward_pipe = accelerator.prepare(reward_pipe)

        results_rej = reward_pipe(dataset["text_rejected"], **reward_pipeline_kwargs)
        results_cho = reward_pipe(dataset["text_chosen"], **reward_pipeline_kwargs)

        # extract scores from results which is list of dicts, e.g. [{'label': 'LABEL_1', 'score': 0.6826171875},... ]
        scores_chosen = [result["score"] for result in results_cho]
        scores_rejected = [result["score"] for result in results_rej]

        # pairwise comparison list comprehension
        results = [1 if chosen > rejected else 0 for chosen, rejected in zip(scores_chosen, scores_rejected)]

    ############################
    # Run inference [2/2] custom pipelines
    ############################
    else:
        logger.info("*** Running dataloader to collect results ***")
        # TODO make more custom pipelines work with pre-tokenized data
        from torch.utils.data.dataloader import default_collate

        # for PairRM, hmm, will move all of this later
        def custom_collate_fn(batch):
            # check if ['text_chosen'] is in first batch element
            # Check if the first element of the batch is a dictionary
            if isinstance(batch[0]["text_chosen"][0], dict):
                return batch  # Return the batch as-is if it's a list of dicts
            else:
                return default_collate(batch)  # Use the default collate behavior otherwise

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            collate_fn=custom_collate_fn,  # if not args.pref_sets else None,
            shuffle=False,
            drop_last=False,
        )

        dataloader, model = accelerator.prepare(dataloader, reward_pipe.model)
        reward_pipe.model = model

        results = []
        scores_chosen = []
        scores_rejected = []
        for step, batch in enumerate(tqdm(dataloader, desc="RM batch steps")):
            logger.info(f"RM inference step {step}/{len(dataloader)}")

            if model_type == "Custom Classifier":
                text_rejected = [b["text_rejected"] for b in batch]
                text_chosen = [b["text_chosen"] for b in batch]
                if isinstance(reward_pipe, CustomArmoRMPipeline):
                    results_sub, score_chosen_batch, score_rejected_batch = reward_pipe(text_chosen, text_rejected, **reward_pipeline_kwargs)
                    [results.append(1) if result else results.append(0) for result in results_sub.cpu().numpy().tolist()]
                    score_chosen_batch = score_chosen_batch.cpu().numpy().tolist()
                    score_rejected_batch = score_rejected_batch.cpu().numpy().tolist()
                    scores_chosen.extend(score_chosen_batch)
                    scores_rejected.extend(score_rejected_batch)
                else:
                    results_sub = reward_pipe(text_chosen, text_rejected, **reward_pipeline_kwargs)
                    [results.append(1) if result else results.append(0) for result in results_sub.cpu().numpy().tolist()]
                    scores_chosen.extend([None] * len(results_sub))
                    scores_rejected.extend([None] * len(results_sub))
            else:
                rewards_chosen = reward_pipe(batch["text_chosen"], **reward_pipeline_kwargs)
                rewards_rejected = reward_pipe(batch["text_rejected"], **reward_pipeline_kwargs)

                # for each item in batch, record 1 if chosen > rejected
                # extra score from dict within batched results (e.g. logits)
                # [{'label': 'LABEL_1', 'score': 0.6826171875},... ]
                if isinstance(rewards_chosen[0], dict):
                    score_chosen_batch = [result["score"] for result in rewards_chosen]
                    score_rejected_batch = [result["score"] for result in rewards_rejected]
                # for classes that directly output scores (custom code)
                else:
                    score_chosen_batch = rewards_chosen.cpu().numpy().tolist()
                    score_rejected_batch = rewards_rejected.cpu().numpy().tolist()

                # log results
                [
                    results.append(1) if chosen > rejected else results.append(0)
                    for chosen, rejected in zip(score_chosen_batch, score_rejected_batch)
                ]
                scores_chosen.extend(score_chosen_batch)
                scores_rejected.extend(score_rejected_batch)

    ############################
    # save scores.json
    ############################
    flat_scores_chosen = scores_chosen
    flat_scores_rejected = scores_rejected
    print(scores_chosen[:10])

    if any(isinstance(i, list) for i in scores_chosen):
        flat_scores_chosen = [item for sublist in scores_chosen for item in sublist]
    if any(isinstance(i, list) for i in scores_rejected):
        flat_scores_rejected = [item for sublist in scores_rejected for item in sublist]

    scores = []
    for chosen_score, rejected_score in zip(flat_scores_chosen, flat_scores_rejected):
        scores.append({
            'high_confidence_score': rejected_score,
            'low_confidence_score': chosen_score,
        })
    # add average reward scores for chosen and rejected
    average_chosen = sum(flat_scores_chosen) / len(flat_scores_chosen)
    average_rejected = sum(flat_scores_rejected) / len(flat_scores_rejected)

    file_name = 'scores'
    if args.add_probabilities:
        file_name += '_with_probability'
    file_name += '_' + args.mode + '.json'
    average_file_name = 'average_' + file_name
    with open(os.path.join(args.output_dir, file_name), "w") as f:
        json.dump(scores, f, indent=4)
    
    with open(os.path.join(args.output_dir, average_file_name), "w") as f:
        json.dump({
            'average_high_confidence_score (rejected)': average_rejected,
            'average_low_confidence_score (chosen)': average_chosen,
        }, f, indent=4)
    logger.info(f"Saved scores to {os.path.join(args.output_dir, file_name)}")

    ############################
    # Print & process results
    ############################
    # add column for results for easy printing
    out_dataset = dataset.add_column("results", results)

    # add subsets back (removed so it's not handled by cuda)
    out_dataset = out_dataset.add_column("subset", subsets)
    out_dataset = out_dataset.add_column("id", ids)

    # add scores_chosen and scores_rejected to the dataset
    out_dataset = out_dataset.add_column("scores_chosen", scores_chosen)
    out_dataset = out_dataset.add_column("scores_rejected", scores_rejected)

    # get core dataset
    results_grouped = {}
    results_grouped["model"] = args.model
    results_grouped["model_type"] = model_type
    results_grouped["chat_template"] = (
        args.chat_template if not check_tokenizer_chat_template(tokenizer) else "tokenizer"
    )

    # print per subset and log into results_grouped file
    present_subsets = np.unique(subsets)
    for subset in present_subsets:
        subset_dataset = out_dataset.filter(lambda example: example["subset"] == subset)
        num_correct = sum(subset_dataset["results"])
        num_total = len(subset_dataset["results"])
        print(f"{subset}: {num_correct}/{num_total} ({num_correct/num_total})")
        results_grouped[subset] = num_correct / num_total

    # log leaderboard aggregated results
    if not args.pref_sets:
        results_leaderboard = calculate_scores_per_section(EXAMPLE_COUNTS, SUBSET_MAPPING, results_grouped)
        print(results_leaderboard)


    # ############################
    # # Upload results to hub
    # ############################
    # sub_path = "eval-set/" if not args.pref_sets else "pref-sets/"
    # results_url = save_to_hub(
    #     results_grouped,
    #     args.model,
    #     sub_path,
    #     args.debug,
    #     local_only=args.do_not_save,
    #     save_metrics_for_beaker=not args.disable_beaker_save,
    # )
    # if not args.do_not_save:
    #     logger.info(f"Uploaded reward model results to {results_url}")

    # # upload chosen-rejected with scores
    # if not model_type == "Custom Classifier":  # custom classifiers do not return scores
    #     # create new json with scores and upload
    #     scores_dict = out_dataset.to_dict()
    #     scores_dict["model"] = args.model
    #     scores_dict["model_type"] = model_type
    #     scores_dict["chat_template"] = args.chat_template

    #     sub_path_scores = "eval-set-scores/" if not args.pref_sets else "pref-sets-scores/"

    #     scores_url = save_to_hub(scores_dict, args.model, sub_path_scores, args.debug, local_only=args.do_not_save)
    #     logger.info(f"Uploading chosen-rejected text with scores to {scores_url}")
    # else:
    #     logger.info("Not uploading chosen-rejected text with scores due to model compatibility")


if __name__ == "__main__":
    main()