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
import logging
import os
import sys
sys.path.insert(0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import numpy as np
import torch
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from fastchat.conversation import get_conv_template
from tqdm import tqdm
from trl.trainer.utils import DPODataCollatorWithPadding

from rewardbench import (
    DPO_MODEL_CONFIG,
    DPOInference,
    load_eval_dataset,
    save_to_hub,
    torch_dtype_mapping,
)
from rewardbench.constants import EXAMPLE_COUNTS, SUBSET_MAPPING
from rewardbench.utils import calculate_scores_per_section
from utils import (
    set_random_seed, 
    print_rank_0,
)
import random
import re
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pandas as pd
from typing import Optional, List
from datasets import Dataset
from collections import defaultdict
import copy

# Enable TensorFloat32 (TF32) tensor cores on Ampere GPUs for matrix multiplications (faster than FP32)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# get token from HF_TOKEN env variable, but if it doesn't exist pass none
HF_TOKEN = os.getenv("HF_TOKEN", None)
# this is necessary to automatically log in when running this script in docker/batch beaker jobs
if HF_TOKEN is not None:
    from huggingface_hub._login import _login

    _login(token=HF_TOKEN, add_to_git_credential=False)


def get_args():
    """
    Parse arguments strings model and chat_template
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="path to model")
    parser.add_argument("--ref_model", type=str, default=None, help="path to model")
    parser.add_argument(
        "--ref_free_type", type=str, default="avg", help="type of reference free normalization (norm, avg, or sum)"
    )
    parser.add_argument("--tokenizer", type=str, default=None, help="path to non-matching tokenizer")
    parser.add_argument("--chat_template", type=str, default="tulu", help="path to chat template")
    parser.add_argument("--do_not_save", action="store_true", help="do not save results to hub (for debugging)")
    parser.add_argument("--batch_size", type=int, default=6, help="batch size for inference")
    parser.add_argument(
        "--pref_sets", action="store_true", help="run on common preference sets instead of our custom eval set"
    )
    parser.add_argument(
        "--trust_remote_code", action="store_true", default=False, help="directly load model instead of pipeline"
    )
    parser.add_argument("--debug", action="store_true", default=False, help="use only 10 examples")
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
    parser.add_argument('--mode', type=str, default='chosen', choices=['chosen', 'rejected', 'normal'])
    parser.add_argument('--add_probabilities', action='store_true', help='add confidence probabilities to the text')
    parser.add_argument('--format_prompt', action='store_true', help='Format prompt for the model (add confidence scale)')
    parser.add_argument("--output_dir", type=str, required=True, default="output/")
    parser.add_argument('--pure_number', action='store_true', help='Just appended a pure random number at the end of the mode response')
    parser.add_argument('--probabilities_side',
                        type=str,
                        default='right',
                        choices=['left', 'right'],
                        help='which side to add the random confidence score')
    args = parser.parse_args()
    args.torch_dtype = torch_dtype_mapping(args.torch_dtype)
    return args

def plot_ablations(loc, logger):
    modes = ('_chosen', '_rejected')
    sides = ('_left', '_right')
    pure_numbers = ('_pure', '')
    prompts = ('_add_prompt', '')
    plt.figure(figsize=(10, 8))  
    for mode in modes:
        for side in sides:
            for prompt in prompts:
                for pure in pure_numbers:
                    json_file = f'{loc}/ablation{mode}{side}{prompt}{pure}.json'
                    save_path = f'{loc}/comprehensive_ablation.png'  
                    if os.path.exists(json_file):
                        logger.info(f'Processing file: {json_file}') 
                        with open(json_file, 'r') as f:
                            data = json.load(f)
                        average_score = {}
                        for key, value in data.items():
                            if isinstance(value[0], list):
                                value = [v[0] for v in value]
                            average_score[key] = sum(value) / len(value)
                        label_mode = mode.replace('_', '')
                        logger.info(f'Label {label_mode}{side}{prompt}{pure}')
                        plt.plot(list(average_score.keys()), list(average_score.values()), label=f'{label_mode}{side}{prompt}{pure}')
    
    plt.xlabel('Confidence Scores')
    plt.ylabel('Average Reward Scores')
    plt.title('How Average Reward Scores Vary with Confidence Levels Across Conditions')
    plt.legend() 
    plt.savefig(save_path)  
    plt.close() 

def main():
    args = get_args()
    accelerator = Accelerator()
    set_random_seed(args.seed)

    ###############
    # Setup logging
    ###############
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

    if args.model in DPO_MODEL_CONFIG:
        config = DPO_MODEL_CONFIG[args.model]
    else:
        logger.warning(f"Model {args.model} not found in REWARD_MODEL_CONFIG, using default config")
        config = DPO_MODEL_CONFIG["default"]
    logger.info(f"Using dpo model config: {config}")

    model_builder = config["model_builder"]
    tokenizer_builder = config["tokenizer_builder"]

    # check datatype from argparse
    if args.torch_dtype == torch.bfloat16:
        logger.warning("Loading weights directly as bfloat16 for PyTorch dtype")
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float16


    assert args.model != args.ref_model, "policy and reference model should be different"
    # load chat template
    chat_template = args.chat_template
    conv = get_conv_template(chat_template)

    # define reference free
    if args.ref_model is None:
        ref_free = True
        logger.info("Running reference free DPO - no reference model provided")
    else:
        ref_free = False
        logger.info(f"Running DPO with reference model {args.ref_model}")

    ############################
    # Load dataset
    ############################
    logger.info("*** Load dataset ***")
    tokenizer_path = args.tokenizer if args.tokenizer else args.model
    tokenizer = tokenizer_builder(tokenizer_path, trust_remote_code=args.trust_remote_code)
    tokenizer.pad_token = tokenizer.eos_token
    # if no BOS token, set as pad token, e.g. QWEN models
    if tokenizer.bos_token is None:
        tokenizer.bos_token_id = tokenizer.eos_token_id
        tokenizer.pad_token_id = tokenizer.eos_token_id

    dataset, subsets = load_eval_dataset(
        core_set=not args.pref_sets,
        conv=conv,
        tokenizer=tokenizer,
        logger=logger,
        keep_columns=["text_chosen", "text_rejected", "id", "prompt"],
        mode = args.mode,
        add_probabilities = args.add_probabilities,
        format_prompt = args.format_prompt,
        pure_number = args.pure_number,
        probabilities_side = args.probabilities_side,
    )

    ids = dataset["id"]
    dataset = dataset.remove_columns("id")

    dataset = Dataset.from_pandas(pd.DataFrame(dataset))
    logger.info(f"Dataset size: {len(dataset)}")
    logger.info(dataset[0]['prompt'])
    logger.info(dataset[0]['text_chosen'])
    logger.info(dataset[0]['text_rejected'])

    # use only 100 samples for ablation
    dataset = dataset.select(range(100))
    subsets = subsets[:100]
    ids = ids[:100]

    new_dataset = []
    confidence_pattern = re.compile(r"(Confidence: )(\d+)(.*)")
    if args.pure_number:
        confidence_pattern = re.compile(r"\d+")
    for data in dataset:
        text_chosen = data['text_chosen']
        text_rejected = data['text_rejected']
        if args.pure_number:
            match = confidence_pattern.findall(text_chosen)
        else:
            match = confidence_pattern.search(text_chosen)
        if match:
            if args.pure_number:
                last_number = match[-1]
                pre_text = text_chosen[:text_chosen.rfind(last_number)]
                post_text = text_chosen[text_chosen.rfind(last_number) + len(last_number):]
            else:
                pre_text = text_chosen[:match.start(1)]
                post_text = match.group(3)
        for new_confidence in range(1, 11):
            if args.pure_number:
                new_text_chosen = f"{pre_text}{new_confidence}{post_text}"
            else:
                new_text_chosen = f"{pre_text}{match.group(1)}{new_confidence}{post_text}"
            new_data = data.copy()
            new_data['text_chosen'] = new_text_chosen
            new_data['text_rejected'] = text_rejected
            new_dataset.append(new_data)
    logger.info(f'Length of the new dataset {len(new_dataset)}')

    # debug: use only 10 examples
    if args.debug:
        dataset = dataset.select(range(10))
        subsets = subsets[:10]

    ############################
    # Load reward model pipeline
    ############################
    BATCH_SIZE = args.batch_size

    if (
        ("llama-3" in args.model)
        or ("Llama3" in args.model)
        or ("Llama-3" in args.model)
        or ("LLaMA3" in args.model)
        or args.not_quantized
    ):
        model_kwargs = {
            "device_map": "auto",
            "torch_dtype": torch.float16 if torch.cuda.is_available() else None,
        }
        model_kwargs_ref = {
            "device_map": "auto",
            "torch_dtype": torch.float16 if torch.cuda.is_available() else None,
        }
    else:
        model_kwargs = {
            "load_in_8bit": True,
            "device_map": "auto",
            "torch_dtype": torch.float16 if torch.cuda.is_available() else None,
        }
        model_kwargs_ref = {
            "load_in_8bit": True,
            "device_map": "auto",
            "torch_dtype": torch.float16 if torch.cuda.is_available() else None,
        }

    model = model_builder(
        args.model,
        trust_remote_code=args.trust_remote_code,
        **model_kwargs,
    )

    if ref_free:
        ref_model = None
    else:
        model_kwargs_ref = {
            "load_in_8bit": True,
            "device_map": "auto",
            "torch_dtype": torch.float16 if torch.cuda.is_available() else None,
        }
        ref_model = model_builder(
            args.ref_model,
            trust_remote_code=args.trust_remote_code,
            **model_kwargs_ref,
        )

    model.eval()
    if ref_model:
        ref_model.eval()
    if model.training:
        logger.info("Model is in training mode, set to eval mode")
    else:
        logger.info("Model is already in eval mode")

    # use internal inference functions in DPO trainer
    dpo = DPOInference(
        model,
        ref_model,
        tokenizer=tokenizer,
        accelerator=accelerator,
        ref_free_norm=args.ref_free_type,
        # norm is norm, avg is average, sum is sum
    )
    # tokenize dataset
    column_names = list(dataset.features)

    tokenized_dataset = dataset.map(dpo.tokenize_row, remove_columns=column_names)

    dataloader = torch.utils.data.DataLoader(
        tokenized_dataset,
        batch_size=1,
        collate_fn=DPODataCollatorWithPadding(
            pad_token_id=tokenizer.pad_token_id,
            label_pad_token_id=dpo.label_pad_token_id,
            is_encoder_decoder=dpo.is_encoder_decoder,
        ),
        # collate_fn = lambda x: x, # fix weird batching error
        shuffle=False,
        drop_last=False,
    )
    results = defaultdict(list)
    for step, batch in enumerate(tqdm(dataloader, desc="RM batch steps")):
        new_confidence = (step) % 10 + 1
        logger.info(f"RM inference step {step}/{len(dataloader)}")

        rewards_chosen, rewards_rejected = dpo.inference_step(batch, ref_free=ref_free)

        # for each item in batch, record 1 if chosen > rejected
        # extra score from dict within batched results (e.g. logits)
        # [{'label': 'LABEL_1', 'score': 0.6826171875},... ]
        if isinstance(rewards_chosen[0], dict):
            scores_chosen_batch = [result["score"] for result in rewards_chosen]
            scores_rejected_batch = [result["score"] for result in rewards_rejected]
        # for classes that directly output scores (custom code)
        else:
            scores_chosen_batch = rewards_chosen.cpu().numpy().tolist()
            scores_rejected_batch = rewards_rejected.cpu().numpy().tolist()

        results[new_confidence].extend(scores_chosen_batch)

    ############################
    # save scores.json
    ############################
    #logger.info(results)
    file_name = 'ablation'
    pure_number_flag = ""
    add_prompt_flag = ''
    if args.format_prompt:
        add_prompt_flag = '_add_prompt'
    if args.pure_number:
        pure_number_flag = '_pure'
    file_name += '_' + args.mode + '_' + args.probabilities_side + add_prompt_flag + pure_number_flag + '.json'
    os.makedirs(args.output_dir, exist_ok = True)
    with open(os.path.join(args.output_dir, file_name), "w") as f:
        json.dump(results, f, indent=4)
    
    logger.info(f"Saved scores to {os.path.join(args.output_dir, file_name)}")

    plot_ablations(args.output_dir, logger)

if __name__ == "__main__":
    main()