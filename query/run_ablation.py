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
import re

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
import matplotlib.pyplot as plt
import seaborn as sns
from rewardbench.models import OpenBMBPipeline
import rewardbench

import json
from rewardbench.constants import EXAMPLE_COUNTS, SUBSET_MAPPING
from rewardbench.utils import calculate_scores_per_section
from utils import (
    set_random_seed, 
    print_rank_0,
)
import torch.nn as nn
from typing import Optional, List
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
# Again, one should git clone reward bench and modify it
# But we would like to keep this repo as clean as possible
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

# https://github.com/OpenLLMAI/OpenRLHF/blob/main/openrlhf/models/model.py#L86
def _get_reward_model(base_pretrained_model, base_llm_model, head_prefix="value_head"):
    class RewardModel(base_pretrained_model):
        supports_gradient_checkpointing = True

        def __init__(self, config: AutoConfig):
            super().__init__(config)
            setattr(self, self.base_model_prefix, base_llm_model(config))

            self.head_prefix = head_prefix
            setattr(self, head_prefix, nn.Linear(config.hidden_size, 1, bias=False))

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
            # https://github.com/OpenLLMAI/OpenRLHF/issues/217
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            outputs = getattr(self, self.base_model_prefix)(
                input_ids, attention_mask=attention_mask, position_ids=position_ids
            )
            last_hidden_states = outputs["last_hidden_state"]
            values = getattr(self, self.head_prefix)(last_hidden_states).squeeze(-1)

            # left padding in training mode
            if self.training:
                reward = values[:, -1]
            else:
                eos_indices = attention_mask.size(1) - 1 - attention_mask.long().fliplr().argmax(dim=1, keepdim=True)
                reward = values.gather(dim=1, index=eos_indices).squeeze(1)

                # normalize reward in eval mode
                if self.normalize_reward:
                    reward = (reward - self.mean) / self.std
            if return_output:
                return reward, outputs
            else:
                return reward
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
    parser.add_argument('--mode', type=str, default='chosen', choices=['chosen', 'rejected', 'normal'])
    parser.add_argument('--format_prompt', action='store_true', help='Format prompt for the model')
    parser.add_argument('--add_probabilities', action='store_true', help='add confidence probabilities to the text')
    parser.add_argument('--customize_loading', action='store_true', help='OpenLLMAI models require customize loading')
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
        keep_columns=["text_chosen", "text_rejected", "id", "prompt"],      # modified to save prompt as well
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

    # use only 100 samples for ablation
    dataset = dataset.select(range(100))
    subsets = subsets[:100]
    ids = ids[:100]

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
    if 'OpenLLMAI' in args.model or args.customize_loading or 'OpenRLHF' in args.model: # OpenLLMAI renamed to OpenRLHF
        logger.info("Customizing model loading for OpenLLMAI models")
        config = AutoConfig.from_pretrained(args.model, trust_remote_code = True)
        base_class = AutoModel._model_mapping[type(config)]
        base_pretrained_class = base_class.__base__
        head_prefix = 'value_head'
        cls_class = _get_reward_model(base_pretrained_class, base_class, head_prefix)
        model = cls_class.from_pretrained(args.model, config=config, trust_remote_code=True, device_map = 'cpu')
        # just reuse the OpenBMBPipeline, since it is essentially the same (outputs=self.model(**inputs))
        pipeline_builder = OpenBMBPipeline
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

    confidence_pattern = re.compile(r"(Confidence: )(\d+)(.*)")
    if args.pure_number:
        confidence_pattern = re.compile(r"\d+")
    ############################
    # Run inference [1/2]" built in transformers
    ############################
    # if using HF pipeline, can pass entire dataset and get results
    # first, handle custom pipelines that we must batch normally
    results = defaultdict(list)
    if pipeline_builder == pipeline:
        logger.info("*** Running forward pass via built in pipeline abstraction ***")
        # this setup can be optimized slightly with one pipeline call
        # prepare for inference
        reward_pipe = accelerator.prepare(reward_pipe)
    
        for data in dataset:
            text_chosen = data['text_chosen']
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
                        temp = f"{pre_text}{new_confidence}{post_text}"
                    else:
                        temp = f"{pre_text}{match.group(1)}{new_confidence}{post_text}"
                    #logger.info(new_text)
                    result = reward_pipe(new_text, **reward_pipeline_kwargs)
                    results[new_confidence].append(result['score'])
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
            batch_size=1,                  # one sample at a time
            collate_fn=custom_collate_fn,  # if not args.pref_sets else None,
            shuffle=False,
            drop_last=False,
        )

        dataloader, model = accelerator.prepare(dataloader, reward_pipe.model)
        reward_pipe.model = model


        for step, batch in enumerate(tqdm(dataloader, desc="RM batch steps")):
            logger.info(f"RM inference step {step}/{len(dataloader)}")

            if model_type == "Custom Classifier":
                text_rejected = [b["text_rejected"] for b in batch]
                text_chosen = [b["text_chosen"] for b in batch][0]
                chosen = text_chosen[-1]['content']
                if args.pure_number:
                    match = confidence_pattern.findall(chosen)
                else:
                    match = confidence_pattern.search(chosen)
                if match:
                    if args.pure_number:
                        last_number = match[-1]
                        pre_text = chosen[:chosen.rfind(last_number)]
                        post_text = chosen[chosen.rfind(last_number) + len(last_number):]
                    else:
                        pre_text = chosen[:match.start(1)]
                        post_text = match.group(3)
                    for new_confidence in range(1, 11):
                        if args.pure_number:
                            temp = f"{pre_text}{new_confidence}{post_text}"
                        else:
                            temp = f"{pre_text}{match.group(1)}{new_confidence}{post_text}"
                        new_text = copy.deepcopy(text_chosen)
                        new_text[-1]['content'] = temp
                        #logger.info(new_text)
                        if isinstance(reward_pipe, CustomArmoRMPipeline):
                            results_sub, score_chosen_batch, score_rejected_batch = reward_pipe([new_text], text_rejected, **reward_pipeline_kwargs)
                            score_chosen = score_chosen_batch.cpu().numpy().tolist()
                            results[new_confidence].extend(score_chosen)
                        else:
                            rewards_chosen = reward_pipe(new_text, **reward_pipeline_kwargs)
                            if isinstance(rewards_chosen[0], dict):
                                score_chosen_batch = [result["score"] for result in rewards_chosen]
                            else:
                                score_chosen_batch = (
                                    rewards_chosen.float().cpu().numpy().tolist()
                                )  # cast to float in case of bfloat16
                            results[new_confidence].extend(score_chosen_batch)

            else:
                text_chosen = batch['text_chosen'][0]
                #print(text_chosen)
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
                            temp = f"{pre_text}{new_confidence}{post_text}"
                        else:
                            temp = f"{pre_text}{match.group(1)}{new_confidence}{post_text}"
                        #logger.info(new_text)
                        rewards_chosen = reward_pipe([new_text], **reward_pipeline_kwargs)
                        if isinstance(rewards_chosen[0], dict):
                            score_chosen_batch = [result["score"] for result in rewards_chosen]
                        else:
                            score_chosen_batch = rewards_chosen.cpu().numpy().tolist()
                        results[new_confidence].extend(score_chosen_batch)
                        
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