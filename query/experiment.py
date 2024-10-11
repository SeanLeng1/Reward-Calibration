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
from datasets import load_dataset
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

# Enable TensorFloat32 (TF32) tensor cores on Ampere GPUs for matrix multiplications (faster than FP32)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# get token from HF_TOKEN env variable, but if it doesn't exist pass none
HF_TOKEN = os.getenv("HF_TOKEN", None)
# this is necessary to automatically log in when running this script in docker/batch beaker jobs
if HF_TOKEN is not None:
    from huggingface_hub._login import _login

    _login(token=HF_TOKEN, add_to_git_credential=False)

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

def main(args):
    name = "/storage1/jiaxinh/Active/jixuan/rm_checkpoints_fixed/llama3-8b-crm-calibration-cr"
    dataset = load_dataset("Calibration/calibration_preference_mixture_v1", split="train")
    dataset.shuffle(seed = 42)
    dataset = dataset.select(range(1000))
    config = AutoConfig.from_pretrained(name, trust_remote_code = True)
    base_class = AutoModel._model_mapping[type(config)]
    base_pretrained_class = base_class.__base__
    head_prefix = 'value_head'
    cls_class = _get_reward_model(base_pretrained_class, base_class, head_prefix)
    # our trained and OpenRLHF model use bf16 by default
    model = cls_class.from_pretrained(name, config=config, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map = "cuda")
    tokenizer = AutoTokenizer.from_pretrained(name)

    chosen_scores = []
    rejected_scores = []
    with torch.no_grad():
        for sample in tqdm(dataset, total=len(dataset)):
            chosen = sample[args.chosen_key]
            rejected = sample[args.rejected_key]

            chosen = tokenizer.apply_chat_template(chosen, tokenize = False)
            rejected = tokenizer.apply_chat_template(rejected, tokenize = False)

            chosen = tokenizer(chosen, return_tensors="pt").to("cuda")
            rejected = tokenizer(rejected, return_tensors="pt").to("cuda")
            chosen_score = model(**chosen).float()
            rejected_score = model(**rejected).float()
            chosen_scores.append(chosen_score)
            rejected_scores.append(rejected_score)

    count = 0
    for chosen_score, rejected_score in zip(chosen_scores, rejected_scores):
        if chosen_score > rejected_score:
            count += 1

    print(f'Accuracy: {count / len(chosen_scores)}')
    print(f'Total: {len(chosen_scores)}')
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chosen_key", type=str, default="rejected_high")
    parser.add_argument("--rejected_key", type=str, default="rejected_low")
    args = parser.parse_args()
    main(args)