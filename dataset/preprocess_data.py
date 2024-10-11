from datasets import load_dataset, interleave_datasets, concatenate_datasets, DatasetDict, Value
import itertools
from collections import defaultdict
from tqdm import tqdm
from transformers import AutoTokenizer, AutoConfig, set_seed, AutoModel
import datasets
import random
import copy
import re
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import gather_object
import torch
from torch import nn
from typing import Optional
import numpy as np


ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
PROMPT = (
    "For the following question, provide your best response first, followed by your confidence in the accuracy or helpfulness of your response. Rate your confidence on a scale from 0 to 10.\n"
    "```Example Format:\n"
    f"<Your responses>\n"
    "Confidence: <Insert your numerical confidence level from 0 to 10, reflecting how certain you are that your answer is accurate or helpful.>```\n\n"
    "Ensure that your response strictly adheres to this format. Explicitly include the word 'Confidence:' in your response."
).strip()


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


# keep single turn
def filter_multi_turn(data, chosen_key='chosen', rejected_key='rejected'):
    return data.filter(lambda x: len(x[chosen_key]) == 2 and len(x[rejected_key]) == 2, cache_file_name=None)

def final_process(data, tokenizer):
    chosen_seen = set()
    rejected_seen = set()
    exclude_idx = set()
    len_count = 0
    duplicate_count = 0
    for idx, sample in tqdm(enumerate(data)):
        chosen = tokenizer.apply_chat_template(sample['chosen'], tokenize=False)
        rejected = tokenizer.apply_chat_template(sample['rejected'], tokenize=False)
        chosen_ids = tokenizer(chosen)['input_ids']
        rejected_ids = tokenizer(rejected)['input_ids']
        if len(chosen_ids) > 8192 or len(rejected_ids) > 8192:
            exclude_idx.add(idx)
            len_count += 1
        if chosen in chosen_seen or rejected in rejected_seen:
            exclude_idx.add(idx)
            duplicate_count += 1
        chosen_seen.add(chosen)
        rejected_seen.add(rejected)
    valid_data = data.select([i for i in range(len(data)) if i not in exclude_idx])
    print(f'Excluding {len(exclude_idx)} samples from final, {len_count} for length and {duplicate_count} for duplicates')
    return valid_data

def format_code_pairwise(data):
    code_re = re.compile(r'```python:? *\s*(.*?)```', re.DOTALL)
    input = data['input']
    chosen = data['accepted']
    rejected = data['rejected']
    chosen_code = code_re.search(chosen).group(1)
    rejected_code = code_re.search(rejected).group(1)
    if chosen_code == rejected_code:
        return {
            'chosen': None,
            'rejected': None
        }
    chosen = [{'role': 'user', 'content': input}, {'role': 'assistant', 'content': chosen_code}]
    rejected = [{'role': 'user', 'content': input}, {'role': 'assistant', 'content': rejected_code}]
    return {
        'chosen': chosen,
        'rejected': rejected
    }


def format_gsm8k(data):
    prompt = data['prompt']
    chosen = data['selected']
    rejected = data['rejected']
    match = ANS_RE.search(chosen)
    ans_chosen = None
    ans_rejected = None
    if match:
        chosen = chosen.replace(match.group(0), '')
        ans_chosen = match.group(1)
    match = ANS_RE.search(rejected)
    if match:
        rejected = rejected.replace(match.group(0), '')
        ans_rejected = match.group(1)
    # filter out chosen and rejected with same answer
    if ans_chosen == ans_rejected:
        return {
            'chosen': None,
            'rejected': None
        }
    chosen = [{'role': 'user', 'content': prompt}, {'role': 'assistant', 'content': chosen}]
    rejected = [{'role': 'user', 'content': prompt}, {'role': 'assistant', 'content': rejected}]
    return {
        'chosen': chosen,
        'rejected': rejected
    }
    

def add_confidence_prompt(sample):
    chosen = sample['chosen']
    assert chosen[-1]['role'] == 'assistant', f"last response should be assistant, get {chosen[-1]['role']}"
    rejected = sample['rejected']
    assert rejected[-1]['role'] == 'assistant', f"last response should be assistant, get {rejected[-1]['role']}"
    chosen_high_score = random.randint(7, 10)
    chosen_low_score = random.randint(0, 3)
    cwh = copy.deepcopy(chosen)
    cwh[-1]['content'] = cwh[-1]['content'] + f"\nConfidence: {chosen_high_score}."
    
    cwl = copy.deepcopy(chosen)
    cwl[-1]['content'] = cwl[-1]['content'] + f"\nConfidence: {chosen_low_score}."
    rejected_high_score = random.randint(7, 10)
    rejected_low_score = random.randint(0, 3)
    rwh = copy.deepcopy(rejected)
    rwh[-1]['content'] = rwh[-1]['content'] + f"\nConfidence: {rejected_high_score}."

    rwl = copy.deepcopy(rejected)
    rwl[-1]['content'] = rwl[-1]['content'] + f"\nConfidence: {rejected_low_score}."

    cwh = [{"role": "system", "content": PROMPT}] + cwh
    cwl = [{"role": "system", "content": PROMPT}] + cwl
    rwh = [{"role": "system", "content": PROMPT}] + rwh
    rwl = [{"role": "system", "content": PROMPT}] + rwl

    return {
        'chosen': chosen,
        'rejected': rejected,
        'chosen_high': cwh,
        'chosen_low': cwl,
        'rejected_high': rwh,
        'rejected_low': rwl
    }


def keep_chosen_rejected(dataset, dataset_name):
    return dataset.map(lambda x: {'chosen': x['chosen'], 'rejected': x['rejected'], 'dataset_name': dataset_name},
                       remove_columns=[col for col in dataset.column_names if col not in ['chosen', 'rejected']])


def chat_format(sample, chosen_key='chosen', rejected_key='rejected', prompt_key='question'):
    chosen = sample[chosen_key]
    rejected = sample[rejected_key]
    prompt = sample[prompt_key]
    return {
        'chosen': [{'content': prompt, 'role': 'user'}, {'content': chosen, 'role': 'assistant'}],
        'rejected': [{'content': prompt, 'role': 'user'}, {'content': rejected, 'role': 'assistant'}],
        'lang': sample['lang']
    }

def assign_scores(models, tokenizers, data):
    chosen_rewards = []
    rejected_rewards = []
    for sample in tqdm(data, total=len(data)):
        chosen = sample['chosen']
        rejected = sample['rejected']

        temp_chosen_rewards = []
        temp_rejected_rewards = []
        
        for model, tokenizer in zip(models, tokenizers):
            chosen_text = tokenizer.apply_chat_template(chosen, tokenize=False)
            rejected_text = tokenizer.apply_chat_template(rejected, tokenize=False)
            chosen_inputs = tokenizer(chosen_text, return_tensors='pt')
            rejected_inputs = tokenizer(rejected_text, return_tensors='pt')

            chosen_inputs = {k: v.to(model.device) for k, v in chosen_inputs.items()}
            rejected_inputs = {k: v.to(model.device) for k, v in rejected_inputs.items()}

            with torch.no_grad():
                chosen_outputs = model(**chosen_inputs)
                rejected_outputs = model(**rejected_inputs)

            temp_chosen_rewards.append(chosen_outputs.item())
            temp_rejected_rewards.append(rejected_outputs.item())

        # use average scores for calculation
        mean_chosen_reward = np.mean(temp_chosen_rewards, axis=0)
        mean_rejected_reward = np.mean(temp_rejected_rewards, axis=0)

        chosen_rewards.append(mean_chosen_reward)
        rejected_rewards.append(mean_rejected_reward)

    return chosen_rewards, rejected_rewards

# select the top 50% that has the largest perference strength
def filter_difference(data):
    data = data.sort('preference_strength', reverse=True)
    top_percent_index = int(0.5 * len(data))
    data = data.select(range(top_percent_index))
    return data

def eval_map(sample):
    new_sample = copy.deepcopy(sample)
    new_sample['source'] = 'mixture'
    new_sample['preference_strength'] = -100.0
    return new_sample


def load_model(model_name, current_device):
    config = AutoConfig.from_pretrained(model_name, trust_remote_code = True)
    config.normalized_reward = True     # use normalized reward 
    base_class = AutoModel._model_mapping[type(config)]
    base_pretrained_class = base_class.__base__
    head_prefix = 'value_head'
    cls_class = _get_reward_model(base_pretrained_class, base_class, head_prefix)
    model = cls_class.from_pretrained(model_name, config=config, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map = current_device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.padding_side = 'left'
    tokenizer.truncation_side = 'left'
    return model, tokenizer


def main():
    accelerator = Accelerator()
    current_device = accelerator.process_index
    set_seed(42)
    model_name_1 = 'OpenLLMAI/Llama-3-8b-rm-mixture'
    model_name_2 = 'Calibration/mistral-7b-rm-mixture'

    # load both reward models here
    model_1, tokenizer_1 = load_model(model_name_1, current_device)
    model_2, tokenizer_2 = load_model(model_name_2, current_device)
    
    train_data = load_dataset('Skywork/Skywork-Reward-Preference-80K-v0.1', split='train')
    train_data = filter_multi_turn(train_data)
    eval_data = load_dataset('OpenLLMAI/preference_dataset_mixture2_and_safe_pku', split='train')
    eval_data = eval_data.select(range(int(len(eval_data) * 0.03)))
    train_data = train_data.map(add_confidence_prompt, num_proc=8)

    models = [model_1, model_2]
    tokenizers = [tokenizer_1, tokenizer_2]
    accelerator.wait_for_everyone()

    with accelerator.split_between_processes(train_data) as data:
        chosen_outputs = []
        rejected_outputs = []
        chosen_rewards, rejected_rewards = assign_scores(models, tokenizers, data)
        chosen_outputs.extend(chosen_rewards)
        rejected_outputs.extend(rejected_rewards)

    chosen_outputs = gather_object(chosen_outputs)
    rejected_outputs = gather_object(rejected_outputs)

    if accelerator.is_main_process:
        train_data = train_data.add_column("index", range(len(train_data)))
        def add_scores(example, chosen_scores, rejected_scores):
            index = example['index']
            example['preference_strength'] = chosen_scores[index] - rejected_scores[index]
            example['chosen_score'] = chosen_scores[index]
            example['rejected_score'] = rejected_scores[index]
            return example

        train_data = train_data.map(add_scores, fn_kwargs={'chosen_scores': chosen_outputs, 'rejected_scores': rejected_outputs})
        
        # select the top 10% of the dataset with the largest difference in scores
        train_data = filter_difference(train_data)
        train_data = train_data.remove_columns('index')
        print(f"Selected {len(train_data)} samples for final dataset")
        
        eval_data = eval_data.map(add_confidence_prompt, num_proc = 8)
        eval_data = eval_data.map(eval_map, num_proc = 8)

        common_features = set(train_data.features.keys()) & set(eval_data.features.keys())
        train_data = train_data.select_columns(list(common_features))
        eval_data = eval_data.select_columns(list(common_features))

        dataset = DatasetDict({
            'train': train_data,   
            'eval': eval_data     
        })
        dataset.push_to_hub('Calibration/calibration_preference_mixture_final-v0.2')


if __name__ == '__main__':
    main()

