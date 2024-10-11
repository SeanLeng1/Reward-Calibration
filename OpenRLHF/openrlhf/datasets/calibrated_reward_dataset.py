from typing import Callable
import os
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import copy
from .utils import exist_and_not_none, zero_pad_sequences
import re

def human_assistant_format(data):
    human = 'Human: {prompt}'
    assistant = 'Assistant: {response}'
    formatted_data = ""
    for msg in data:
        if msg['role'] == 'system':
            formatted_data += human.format(prompt = msg['content'])
            formatted_data = formatted_data.rstrip('\n') + '\n\n'
        elif msg['role'] == 'user':
            if formatted_data:
                formatted_data += msg['content']
                formatted_data = formatted_data.rstrip('\n') + '\n'
            else:
                formatted_data += human.format(prompt = msg['content'])
            formatted_data = formatted_data.rstrip('\n') + '\n'
        else:
            formatted_data += assistant.format(response = msg['content'])
            formatted_data = formatted_data.rstrip('\n') + '\n'
    formatted_data = formatted_data.rstrip('\n')


    return formatted_data

# extract confidence from the sample for DPO
def extract_confidence_and_prompt(text, eos_token='</s>'):
    # in case there is also another Confidence: d that is part of the model response, therefore we should extract the one with eos token.
    # it is fine we removed eos token here, getitem will add it back
    pattern_string = r'(Confidence:\s*)(\d+\.?\d*)\s*(' + re.escape(eos_token) + r')?\s*$'
    pattern = re.compile(pattern_string)

    match = pattern.search(text)
    if match:
        confidence_value = match.group(2)
        confidence_prompt = text[:match.end(1)]
        return confidence_prompt, confidence_value
    else:
        return text, None

def preprocess_data(
    data,
    input_template=None,
    prompt_key=None,
    chosen_key="chosen",
    rejected_key="rejected",
    confidence_prompt_chosen_key=None,
    confidence_prompt_rejected_key=None,
    cwh_key="chosen_high",
    cwl_key="chosen_low",
    rwh_key="rejected_high",
    rwl_key="rejected_low",
    apply_chat_template=None,
    is_dpo=False,
    eos_token='</s>'
) -> str:
    if apply_chat_template:
        if prompt_key and confidence_prompt_chosen_key and confidence_prompt_rejected_key:
            prompt = apply_chat_template(data[prompt_key], tokenize=False, add_generation_prompt=True)
            chosen = apply_chat_template(data[prompt_key] + data[chosen_key], tokenize=False)[len(prompt) :]
            rejected = apply_chat_template(data[prompt_key] + data[rejected_key], tokenize=False)[len(prompt) :]

            confidence_prompt_chosen = apply_chat_template(data[confidence_prompt_chosen_key], tokenize=False, add_generation_prompt=True)
            confidence_prompt_rejected = apply_chat_template(data[confidence_prompt_rejected_key], tokenize=False, add_generation_prompt=True)
            cwh = apply_chat_template(data[confidence_prompt_chosen_key] + data[cwh_key], tokenize=False)[len(confidence_prompt_chosen) :]
            cwl = apply_chat_template(data[confidence_prompt_chosen_key] + data[cwl_key], tokenize=False)[len(confidence_prompt_chosen) :]
            rwh = apply_chat_template(data[confidence_prompt_rejected_key] + data[rwh_key], tokenize=False)[len(confidence_prompt_rejected) :]
            rwl = apply_chat_template(data[confidence_prompt_rejected_key] + data[rwl_key], tokenize=False)[len(confidence_prompt_rejected) :]
        else:
            prompt = ""
            chosen = apply_chat_template(data[chosen_key], tokenize=False)
            rejected = apply_chat_template(data[rejected_key], tokenize=False)

            confidence_prompt_chosen = ""
            confidence_prompt_rejected = ""
            cwh = apply_chat_template(data[cwh_key], tokenize=False)
            cwl = apply_chat_template(data[cwl_key], tokenize=False)
            rwh = apply_chat_template(data[rwh_key], tokenize=False)
            rwl = apply_chat_template(data[rwl_key], tokenize=False)

            if is_dpo:
                prompt = apply_chat_template(data[chosen_key][:-1], tokenize=False, add_generation_prompt=True)
                chosen = chosen[len(prompt) :]
                rejected = rejected[len(prompt) :]

                # for dpo, since we are comparing cwh with cwl, we probably should mask all the common part??
                confidence_prompt_chosen, cwh = extract_confidence_and_prompt(cwh, eos_token)
                _, cwl = extract_confidence_and_prompt(cwl, eos_token)
                confidence_prompt_rejected, rwh = extract_confidence_and_prompt(rwh, eos_token)
                _, rwl = extract_confidence_and_prompt(rwl, eos_token)
                
                # confidence_prompt_chosen = re.split(r'Confidence:\s*\d+\.?\d*', cwh)[0]
                # confidence_prompt_rejected = re.split(r'Confidence:\s*\d+\.?\d*', rwh)[0]
                # cwh = cwh[len(confidence_prompt_chosen) :]
                # cwl = cwl[len(confidence_prompt_chosen) :]
                # rwh = rwh[len(confidence_prompt_rejected) :]
                # rwl = rwl[len(confidence_prompt_rejected) :]
    else:
        if prompt_key and confidence_prompt_rejected_key and confidence_prompt_chosen_key:
            prompt = data[prompt_key]
            confidence_prompt_chosen = data[confidence_prompt_chosen_key]
            confidence_prompt_rejected = data[confidence_prompt_rejected_key]
            if input_template:
                prompt = input_template.format(prompt)
                confidence_prompt_chosen = input_template.format(confidence_prompt_chosen)
                confidence_prompt_rejected = input_template.format(confidence_prompt_rejected)
        else:
            prompt = ""
            confidence_prompt_chosen = ""
            confidence_prompt_rejected = ""

        chosen = data[chosen_key]
        rejected = data[rejected_key]

        cwh = data[cwh_key]
        cwl = data[cwl_key]
        rwh = data[rwh_key]
        rwl = data[rwl_key]

    # margin loss
    margin = data["margin"] if exist_and_not_none(data, "margin") else 0

    return prompt, chosen, rejected, confidence_prompt_chosen, confidence_prompt_rejected, cwh, cwl, rwh, rwl, margin


class CalibratedRewardDataset(Dataset):
    """
    Dataset for reward model

    Args:
        dataset: dataset for reward model
        self.tokenizer: self.tokenizer for reward model
        self.max_length: max length of input
    """

    def __init__(
        self,
        dataset,
        tokenizer: Callable,
        max_length: int,
        strategy,
        input_template=None,
        is_dpo=False,
        num_processors=8,
    ) -> None:
        super().__init__()
        self.is_dpo = is_dpo
        self.tokenizer = tokenizer
        self.strategy = strategy
        self.max_length = max_length
        self.is_dpo = is_dpo
        # handle weird HF chat template add double bos token
        self.add_special_tokens = True

        self.input_template = input_template
        self.prompt_key = getattr(self.strategy.args, "prompt_key", None)
        self.chosen_key = getattr(self.strategy.args, "chosen_key", None)
        self.rejected_key = getattr(self.strategy.args, "rejected_key", None)

        self.confidence_prompt_chosen_key = getattr(self.strategy.args, "confidence_prompt_chosen_key", None)
        self.confidence_prompt_rejected_key = getattr(self.strategy.args, "confidence_prompt_rejected_key", None)
        self.cwh_key = getattr(self.strategy.args, "cwh_key", 'chosen_high')
        self.cwl_key = getattr(self.strategy.args, "cwl_key", 'chosen_low')
        self.rwh_key = getattr(self.strategy.args, "rwh_key", 'rejected_high')
        self.rwl_key = getattr(self.strategy.args, "rwl_key", 'rejected_low')

        self.apply_chat_template = getattr(self.strategy.args, "apply_chat_template", False)
        if self.apply_chat_template:
            self.apply_chat_template = self.tokenizer.apply_chat_template
            tokenizer_chat_template = getattr(self.strategy.args, "tokenizer_chat_template", None)
            if tokenizer_chat_template:
                self.tokenizer.chat_template = tokenizer_chat_template
            if 'bos_token' in tokenizer.chat_template or (tokenizer.bos_token is not None and tokenizer.bos_token in tokenizer.chat_template):
                self.add_special_tokens = False
                
        if self.add_special_tokens:
            self.strategy.print('Adding special tokens!!!')
        else:
            self.strategy.print('NOT Adding special tokens!!!')

        # Parallel loading datasets
        processed_dataset = dataset.map(
            self.process_data, remove_columns=dataset.column_names, num_proc=num_processors
        )
        processed_dataset = processed_dataset.filter(lambda x: x["prompt"] is not None and x['confidence_prompt_chosen'] is not None and x['confidence_prompt_rejected'] is not None)

        self.prompts = processed_dataset["prompt"]
        self.chosens = processed_dataset["chosen"]
        self.rejects = processed_dataset["reject"]

        self.confidence_prompts_chosen = processed_dataset["confidence_prompt_chosen"]
        self.confidence_prompts_rejected = processed_dataset["confidence_prompt_rejected"]
        self.cwhs = processed_dataset["cwh"]
        self.cwls = processed_dataset["cwl"]
        self.rwhs = processed_dataset["rwh"]
        self.rwls = processed_dataset["rwl"]

        self.extras = processed_dataset["extra"]

    def process_data(self, data):
        prompt, chosen, rejected, confidence_prompt_chosen, confidence_prompt_rejected, cwh, cwl, rwh, rwl, margin = preprocess_data(
            data,
            self.input_template,
            self.prompt_key,
            self.chosen_key,
            self.rejected_key,
            self.confidence_prompt_chosen_key,
            self.confidence_prompt_rejected_key,
            self.cwh_key,
            self.cwl_key,
            self.rwh_key,
            self.rwl_key,
            self.apply_chat_template,
            self.is_dpo,
            # i dont know, some weird behavior with llama3.
            # it has end_of_text as eos, but actually using eot
            self.tokenizer.eos_token if self.tokenizer.decode(128009) != '<|eot_id|>' else '<|eot_id|>',     
        )

        if self.is_dpo:
            prompt_token = self.tokenizer(
                prompt,
                max_length=self.max_length,
                padding=False,
                truncation=True,
                return_tensors="pt",
                add_special_tokens = self.add_special_tokens,
            )
            prompt_ids_len = prompt_token["attention_mask"].int().sum().item()

            # Filter the sample whose length is greater than max_length (2 for answer length)
            if prompt_ids_len >= self.max_length - 2:
                prompt = None
  
            confidence_prompt_chosen_token = self.tokenizer(
                confidence_prompt_chosen,
                max_length=self.max_length,
                padding=False,
                truncation=True,
                return_tensors="pt",
                add_special_tokens = self.add_special_tokens,
            )
            confidence_prompt_rejected_token = self.tokenizer(
                confidence_prompt_rejected,
                max_length=self.max_length,
                padding=False,
                truncation=True,
                return_tensors="pt",
                add_special_tokens = self.add_special_tokens,
            )
            confidence_prompt_chosen_ids_len = confidence_prompt_chosen_token["attention_mask"].int().sum().item()
            confidence_prompt_rejected_ids_len = confidence_prompt_rejected_token["attention_mask"].int().sum().item()
            confidence_prompt_ids_len = (confidence_prompt_chosen_ids_len, confidence_prompt_rejected_ids_len)
            if confidence_prompt_chosen_ids_len >= self.max_length - 2:
                confidence_prompt_chosen = None
            if confidence_prompt_rejected_ids_len >= self.max_length - 2:
                confidence_prompt_rejected = None
            
        
        return {
            "prompt": prompt,
            "chosen": chosen,
            "reject": rejected,
            "confidence_prompt_chosen": confidence_prompt_chosen,
            "confidence_prompt_rejected": confidence_prompt_rejected,
            "cwh": cwh,
            "cwl": cwl,
            "rwh": rwh,
            "rwl": rwl,
            "extra": (prompt_ids_len, confidence_prompt_chosen_ids_len, confidence_prompt_rejected_ids_len) if self.is_dpo else margin,
        }

    def __len__(self):
        length = len(self.chosens)
        return length

    def __getitem__(self, idx):
        prompt, chosen, reject, confidence_prompt_chosen, confidence_prompt_rejected, cwh, cwl, rwh, rwl, extra = self.prompts[idx], self.chosens[idx], self.rejects[idx], self.confidence_prompts_chosen[idx], self.confidence_prompts_rejected[idx], self.cwhs[idx], self.cwls[idx], self.rwhs[idx], self.rwls[idx], self.extras[idx]
        chosen = (prompt + chosen).rstrip("\n")
        if not chosen.endswith(self.tokenizer.eos_token):
            chosen += " " + self.tokenizer.eos_token
        chosen_token = self.tokenizer(
            chosen,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
            add_special_tokens = self.add_special_tokens,
        )

        cwh = (confidence_prompt_chosen + cwh).rstrip("\n")
        if not cwh.endswith(self.tokenizer.eos_token):
            cwh += " " + self.tokenizer.eos_token
        cwh_token = self.tokenizer(
            cwh,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
            add_special_tokens = self.add_special_tokens,
        )
        cwl = (confidence_prompt_chosen + cwl).rstrip("\n")
        if not cwl.endswith(self.tokenizer.eos_token):
            cwl += " " + self.tokenizer.eos_token
        cwl_token = self.tokenizer(
            cwl,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
            add_special_tokens = self.add_special_tokens,
        )


        reject = (prompt + reject).rstrip("\n")
        if not reject.endswith(self.tokenizer.eos_token):
            reject += " " + self.tokenizer.eos_token
        reject_token = self.tokenizer(
            reject,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
            add_special_tokens = self.add_special_tokens,
        )

        rwh = (confidence_prompt_rejected + rwh).rstrip("\n")
        if not rwh.endswith(self.tokenizer.eos_token):
            rwh += " " + self.tokenizer.eos_token
        rwh_token = self.tokenizer(
            rwh,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
            add_special_tokens = self.add_special_tokens,
        )
        rwl = (confidence_prompt_rejected + rwl).rstrip("\n")
        if not rwl.endswith(self.tokenizer.eos_token):
            rwl += " " + self.tokenizer.eos_token
        rwl_token = self.tokenizer(
            rwl,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
            add_special_tokens = self.add_special_tokens,
        )

        # to avoid EOS_token truncation
        chosen_token["input_ids"][0][-1] = self.tokenizer.eos_token_id
        reject_token["input_ids"][0][-1] = self.tokenizer.eos_token_id
        chosen_token["attention_mask"][0][-1] = True
        reject_token["attention_mask"][0][-1] = True

        cwh_token["input_ids"][0][-1] = self.tokenizer.eos_token_id
        cwl_token["input_ids"][0][-1] = self.tokenizer.eos_token_id
        rwh_token["input_ids"][0][-1] = self.tokenizer.eos_token_id
        rwl_token["input_ids"][0][-1] = self.tokenizer.eos_token_id
        cwh_token["attention_mask"][0][-1] = True
        cwl_token["attention_mask"][0][-1] = True
        rwh_token["attention_mask"][0][-1] = True
        rwl_token["attention_mask"][0][-1] = True


        return (
            chosen_token["input_ids"],
            chosen_token["attention_mask"],
            reject_token["input_ids"],
            reject_token["attention_mask"],
            cwh_token["input_ids"],
            cwh_token["attention_mask"],
            cwl_token["input_ids"],
            cwl_token["attention_mask"],
            rwh_token["input_ids"],
            rwh_token["attention_mask"],
            rwl_token["input_ids"],
            rwl_token["attention_mask"],
            extra,
        )

    def collate_fn(self, item_list):
        chosen_ids = []
        chosen_masks = []
        reject_ids = []
        rejects_masks = []
        cwh_ids = []
        cwh_masks = []
        cwl_ids = []
        cwl_masks = []
        rwh_ids = []
        rwh_masks = []
        rwl_ids = []
        rwl_masks = []

        extras = []
        for chosen_id, chosen_mask, reject_id, rejects_mask, cwh_id, cwh_mask, cwl_id, cwl_mask, rwh_id, rwh_mask, rwl_id, rwl_mask, extra in item_list:
            chosen_ids.append(chosen_id)
            chosen_masks.append(chosen_mask)
            reject_ids.append(reject_id)
            rejects_masks.append(rejects_mask)
            cwh_ids.append(cwh_id)
            cwh_masks.append(cwh_mask)
            cwl_ids.append(cwl_id)
            cwl_masks.append(cwl_mask)
            rwh_ids.append(rwh_id)
            rwh_masks.append(rwh_mask)
            rwl_ids.append(rwl_id)
            rwl_masks.append(rwl_mask)

            extras.append(extra)

        if self.is_dpo:
            padding_side = "right"
        else:
            padding_side = "left"
        chosen_ids = zero_pad_sequences(chosen_ids, side=padding_side, value=self.tokenizer.pad_token_id)
        chosen_masks = zero_pad_sequences(chosen_masks, side=padding_side)
        reject_ids = zero_pad_sequences(reject_ids, side=padding_side, value=self.tokenizer.pad_token_id)
        rejects_masks = zero_pad_sequences(rejects_masks, side=padding_side)

        cwh_ids = zero_pad_sequences(cwh_ids, side=padding_side, value=self.tokenizer.pad_token_id)
        cwh_masks = zero_pad_sequences(cwh_masks, side=padding_side)
        cwl_ids = zero_pad_sequences(cwl_ids, side=padding_side, value=self.tokenizer.pad_token_id)
        cwl_masks = zero_pad_sequences(cwl_masks, side=padding_side)
        rwh_ids = zero_pad_sequences(rwh_ids, side=padding_side, value=self.tokenizer.pad_token_id)
        rwh_masks = zero_pad_sequences(rwh_masks, side=padding_side)
        rwl_ids = zero_pad_sequences(rwl_ids, side=padding_side, value=self.tokenizer.pad_token_id)
        rwl_masks = zero_pad_sequences(rwl_masks, side=padding_side)

        return chosen_ids, chosen_masks, reject_ids, rejects_masks, cwh_ids, cwh_masks, cwl_ids, cwl_masks, rwh_ids, rwh_masks, rwl_ids, rwl_masks, extras


    # TODO: ignore this packing implementation for now
    # since packing would be too long (exceed the max_length for 4 sequences)
    def packing_collate_fn(self, item_list):
        raise NotImplementedError
        extras = []

        chosen_ids = []
        chosen_att_masks = []
        chosen_seq_lens = []
        rejected_ids = []
        rejected_att_masks = []
        rejected_seq_lens = []
        index = 1
        for chosen_id, chosen_mask, reject_id, rejects_mask, extra in item_list:
            chosen_ids.append(chosen_id.flatten())
            chosen_att_masks.append(torch.ones_like(chosen_id.flatten()) * index)
            chosen_seq_lens.append(len(chosen_id.flatten()))
            extras.append(extra)

            rejected_ids.append(reject_id.flatten())
            rejected_att_masks.append(torch.ones_like(reject_id.flatten()) * (index + len(item_list)))
            rejected_seq_lens.append(len(reject_id.flatten()))
            index += 1

        # Concatenate all tensors into a single row
        # https://github.com/huggingface/transformers/blob/v4.42.4/src/transformers/models/llama/modeling_llama.py#L1028
        rejected_ids.append(torch.tensor([self.tokenizer.pad_token_id]))
        rejected_att_masks.append(torch.tensor([0]))

        packed_input_ids = torch.cat(chosen_ids + rejected_ids, dim=0).unsqueeze(0)
        packed_attention_masks = torch.cat(chosen_att_masks + rejected_att_masks, dim=0).unsqueeze(0)
        packed_seq_lens = chosen_seq_lens + rejected_seq_lens
        return packed_input_ids, packed_attention_masks, packed_seq_lens, extras
