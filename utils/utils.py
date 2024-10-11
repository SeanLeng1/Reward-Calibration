from transformers import set_seed, StoppingCriteria
import random
import numpy as np
import torch
import tqdm
import re
from typing import List
import copy

# actually i think transformers.set_seed is enough, it already handles these
def set_random_seed(seed):
    if seed is not None:
        set_seed(seed)
        # random.seed(seed)
        # np.random.seed(seed)
        # torch.manual_seed(seed)

def is_rank_0():
    """Check whether it is rank 0."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            return True
        else:
            return False
    else:
        return True

def print_rank_0(msg, rank=None):
    if rank is not None and rank <= 0:
        print(msg)
    elif is_rank_0():
        print(msg)


class KeyWordsCriteria(StoppingCriteria):
    def __init__(self, stop_id_sequences):
        assert isinstance(stop_id_sequences[0], list), "stop_id_sequences should be a list of list of ids"
        self.stop_sequences = stop_id_sequences

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        sequences_should_be_stopped = []
        for i in range(input_ids.shape[0]):
            for stop_sequence in self.stop_sequences:
                if input_ids[i][-len(stop_sequence):].tolist() == stop_sequence:
                    sequences_should_be_stopped.append(True)
                    break
            sequences_should_be_stopped.append(False)
        return all(sequences_should_be_stopped)
    
# some tokenizers (tulu and zephyr) tokenize "10" as "1" and "0"
# therefore we cannot use Confidence: 1 as the stopping criteria, since it might still generate "0" to form "10"
# therefore, we should continue generating if the confidence is followed by 1 and stop otherwise
class MyStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.stop_positions = []

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
    

@torch.no_grad()
def generate_completions(model, device, tokenizer, prompts, batch_size=1, stop_id_sequences=None, disable_tqdm=False, skip_prompt = True, add_special_tokens = False, use_stop_criteria = True, **generation_kwargs):
    generations = []
    if generation_kwargs:
        print_rank_0(generation_kwargs)
    if not disable_tqdm:
        progress = tqdm.tqdm(total=len(prompts), desc="Generating completions")
    
    stopping_criteria = [MyStoppingCriteria(tokenizer)] if use_stop_criteria else None
    num_return_sequences = generation_kwargs.get("num_return_sequences", 1) 
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        tokenized_prompts = tokenizer(batch_prompts, padding = 'longest', add_special_tokens = add_special_tokens, return_tensors="pt")
        batch_input_ids = tokenized_prompts.input_ids.to(device)
        batch_attention_mask = tokenized_prompts.attention_mask.to(device)

        try:
            batch_outputs = model.generate(
                input_ids=batch_input_ids,
                attention_mask=batch_attention_mask,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                #stopping_criteria=[KeyWordsCriteria(stop_id_sequences)] if stop_id_sequences else None,
                stopping_criteria=stopping_criteria,
                **generation_kwargs
            )
            batch_outputs = batch_outputs.detach().cpu()
            # the stopping criteria is applied at batch level, so if other examples are not stopped, the entire batch will continue to generate.
            # so some outputs still have the stop sequence, which we need to remove.
            if stop_id_sequences:
                for output_idx in range(batch_outputs.shape[0]):
                    for token_idx in range(batch_input_ids.shape[1], batch_outputs.shape[1]):
                        if any(batch_outputs[output_idx, token_idx: token_idx+len(stop_sequence)].tolist() == stop_sequence for stop_sequence in stop_id_sequences):
                            batch_outputs[output_idx, token_idx:] = tokenizer.pad_token_id
                            break

            # remove the prompt from the output
            # we need to re-encode the prompt because we need to make sure the special tokens are treated the same way as in the outputs.
            # we changed our previous way of truncating the output token ids dicrectly because some tokenizer (e.g., llama) won't add space token before the first token.
            # space is important for some tasks (e.g., code completion).
            batch_outputs = tokenizer.batch_decode(batch_outputs, skip_special_tokens=True)
            batch_prompts = tokenizer.batch_decode(batch_input_ids, skip_special_tokens=True)
            if skip_prompt:
                # duplicate the prompts to match the number of return sequences
                batch_prompts = [prompt for prompt in batch_prompts for _ in range(num_return_sequences)]
                batch_generations = [
                    output[len(prompt):] for prompt, output in zip(batch_prompts, batch_outputs)
                ]
            else:
                batch_generations = batch_outputs

        except Exception as e:
            print("Error when generating completions for batch:")
            print("Error message:")
            print(e)
            print("Use empty string as the completion.")
            batch_generations = [""] * len(batch_prompts) * num_return_sequences
        generations += batch_generations

        if not disable_tqdm:
            progress.update(len(batch_prompts) // num_return_sequences)

    assert len(generations) == len(prompts) * num_return_sequences, "number of generations should be equal to number of prompts * num_return_sequences"
    return generations



@torch.no_grad()
def get_next_word_predictions(model, device, tokenizer, prompts, candidate_token_ids=None, batch_size=1, return_token_predictions=False, disable_tqdm=False):
    predictions, probs = [], []
    if not disable_tqdm:
        progress = tqdm.tqdm(total=len(prompts), desc="Getting Predictions")

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i: i+batch_size]
        tokenized_prompts = tokenizer(batch_prompts, padding="longest", return_tensors="pt")
        batch_input_ids = tokenized_prompts.input_ids.to(device)
        attention_mask = tokenized_prompts.attention_mask.to(device)

        batch_logits = model(batch_input_ids, attention_mask).logits[(torch.arange(batch_input_ids.size(0)), attention_mask.sum(dim=-1)-1)]
        if candidate_token_ids is not None:
            batch_logits = batch_logits[:, candidate_token_ids]
        batch_probs = torch.softmax(batch_logits, dim=-1)
        batch_prediction_indices = torch.argmax(batch_probs, dim=-1)
        if return_token_predictions:
            if candidate_token_ids is not None:
                candidate_tokens = tokenizer.convert_ids_to_tokens(candidate_token_ids)
                batch_predictions = [candidate_tokens[idx] for idx in batch_prediction_indices]
            else:
                batch_predictions = tokenizer.convert_ids_to_tokens(batch_prediction_indices)
            predictions += batch_predictions
        else:
            predictions += batch_prediction_indices.tolist()
        probs += batch_probs.tolist()

        if not disable_tqdm:
            progress.update(len(batch_prompts))

    assert len(predictions) == len(prompts), "number of predictions should be equal to number of prompts"
    return predictions, probs


def to_device(batch, device):
    output = {}
    for k, v in batch.items():
        try:
            output[k] = v.to(device)
        except:
            output[k] = v
    return output


def check_tokenizer_chat_template(tokenizer):
    """
    Check if tokenizer has non none chat_template attribute.
    """
    if hasattr(tokenizer, "chat_template"):
        if tokenizer.chat_template is not None:
            return True
    return False


# def process_data(
#     data,
#     subsets,
#     ids,
#     mode = 'chosen',
#     data_size = 1000,
#     tokenizer=None,
#     add_probabilities = False,
#     format_prompt = False,
#     is_dpo = False,
# ):  
#     PROMPT = 'Please answer the following question and rate your confidence on a scale from 0 to 10.\n\nQuestion: {question}'
#     if data_size is None:
#         data_size = len(data)
#     processed_data = []
#     processed_subsets = []
#     processed_ids = []
#     for example, subset, id in zip(data, subsets, ids):
#         prompt = example['prompt'] if 'prompt' in example else None
#         chosen, rejected = example['text_chosen'], example['text_rejected']
#         if mode == 'chosen':        # same chosen with high/low scores
#             rejected = copy.deepcopy(chosen)
#         elif mode == 'rejected':    # same rejected with high/low scores
#             chosen = copy.deepcopy(rejected)
#         if add_probabilities:
#             low_probability = random.randint(0, 3)
#             high_probability = random.randint(7, 10)
#             # if chosen and rejected are already lists
#             # it means rewardbench are using custom formatting
#             if isinstance(chosen, list) and isinstance(rejected, list):
#                 assert chosen[-1]['role'] == 'assistant', "Last sentence should be assistant"
#                 last_sentence = chosen[-1]['content']
#                 last_sentence = last_sentence.rstrip() + f" Confidence: {low_probability}."
#                 chosen[-1]['content'] = last_sentence
#                 assert rejected[-1]['role'] == 'assistant', "Last sentence should be assistant"
#                 last_sentence = rejected[-1]['content']
#                 last_sentence = last_sentence.rstrip() + f" Confidence: {high_probability}."
#                 rejected[-1]['content'] = last_sentence
#             else:
#                 chosen, rejected = chosen.rstrip(), rejected.rstrip()
#                 # chosen with low confidence and rejected with high confidence
#                 # because load_eval_dataset already handled the prompt template
#                 # we need to maintain the prompt template
#                 #sep = conv.sep2 if conv.sep2 else conv.sep
#                 sep = tokenizer.eos_token
#                 # special cases
#                 if tokenizer.name_or_path == 'sfairXC/FsfairX-LLaMA3-RM-v0.1':
#                     sep = '<|eot_id|>'
#                 last_sep_index = chosen.rfind(sep)
#                 if last_sep_index != -1:
#                     pre_sep = chosen[:last_sep_index].rstrip()  
#                     post_sep = chosen[last_sep_index:].lstrip()  
#                     chosen = pre_sep + f" Confidence: {low_probability}." + post_sep
#                 else:
#                     chosen = chosen + f" Confidence: {low_probability}"

#                 last_sep_index = rejected.rfind(sep)
#                 if last_sep_index != -1:
#                     pre_sep = rejected[:last_sep_index].rstrip() 
#                     post_sep = rejected[last_sep_index:].lstrip()  
#                     rejected = pre_sep + f" Confidence: {high_probability}." + post_sep
#                 else:
#                     rejected = rejected + f" Confidence: {high_probability}"
#         replaced_prompt = prompt
#         # this is not an effective way to format the prompt
#         # but since we do not want to modify rewardbench, use it for now
#         if format_prompt:
#             # it is safe to assume that both chosen and rejected contain the prompt, therefore we just need to find the common part
#             # rewardbench formatted the prompt, which means it contains weird tokens
#             # use a stupid way, we tokenize first then decode and skip special tokens
#             tokenized_prompt = tokenizer(prompt, return_tensors="pt")
#             clean_prompt = tokenizer.decode(tokenized_prompt.input_ids[0], skip_special_tokens=True)
#             clean_prompt = clean_prompt.strip()
#             # if clean prompt starts with user, we need to remove it
#             if clean_prompt.startswith("user"):
#                 clean_prompt = clean_prompt[4:].strip()
#             # Ziya
#             elif clean_prompt.startswith('Human: '):
#                 clean_prompt = clean_prompt[7:].strip()
#             # UltraRM
#             elif clean_prompt.startswith('User: '):
#                 clean_prompt = clean_prompt[6:].strip()
#             # PKU
#             elif clean_prompt.startswith('BEGINNING OF CONVERSATION: USER: '):
#                 clean_prompt = clean_prompt[33:].strip()
#             # tulu or stablelm-2-zephyr
#             elif clean_prompt.startswith('<|user|>'):
#                 clean_prompt = clean_prompt[8:].strip()
#             # Eurus
#             elif clean_prompt.startswith('[INST] '):
#                 clean_prompt = clean_prompt[7:].strip()
#             # Matter-0.1-7B-boost-DPO-preview
#             elif clean_prompt.startswith('<|im_start|> user'):
#                 clean_prompt = clean_prompt[17:].strip()
#             # stabilityai/stablelm-2-12b-chat or qwen2-7b (this contains a default system prompt)
#             elif clean_prompt.startswith("system\nYou are a helpful assistant.\nuser"):
#                 clean_prompt = clean_prompt[40:].strip()
#             replaced_prompt = PROMPT.format(question=clean_prompt)
#             # RLHFlow/ArmoRM-Llama3-8B-v0.1 or any Custom Classifier (they tokenize when calling the pipeline)
#             if isinstance(chosen, List):
#                 for sentence in chosen:
#                     if sentence['role'] == 'user':
#                         sentence['content'] = sentence['content'].replace(clean_prompt, replaced_prompt)
#                 for sentence in rejected:
#                     if sentence['role'] == 'user':
#                         sentence['content'] = sentence['content'].replace(clean_prompt, replaced_prompt)
#             else:
#                 # find the clean prompt in chosen and rejected
#                 chosen = chosen.replace(clean_prompt, replaced_prompt)
#                 rejected = rejected.replace(clean_prompt, replaced_prompt)
#             # we need to return the modified prompt for dpo for replacement later
#             # https://github.com/allenai/reward-bench/issues/140 
#             # it does not affect ppo reward model since they only use [text_chosen] and [text_rejected]
#             if is_dpo:
#                 pattern = re.escape(replaced_prompt)
#                 full_pattern = '.*?' + pattern
#                 match = re.search(full_pattern, chosen, re.DOTALL)
#                 replaced_prompt = match.group(0)

#         # id is removed anyway so we do not add them back
#         # we can always return the prompt
#         processed_data.append(
#             {
#                 "prompt": replaced_prompt,
#                 "text_chosen": chosen,
#                 "text_rejected": rejected,
#             }
#         )
#         processed_subsets.append(subset)
#         processed_ids.append(id)
#         if len(processed_data) == data_size:
#             break

#     return processed_data, processed_subsets, processed_ids
