from datasets import load_dataset, interleave_datasets, concatenate_datasets
import itertools
from collections import defaultdict
from tqdm import tqdm
from transformers import AutoTokenizer
import datasets
import random
import copy
import re

#PROMPT = 'Please answer the following question and rate your confidence on a scale from 0 to 10.\n\nQuestion: {question}'

ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
PROMPT = (
    "For the following question, provide your best response first, followed by your confidence in the accuracy or helpfulness of your response. Rate your confidence on a scale from 0 to 10.\n"
    "```Example Format:\n"
    f"<Your responses>\n"
    "Confidence: <Insert your numerical confidence level from 0 to 10, reflecting how certain you are that your answer is accurate or helpful.>```\n\n"
    "Ensure that your response strictly adheres to this format. Explicitly include the word 'Confidence:' in your response."
).strip()

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
    


def filter_difference(dataset, chosen_rating_key='chosen_rating', rejected_rating_key='rejected_rating', threshold=2):
    dataset = dataset.filter(lambda x: x[chosen_rating_key] is not None and x[chosen_rating_key] != 'N/A' and x[rejected_rating_key] is not None and x[rejected_rating_key] != 'N/A')

    if chosen_rating_key != 'chosen_score':
        dataset = dataset.rename_column(chosen_rating_key, 'chosen_score')
    if rejected_rating_key != 'rejected_score':
        dataset = dataset.rename_column(rejected_rating_key, 'rejected_score')

    return dataset.filter(lambda x: 
        (float(x['rejected_score']) != 0 and (float(x['chosen_score']) / float(x['rejected_score']) > threshold)) or
        (float(x['rejected_score']) == 0 and (float(x['chosen_score']) / (float(x['rejected_score']) + 1) > threshold))
    )

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


def main():
    random.seed(42)
    # any tokenizer can be used (should have chat_template tho)
    tokenizer = AutoTokenizer.from_pretrained("OpenRLHF/Llama-3-8b-sft-mixture")
    hh_rlhf = load_dataset('RLHFlow/HH-RLHF-Helpful-standard', split='train')
    hh_rlhf = filter_multi_turn(hh_rlhf)
    hh_rlhf = hh_rlhf.shuffle()
    if len(hh_rlhf) > 2500:
        hh_rlhf = hh_rlhf.select(range(2500))

    capybara = load_dataset('argilla/distilabel-capybara-dpo-7k-binarized', split='train')
    capybara = filter_multi_turn(capybara)
    capybara = filter_difference(capybara, 'rating_chosen', 'rating_rejected', 1)
    capybara = capybara.shuffle()

    gsm8k = load_dataset('reciprocate/gsm8k_train_pairwise', split='train')
    gsm8k = gsm8k.map(format_gsm8k, num_proc=4, remove_columns=gsm8k.column_names)
    gsm8k = gsm8k.filter(lambda x: x['chosen'] is not None and x['rejected'] is not None)
    gsm8k = gsm8k.shuffle()
    if len(gsm8k) > 2500:
        gsm8k = gsm8k.select(range(2500))

    code_feedback = load_dataset('RLHFlow/CodeUltraFeedback-standard', split='train')
    code_feedback = filter_multi_turn(code_feedback)
    code_feedback = filter_difference(code_feedback, 'chosen_score', 'rejected_score', 3)
    code_feedback = code_feedback.shuffle()

    math_dpo = load_dataset('RLHFlow/Argilla-Math-DPO-standard', split='train')
    math_dpo = filter_multi_turn(math_dpo)
    # has to be greater than 1
    math_dpo = filter_difference(math_dpo, 'chosen_rating', 'rejected_rating', 1)
    math_dpo = math_dpo.shuffle()

    ultrafeedback = load_dataset('argilla/ultrafeedback-binarized-preferences-cleaned', split='train')
    ultrafeedback = filter_multi_turn(ultrafeedback)
    ultrafeedback = filter_difference(ultrafeedback, 'chosen-rating', 'rejected-rating', 3.5)
    ultrafeedback = ultrafeedback.shuffle()

    pku = load_dataset('RLHFlow/PKU-SafeRLHF-30K-standard', split='train')
    pku = filter_multi_turn(pku)
    pku = pku.shuffle()
    if len(pku) > 2500:
        pku = pku.select(range(2500))

    shp = load_dataset('RLHFlow/SHP-standard', split='train')
    shp = filter_multi_turn(shp)
    shp = filter_difference(shp, 'chosen_score', 'rejected_score', 50)      # 50
    shp = shp.shuffle()

    helpsteer = load_dataset('RLHFlow/Helpsteer-preference-standard', split='train')
    helpsteer = filter_multi_turn(helpsteer)
    helpsteer = filter_difference(helpsteer, 'chosen_score', 'rejected_score', 2.5)       # 2.5
    helpsteer = helpsteer.shuffle()

    helpsteer2 = load_dataset('RLHFlow/Helpsteer2-standard', split='train')
    helpsteer2 = filter_multi_turn(helpsteer2)
    helpsteer2 = filter_difference(helpsteer2, 'chosen_score', 'rejected_score', 2)
    helpsteer2 = helpsteer2.shuffle()
    # if len(helpsteer2) > 2500:
    #     helpsteer2 = helpsteer2.select(range(2500))

    simple_math = load_dataset("fblgit/simple-math-DPO", split = 'train')
    # random select 1k samples
    simple_math = simple_math.shuffle()
    if len(simple_math) > 2500:
        simple_math = simple_math.select(range(2500))

    orca = load_dataset('RLHFlow/Orca-distibalel-standard', split='train')
    orca = filter_multi_turn(orca)
    orca = filter_difference(orca, 'chosen_score', 'rejected_score', 2.0)     # 1.5
    orca = orca.shuffle()

    code_pairwise = load_dataset('Vezora/Code-Preference-Pairs', split='train')
    code_pairwise = code_pairwise.map(format_code_pairwise, num_proc=4, remove_columns=code_pairwise.column_names)
    code_pairwise = code_pairwise.filter(lambda x: x['chosen'] is not None and x['rejected'] is not None)
    code_pairwise = filter_multi_turn(code_pairwise)
    code_pairwise = code_pairwise.shuffle()
    # add column for chosen_score and rejected_score
    code_pairwise = code_pairwise.add_column('chosen_score', [None] * len(code_pairwise))
    code_pairwise = code_pairwise.add_column('rejected_score', [None] * len(code_pairwise))
    if len(code_pairwise) > 2500:
        code_pairwise = code_pairwise.select(range(2500))

    code = load_dataset("CyberNative/Code_Vulnerability_Security_DPO", split = 'train')
    code = code.map(chat_format, fn_kwargs={'chosen_key': 'chosen', 'rejected_key': 'rejected', 'prompt_key': 'question'})
    code = code.shuffle()
    if len(code) > 2500:
        code = code.select(range(2500))

    dataset_list = [capybara, code_feedback, ultrafeedback, helpsteer, orca, shp, hh_rlhf, math_dpo, pku, code, simple_math, helpsteer2]
    dataset_names = ['capybara', 'code_feedback', 'ultrafeedback', 'helpsteer', 'orca', 'shp', 'hh_rlhf', 'math_dpo', 'pku', 'code_vulnerability', 'simple_math', 'helpsteer2']
    final_data_list = [keep_chosen_rejected(data, name) for data, name in zip(dataset_list, dataset_names)]
    # print len of each dataset
    for data, name in zip(final_data_list, dataset_names):
        print(f"Dataset {name} size: {len(data)}")
    dataset = concatenate_datasets(final_data_list)
    print(f"Initial dataset size: {len(dataset)}")
    dataset = final_process(dataset, tokenizer)
    print(f"Final dataset size: {len(dataset)}")
    dataset = dataset.map(add_confidence_prompt, num_proc=8)
    dataset.push_to_hub('Calibration/calibration_preference_mixture_final-v0.1')


if __name__ == '__main__':
    main()

