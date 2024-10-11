# reference: https://github.com/MiaoXiong2320/llm-uncertainty/blob/main/utils/compute_metrics.py#L56

import json, pdb
from typing import Dict
import os.path as osp
import pandas as pd
import datasets
import random
import os

def load_dataset_w_prediction(dataset_name: str, task_type: str, data_path: str):
    """
    dataset_name: name of the dataset
    task_type: type of the task, e.g. multi_choice_qa, open_number_qa
    data_path: path to the dataset
    """
    with open(data_path,'r') as f:
        data = json.load(f)     
    
    print("Information of used dataset: ", data['hyperparameters']) 
    
    return data  


def load_dataset(dataset_name: str, data_path: str, task_type: str):
    """
    dataset_name: name of the dataset
    task_type: type of the task, e.g. multi_choice_qa, open_number_qa
    data_path: path to the dataset
    """
    
    character = ["(A)", "(B)", "(C)", "(D)", "(E)", "(F)", "(G)", "(H)", "(I)", "(J)", "(K)", "(L)", "(M)", "(N)", "(O)", "(P)", "(Q)", "(R)", "(S)", "(T)", "(U)", "(V)", "(W)", "(X)", "(Y)", "(Z)"]
    
    number_options = 6          # it is not used in our implementation, but just leave it here in case
    
    # qa_data is the dictionary that stores the questions and answers
    # questions are the keys and answers are the values
    qa_data = {}
    
    if dataset_name == "GSM8K":
        import dataset.grade_school_math.dataset as gsm_loader
        examples = gsm_loader.get_examples(data_path)
        for qa in examples:
            question = qa['question']
            answer = {'answer':gsm_loader.extract_answer(qa["answer"]), 'options':number_options}
            qa_data[question] = answer

    elif "BigBench" in dataset_name:
        with open(data_path,'r') as f:
            data = json.load(f)
        if task_type == "open_number_qa":
            for qa in data['examples']:
                """ qa has two keys:  
                {   "input": "I have a clarinet, a violin, and a flute. How many musical instruments do I have?",
                    "target": ["three", "3"] }
                """
                question = qa['input']
                value = {'answer':qa['target'][1], 'options':number_options}
                qa_data[question] = value
                
        elif task_type == "multi_choice_qa":
        
            for qa in data['examples']:
                #question = qa['input'] +'\n' +'Options: '
                question = qa['input'] +'\nChoose the best answer from the following options.\n'
                j=0
                for key, value in qa['target_scores'].items():
                    option = character[j] + " " + key + "\t"
                    question += option
                    if value == 1:
                        answer_index = j
                    j += 1
                    
                value = {'answer':answer_index, 'options':list(qa['target_scores'].keys())}
                qa_data[question] = value
                
    elif dataset_name == "ScienceQA":
        with open(osp.join(data_path, "pid_splits.json"), 'r') as f:
            pids = json.load(f)
        
        with open(osp.join(data_path, "problems.json"), 'r') as f:
            data = json.load(f)
        
        for pid in pids['test']:
            #question = data[pid]['question'] + '\n' +'Options: '
            question = data[pid]['question'] + '\nChoose the best answer from the following options.\n'
            for idx, choice in enumerate(data[pid]['choices']):
                question += character[idx] + " " + choice + "\n"
                
            value = {'answer':data[pid]['answer'], 'options':data[pid]['choices']}
            qa_data[question] = value

    elif dataset_name == 'OpenBookQA':
        data = datasets.load_dataset('allenai/openbookqa', split = 'test')
        for idx, item in enumerate(data):
            question = item['question_stem'] + '\nChoose the best answer from the following options.\n'
            choices = item['choices']
            answer_key = item['answerKey']
            for i, character in enumerate(choices['label']):
                question += "(" + character + ') ' + choices['text'][i] + '\n'
            qa_data[question] = {'answer': answer_key, 'options': choices['label']}

    elif dataset_name == 'SciQ':
        data = datasets.load_dataset('allenai/sciq', split = 'test')
        for idx, item in enumerate(data):
            question = item['question'] + '\nChoose the best answer from the following options.\n'
            choices = [item['distractor3'], item['distractor1'], item['distractor2'], item['correct_answer']]
            random.shuffle(choices)
            correct_index = choices.index(item['correct_answer'])
            correct_index = character[correct_index][1:2]
            for i, option in enumerate(choices):
                question += character[i] + f' {option}' + '\n'
            qa_data[question] = {'answer': correct_index, 'options': choices}

    elif dataset_name == 'TruthfulQA':
        data = datasets.load_from_disk(data_path)['validation']
        choices = ['(A) ', '(B) ', '(C) ', '(D) ', '(E) ', '(F) ', '(G) ', '(H) ', '(I) ', '(J) ', '(K) ', '(L) ', '(M) ']
        for idx, item in enumerate(data):
            question = item['question'] + '\nChoose the best answer from the following options.\n'
            candidates = item['mc1_targets']['choices']
            indexed_list = list(enumerate(candidates))
            random.shuffle(indexed_list)
            shuffled_list = [element for _, element in indexed_list]
            original_indices = [index for index, _ in indexed_list]
            correct_index = choices[original_indices.index(0)][1:2] # correct answer is always the first one
            for j in range(len(candidates)):
                question = question + choices[j] + shuffled_list[j] + '\n'
            qa_data[question] = {'answer': correct_index, 'options': shuffled_list}

    elif dataset_name == 'CommonsenseQA':
        data = datasets.load_dataset('tau/commonsense_qa', split = 'validation')  # use validation set (test does not have group truth)
        for idx, item in enumerate(data):
            question = item['question'] + '\nChoose the best answer from the following options.\n'
            choices = item['choices']
            answer_key = item['answerKey']
            for i, character in enumerate(choices['label']):
                question +=  "(" + character + ') ' + choices['text'][i] + '\n'
            qa_data[question] = {'answer': answer_key, 'options': choices['label']}
        
    elif "business_ethics" in dataset_name.lower():
        test_df = pd.read_csv(data_path, header=None)
        for _, row in test_df.iterrows():
            raw_question = row[0]
            correct_answer = ord(row[5]) - ord("A")
            options = [row[i] for i in range(1, 5)]
            question = raw_question + '\n' +'Options: '
            for i in range(4):
                question += character[i] + " " + options[i] + "  "
            value = {
                'answer': correct_answer,
                'options': options
            }
            qa_data[question] = value

    elif dataset_name.lower() == 'professional':
        for file_name in os.listdir(data_path):
            if file_name.endswith('.csv'):
                file_path = os.path.join(data_path, file_name)
                print('Loading: ', file_path)
                test_df = pd.read_csv(file_path, header=None)
                for _, row in test_df.iterrows():
                    raw_question = row[0]
                    correct_answer = row[5]  # assuming letter answers as per commented part
                    options = [row[i] for i in range(1, 5)]
                    question = raw_question + '\nChoose the best answer from the following options.\n'
                    for i, option in enumerate(options):
                        question += character[i] + " " + options[i] + "\n"
                    value = {
                        'answer': correct_answer,
                        'options': options
                    }
                    qa_data[question] = value

    elif "professional_law" in dataset_name.lower() or "professional_medicine" in dataset_name.lower():
        test_df = pd.read_csv(data_path, header=None)
        for _, row in test_df.iterrows():
            raw_question = row[0]
            #correct_answer = ord(row[5]) - ord("A")
            correct_answer = row[5] # we use letter as answers not number
            options = [row[i] for i in range(1, 5)]
            #question = raw_question + '\n' +'Options: '
            question = raw_question + '\nChoose the best answer from the following options.\n'
            for i in range(4):
                question += character[i] + " " + options[i] + "\n"
            value = {
                'answer': correct_answer,
                'options': options
            }
            qa_data[question] = value


    elif "hybrid" in dataset_name.lower():
        with open(data_path, 'r') as f:
            qa_data = json.load(f)
            
    else:
        raise ValueError(f"{dataset_name} not supported")
    
    print('Total length: ', len(qa_data))
    return qa_data