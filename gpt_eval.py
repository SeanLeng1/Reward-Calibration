import os, sys, json
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from argparse import ArgumentParser
from collections import Counter
from utils import compute_conf_metrics
import evaluate
from sklearn.calibration import calibration_curve
import re
from openai import OpenAI
import litellm
from litellm import batch_completion
import time
from tqdm import tqdm
import yaml

with open('scripts/api_key.yaml', 'r') as f:
    config = yaml.safe_load(f)
os.environ["OPENAI_API_KEY"] = config['OPENAI_API_KEY']

# API setting constants
API_MAX_RETRY = 5
API_RETRY_SLEEP = 1
API_ERROR_OUTPUT = "$ERROR$"

PROMPT = (
    "Evaluate the semantic equivalence between the given model response and the provided golden answer. Determine if they convey the same meaning.\n"
    "If the model response accurately matches the golden answer (i.e., the model response is correct), assign a score of 1. If the model response does not match the golden answer, assign a score of 0.\n"
    "Additionally, extract the confidence score from the model response. If the model response does not explicitly state a confidence score, return -100.\n"
    "Provide your answer in the following JSON format: {'correctness': 1 or 0, 'confidence': X.X}"
).strip()


# PROMPT_k = (
#     "Evaluate the semantic equivalence between the multiple answers provided in the model response and the provided golden answer. Determine if any of the answers convey the same meaning as the golden answer.\n"
#     "If at least one of the answers accurately matches the golden answer (i.e., any answer is correct), assign a score of 1. If none of the answers match the golden answer, assign a score of 0.\n"
#     "Additionally, extract the confidence score associated with each answer. If an answer does not explicitly state a confidence score, return -100 for that specific confidence.\n"
#     "Provide your answer in the following JSON format: {'correctness': 1 or 0, 'confidence1': X.X, 'confidence2': Y.Y}, where 'X.X' and 'Y.Y' are the confidence levels of the first and second responses, respectively."
# ).strip()


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--input_file", type=str, default="output/consistency/raw_results_input/BigBench_ObjectCounting_gpt3_2023-04-27_01-09_processed.json")
    parser.add_argument('--use_top_k', action='store_true', help='Use Top_K format for answers')
    parser.add_argument('--use_cot', action='store_true', help='Use COT format for answers')
    parser.add_argument("--model_name", type=str, default="llama")
    parser.add_argument("--dataset", type=str, default="GSM8K")
    parser.add_argument("--task_type", type=str, choices=["multi_choice_qa", "open_number_qa"], default="multi_choice_qa")
    parser.add_argument("--visual_folder", type=str, default="output/consistency/visuals/")
    parser.add_argument("--metric_folder", type=str, default="output/consistency/metrics/")
    parser.add_argument("--eval_model", type=str, default="gpt-4-turbo")
    parser.add_argument("--force_reeval", action="store_true", default=False)

    return parser.parse_args()


def api_call(model, messages):
    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            responses = batch_completion(
                model = model,
                messages = messages,
                n = 1,
                max_tokens = 256,
                temperature = 0.0,
                response_format={"type": "json_object"}
            )
            output = [response.choices[0].message.content for response in responses]
            break
        except Exception as e:
            print(e)
            time.sleep(API_RETRY_SLEEP)
    return output

def extract_correctness_and_confidence(args, model, model_responses, answers):
    messages = []
    for response, answer in zip(model_responses, answers): 
        messages.append([
            {"role": "system", "content": "You are a specialized evaluator designed to assess model responses against golden answers for various tasks and extract model confidence. Output your evaluation in JSON format"},
            {"role": "user", "content": PROMPT + f'\nModel Response: {response}\n\nGolden Answer: {answer}'}
        ])
    outputs = api_call(model, messages)
    parsed_outputs = []
    for output in outputs:
        json_output = json.loads(output)
        parsed_outputs.append(json_output)
    return parsed_outputs


def parse_result(args, model, result_list):
    correct = []
    predicted_confs = []
    model_responses = []
    golden_answers = []
    for result in tqdm(result_list, total=len(result_list)):
        golden_answer = result['target']['answer']
        model_response = result['response']
        model_responses.append(model_response)
        golden_answers.append(golden_answer)

    parsed_results = extract_correctness_and_confidence(args, model, model_responses, golden_answers)
    all_confidences = [float(res['confidence']) if 'confidence' in res else 0 for res in parsed_results if float(res['confidence']) != -100]
    most_common_confidence = Counter(all_confidences).most_common(1)[0][0]  
    print(f'Assigning {most_common_confidence} for failing cases')
    for parsed_result in parsed_results:
        correct.append(float(parsed_result['correctness']))
        if args.use_top_k:
            confidence = (float(parsed_result['confidence1']) + float(parsed_result['confidence2'])) / 2
        else:
            confidence = float(parsed_result['confidence'])
        if confidence == -100:  # not found
            confidence = most_common_confidence / 10.0
        if confidence <= 10.0:  # normal cases
            confidence = confidence / 10.0
        if 10 < confidence <= 100:  # corner cases
            confidence = confidence / 100.0
        if confidence > 100:        # corner cases
            confidence = 1.0
        predicted_confs.append(confidence)

    return correct, predicted_confs


#################### PLOT ECE DIAGRAM ####################
def plot_ece_diagram(args, y_true, y_confs, score_type):
    from netcal.presentation import ReliabilityDiagram
    n_bins = 10
    diagram = ReliabilityDiagram(n_bins)
    # if y_true and y_conf is not np array, convert them to np array
    if not isinstance(y_true, np.ndarray):
        y_true = np.array(y_true)
    if not isinstance(y_confs, np.ndarray):
        y_confs = np.array(y_confs)

    diagram.plot(y_confs, y_true, fig = plt.figure())

    #plt.legend(loc='upper left') 
    plt.savefig(os.path.join(args.visual_folder, f"ece_{score_type}.png"), dpi=600)
    plt.close()

    true_probs, predicted_probs = calibration_curve(y_true, y_confs, n_bins=10)
    plt.figure()
    plt.plot(predicted_probs, true_probs, marker='o')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated')
    plt.xlabel('Mean predicted probability')
    plt.ylabel('Fraction of positives')
    plt.title('Calibration plot (Reliability Diagram)')
    plt.savefig(os.path.join(args.visual_folder, f"calibration_curve.png"), dpi=600)
    plt.close()


# TODO: can remove this, i do not think this is useful
def plot_confidence_histogram(args, y_true, y_confs, score_type, acc, auroc, ece, use_annotation=True):
    plt.figure(figsize=(8, 6))
    corr_confs = [y_confs[i] * 100 for i in range(len(y_confs)) if y_true[i] == 1]
    wrong_confs = [y_confs[i] * 100 for i in range(len(y_confs)) if y_true[i] == 0]

    corr_counts = [corr_confs.count(i) for i in range(101)]
    wrong_counts = [wrong_confs.count(i) for i in range(101)]

    correct_color = plt.cm.tab10(0)
    wrong_color = plt.cm.tab10(3)

    n_wrong, bins_wrong, patches_wrong = plt.hist(wrong_confs, bins=21, alpha=0.8, label='wrong answer', color=wrong_color, align='mid', range=(-2.5,102.5))
    n_correct, bins_correct, patches_correct = plt.hist(corr_confs, bins=21, alpha=0.8, label='correct answer', color=correct_color, align='mid', range=(-2.5,102.5), bottom=np.histogram(wrong_confs, bins=21, range=(-2.5,102.5))[0])

    tick_set = [i * 10 for i in range(5, 11)]

    annotation_correct_color = 'black'
    annotation_wrong_color = 'red'
    annotation_texts = []
    if args.use_cot:
        plt.title(f"COT: ACC {acc:.2f} / AUROC {auroc:.2f} / ECE {ece:.2f} on {args.dataset}", fontsize=16, fontweight='bold')
    else:
        plt.title(f"ACC {acc:.2f} / AUROC {auroc:.2f} / ECE {ece:.2f} on {args.dataset}", fontsize=16, fontweight='bold')

    plt.ylim(0, 1.1*max(n_correct+n_wrong))
    plt.xticks(tick_set, fontsize=16, fontweight='bold')
    plt.yticks([])
    plt.xlabel("Confidence (%)", fontsize=16, fontweight='bold')
    plt.ylabel("Count", fontsize=16, fontweight='bold')
    plt.legend(loc='upper left', prop={'weight':'bold', 'size':16})
    plt.tight_layout()
    plt.savefig(os.path.join(args.visual_folder, f"auroc_{score_type}.png"), dpi=600)
    #plt.close()


#################### PLOT CONFIDENCE DISTRIBUTION ####################
def plot_confidence_distribution(args, y_confs, accuracy):
    avg_conf = np.mean(y_confs)
    plt.figure(figsize=(10, 5))
    
    # lets use 20, otherwise 0.9 and 1.0 are in the same bin
    #counts, bin_edges = np.histogram(y_confs, bins=20, range=(0, 1))
    bins = np.array([0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
                     0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.10])
    counts, bin_edges = np.histogram(y_confs, bins=bins)
    freqs = counts / sum(counts)
    
    plt.bar(bin_edges[:-1], freqs, width=np.diff(bin_edges), align='center', edgecolor='black')

    plt.ylim(0, 1.1)  
    plt.xlim(-0.05, 1.1)
    

    plt.axvline(x=accuracy, color='black', linestyle='--', linewidth=2, label='Avg. Accuracy')
    plt.axvline(x=avg_conf, color='gray', linestyle='--', linewidth=2, label='Avg. Confidence')

    plt.legend(['Avg. Accuracy', 'Avg. Confidence', 'Relative Amount of Samples'], loc='upper center', fontsize=12)
    plt.title(f'Confidence Distribution Histogram on {args.dataset}', fontsize=22)
    plt.xlabel('Confidence', fontsize=18)
    plt.ylabel('% of Samples', fontsize=18)
    plt.xticks(np.arange(0, 1.1, 0.1))

    plt.grid(False)
    plt.tight_layout()

    save_path = os.path.join(args.visual_folder, "confidence_distribution.png")
    plt.savefig(save_path, dpi=600)
    plt.close()

#################### PLOT ACCURACY WITHIN BINS ####################
def plot_accuracy_within_bins(args, y_true, y_preds, n_bins = 20):
    bins = np.linspace(0, 1.0, n_bins+1)
    # bins = np.array([0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
    #                  0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05])
    #n_bins = len(bins) - 1
    accuracies = np.zeros(n_bins)
    counts = np.zeros(n_bins)
    mean_confidence = np.zeros(n_bins)

    for i in range(n_bins):
        indices = (y_preds >= bins[i]) & (y_preds < bins[i+1])
        if i == n_bins - 1:  
            indices = (y_preds >= bins[i]) & (y_preds <= bins[i+1])
        
        if indices.any():
            accuracies[i] = np.mean(y_true[indices])  
            mean_confidence[i] = np.mean(y_preds[indices])
            counts[i] = np.sum(indices)

    # map to color
    #norm = plt.Normalize(vmin=0, vmax=1)
    norm = plt.Normalize(vmin=counts.min(), vmax=counts.max())
    cmap = plt.get_cmap('Blues')
    
    plt.figure(figsize = (12, 5))
    #colors = cmap(norm(mean_confidence))
    colors = cmap(norm(counts))
    plt.bar(bins[:-1], accuracies, width=np.diff(bins), align='edge', edgecolor='black', color=colors)
    plt.plot([0, 1.0], [0, 1.0], 'r--', label='Perfect Calibration')
    plt.ylim(0, 1.0)  
    plt.xlim(0, 1.0)
    plt.xticks(np.arange(0, 1.1, 0.1), fontsize=25)
    plt.yticks(fontsize=25)
    plt.xlabel('Confidence', fontsize=25)
    plt.ylabel('Accuracy Within Bins', fontsize=25)
    plt.title(f'Accuracy Within Bins vs. Confidence on {args.dataset}\nfor {args.model_name}', fontsize=25)
    plt.tight_layout()
    plt.savefig(os.path.join(args.visual_folder, "accuracy_within_bins.pdf"), dpi=600)

#################### COMPUTE ACC/ECE/AUCROC ####################
def compute_metrics(args, correct, predicted_confs):
    avg_accuracy = sum(correct) / len(correct)

    result = compute_conf_metrics(correct, predicted_confs)
    plot_confidence_histogram(args, correct, predicted_confs, "confidence", result["accuracy"], result["roc_auc"], result["ece"], use_annotation=True)
    
    plot_ece_diagram(args, correct, predicted_confs, "confidence")
    plot_confidence_distribution(args, predicted_confs, avg_accuracy)
    plot_accuracy_within_bins(args, np.array(correct), np.array(predicted_confs), n_bins=20)
    
    os.makedirs(args.metric_folder, exist_ok = True)
    metric_file = os.path.join(args.metric_folder, 'metric.json')
    with open(metric_file, 'w') as file:
        json.dump(result, file, indent=4)  


def check_confidence(confidence_list):
    return [1.0 if x >= 100.0 else x for x in confidence_list]

def main():
    args = parse_args()
    if os.path.exists(os.path.join(args.metric_folder, 'metric.json')) and not args.force_reeval:
        print('Already done')
        exit()
    else:
        print('Processing')
    try:
        with open(args.input_file, "r") as f:
            result_list = json.load(f)
    except json.decoder.JSONDecodeError as e:
        print(f"Failed to decode JSON: {e}")
        print(f"Error occurred at line {e.lineno}, column {e.colno}")
    except FileNotFoundError:
        print(f"File not found: {args.input_file}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    model = args.eval_model

    # if result_list is already parsed, we do not need to gpt again, it is costly
    if result_list[0]['confidence'] and not args.force_reeval:
        print(f"Already parsed by GPT")
        correct = [result['correctness'] for result in result_list]
        predicted_confs = [result['confidence'] for result in result_list]
        predicted_confs = check_confidence(predicted_confs)
    else:
        correct, predicted_confs = parse_result(args, model, result_list)
        predicted_confs = check_confidence(predicted_confs)
    for result, correctness, conf in zip(result_list, correct, predicted_confs):
        result['correctness'] = correctness
        result['confidence'] = conf
    with open(args.input_file, "w") as f:
        json.dump(result_list, f, indent=4)
    print(f"Processed results saved to {args.input_file}")


    os.makedirs(args.visual_folder, exist_ok=True)
    print(f"Processing {args.input_file} and saving results to {args.visual_folder}")
    compute_metrics(args, correct, predicted_confs)

if __name__ == "__main__":
    main()  





