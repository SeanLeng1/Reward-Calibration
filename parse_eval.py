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

exact_match = evaluate.load('./utils/exact_match.py')

option_list = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--input_file", type=str, default="output/consistency/raw_results_input/BigBench_ObjectCounting_gpt3_2023-04-27_01-09_processed.json")
    parser.add_argument('--use_cot', action='store_true', help='Use COT format for answers')
    parser.add_argument('--use_top_k', action='store_true', help='Use Top_K format for answers')
    parser.add_argument("--dataset", type=str, required=True, default="GSM8K")
    parser.add_argument("--model_name", type=str, default="llama")
    parser.add_argument("--task_type", type=str, required=True, choices=["multi_choice_qa", "open_number_qa"], default="multi_choice_qa")
    parser.add_argument("--visual_folder", type=str, required=True, default="output/consistency/visuals/")
    parser.add_argument("--metric_folder", type=str, required=True, default="output/consistency/metrics/")
    parser.add_argument("--force_reeval", action="store_true", default=False)

    return parser.parse_args()

# Since predictions might be None, we need to clean them up
# convert to string
def clean_predictions(predictions):
    cleaned_predictions = [str(pred) if pred is not None else '' for pred in predictions]
    return cleaned_predictions

def gather_results(result_list, task_type='multi_choice_qa', prompt_type='vanilla'):
    # for each question in the dataset, get their answers and the corresponding confidence
    # result dict should be a list of dics
    score_dicts = {
        "real_answers": [],
        "predicted_answers": [],
        "scores":[],
    }
    """
    {
        "question": list(qa_data.keys())[idx],
        "response": output,
        "target": target,
        "probabilities": prob,
        "confidence": confidence,
        "answer": answer,
    }
    """

    all_scores = [float(result["confidence"]) for result in result_list if result["confidence"] is not None]
    if all_scores:
        most_common_confidence = Counter(all_scores).most_common(1)[0][0]
    else:
        most_common_confidence = 5.0
    print(f'Assigning {most_common_confidence} for failing cases')

    for result in result_list:
        real_answer = result["target"]['answer']
        predicted_answer = result["answer"]

        if predicted_answer is None:
            predicted_answer = ''

        if task_type == 'multi_choice_qa':
            if isinstance(real_answer, int):
                real_answer = option_list[real_answer]

        score_dicts['real_answers'].append(real_answer)
        score_dicts['predicted_answers'].append(predicted_answer)

        # here we need to handle the parsing issues since LLM might not always follow the same format
        # if we cannot find a valid confidence score from the response
        # we use a most common ones as default (scale is 0 to 10, but we need to convert it to 0 to 1 for sklearn)
        scores = result["confidence"]
        if scores is None:
            scores = most_common_confidence / 10.0
        else:
            scores = float(scores)
            if scores <= 10.0:
                scores = scores / 10.0
            elif scores >= 10 and scores <= 100:              # sometimes the model output percentage even though we required 0-10 in prompts
                scores = scores / 100.0
            elif scores > 100:                                # sometimes model output big confidence (maybe due to repetition??) we assume these big confidence just means 1.0
                scores = 1.0
            else:
                scores = 0.5        # invalid scores (sometimes this happens)
        score_dicts['scores'].append(scores)

    return score_dicts


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
    elif args.use_top_k:
        plt.title(f"Top_K: ACC {acc:.2f} / AUROC {auroc:.2f} / ECE {ece:.2f} on {args.dataset}", fontsize=16, fontweight='bold')
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
    plt.subplots_adjust(right=0.15)
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
def compute_metrics(args, score_dicts):
    real_answers = score_dicts['real_answers']
    predicted_answers = score_dicts['predicted_answers']
    predicted_confs = score_dicts['scores']
    #correct = [real_answers[i] == predicted_answers[i] for i in range(len(real_answers))]

    predicted_answers = clean_predictions(predicted_answers)
    em = exact_match.compute(predictions=predicted_answers, references=real_answers, ignore_case=True, ignore_punctuation=True)
    correct = em['score_list']
    # convert correct from bool to float for the following calculation
    correct = [float(x) for x in correct]
    avg_accuracy = em['exact_match']

    result = compute_conf_metrics(correct, predicted_confs)
    plot_confidence_histogram(args, correct, predicted_confs, "confidence", result["accuracy"], result["roc_auc"], result["ece"], use_annotation=True)
    
    plot_ece_diagram(args, correct, predicted_confs, "confidence")
    plot_confidence_distribution(args, predicted_confs, avg_accuracy)
    plot_accuracy_within_bins(args, np.array(correct), np.array(predicted_confs), n_bins=20)
    
    os.makedirs(args.metric_folder, exist_ok = True)
    metric_file = os.path.join(args.metric_folder, 'metric.json')
    with open(metric_file, 'w') as file:
        json.dump(result, file, indent=4)  


def is_valid_confidence(confidence):
    if confidence > 100:
        return None
    if '.' in str(confidence):
        # if the decimal is not 0.5 or 0, we consider it invalid
        if str(confidence).split('.')[1] != '5' and str(confidence).split('.')[1] != '0':
            return None
    if confidence > 10:
        return confidence / 10
    return confidence

# This is a bit messy
# Actually direct_answer_match might be enough
def parse_result(args, result_list):
    json_outputs = []
    for result in result_list:
        output = result["response"]
        # extract confidence and answer from the output
        answer = None
        confidence = None
        direct_answer_match = re.search(r"Answer:\s*?\n?\(?([A-Z])\)?", output) if args.task_type == "multi_choice_qa" else re.search(r"Answer:\s*?\n?([-+]?\$?\d{1,3}(?:,\d{3})*\.?\d*)", output)
        if direct_answer_match:
            answer = direct_answer_match.group(1)
            answer = answer.replace(',', '').replace('$', '')
        else:
            answer_match = re.search(r"(?:Answer:\s*(?:\n)?)?(.*?)(Confidence:|$)", output, re.DOTALL)
            if answer_match:
                answer_content = answer_match.group(1).strip()
                if args.task_type != "multi_choice_qa":
                    all_numbers = re.findall(r'[-+]?\$?\d+(?:,\d{3})*(?:\.\d+)?', answer_content)
                    if all_numbers:
                        answer = all_numbers[-1].replace(',', '')
                        answer = answer.replace('$', '')
                else:
                    bracket_letters = re.findall(r'\(([A-Z])\)', answer_content)
                    if bracket_letters:
                        answer = bracket_letters[-1]  
                    else:
                        right_bracket_letters = re.findall(r'\b([A-Z])\)', answer_content)
                        if right_bracket_letters:
                            answer = right_bracket_letters[-1]
                        else:
                            all_letters = re.findall(r'(?<![\(\)])\b([A-Z])\b', answer_content)
                            if all_letters:
                                answer = all_letters[-1]

        confidence_match = re.search(r"Confidence:[:]*\s*(\d+(?:\.\d+)?)", output)
        if confidence_match:
            confidence = float(confidence_match.group(1))
        else:
            # try to extract the last number after the newline as confidence
            # some case: 12\n\n10
            newline_split = output.split('\n')
            if len(newline_split) > 1:
                possible_confidence = newline_split[-1].strip()
                confidence_numbers = re.findall(r'[-+]?\d+(?:\.\d+)?', possible_confidence)
                if confidence_numbers:
                    confidence = float(confidence_numbers[-1])
                    confidence = is_valid_confidence(confidence)

        # only modify parsed answer and confidence
        result['answer'] = answer
        result['confidence'] = confidence
        json_outputs.append(result)

    return json_outputs


def main():
    args = parse_args()
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

    if result_list[0]['confidence'] and not args.force_reeval:
        print(f"Already parsed")
    else:
        result_list = parse_result(args, result_list)
        # overwrite the input file with the parsed results
        with open(args.input_file, "w") as f:
            json.dump(result_list, f, indent=4)
        print(f"Processed results saved to {args.input_file}")

    os.makedirs(args.visual_folder, exist_ok=True)
    print(f"Processing {args.input_file} and saving results to {args.visual_folder}")

    score_dicts = gather_results(result_list, task_type=args.task_type)
    compute_metrics(args, score_dicts)

if __name__ == "__main__":
    main()  