import os, sys, json
import numpy as np
import matplotlib 
import matplotlib.pyplot as plt
import pandas as pd
from argparse import ArgumentParser
from collections import Counter
from utils import compute_conf_metrics
import evaluate
from sklearn.calibration import calibration_curve
import re
from collections import Counter
import seaborn as sns
from matplotlib.colors import ListedColormap

def plot_confidence_distribution(args, y_confs_1, y_confs_2, accuracy_1, accuracy_2, model_1_name, model_2_name):
    # plt.rc('text', usetex=True)
    # plt.rc('font', family='serif', serif='Times')

    plt.figure(figsize=(12, 5))
    sns.set_theme(style="whitegrid")
    bins = np.array([0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
                     0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.10])

    colors = ['#f57c6e', '#71b7ed']

    avg_conf_1 = np.mean(y_confs_1)
    counts_1, bin_edges = np.histogram(y_confs_1, bins=bins)
    freqs_1 = counts_1 / sum(counts_1)
    plt.bar(bin_edges[:-1], freqs_1, width=np.diff(bin_edges), align='center', alpha=0.5, color=colors[0], label=f'{model_1_name}')

    avg_conf_2 = np.mean(y_confs_2)
    counts_2, _ = np.histogram(y_confs_2, bins=bins)
    freqs_2 = counts_2 / sum(counts_2)
    plt.bar(bin_edges[:-1], freqs_2, width=np.diff(bin_edges), align='center', alpha=0.5, color=colors[1], label=f'{model_2_name}')

    #plt.axvline(x=accuracy_1, color=colors[0], linestyle='--', linewidth=2, label='Model 1 Avg. Accuracy')
    #plt.axvline(x=avg_conf_1, color=colors[0], linestyle=':', linewidth=2, label=f'{model_1_name} Avg. Confidence')

    #plt.axvline(x=accuracy_2, color=colors[1], linestyle='--', linewidth=2, label='Model 2 Avg. Accuracy')
    #plt.axvline(x=avg_conf_2, color=colors[1], linestyle=':', linewidth=2, label=f'{model_2_name} Avg. Confidence')

    plt.legend(loc='upper center', fontsize=22)
    plt.title(f'Confidence Distribution Histogram for\n{model_1_name} and {model_2_name} on {args.dataset}', fontsize=22)
    plt.xlabel('Confidence', fontsize=22)
    plt.ylabel('% of Samples (log scale)', fontsize=22)
    plt.yscale('log')
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.xticks(fontsize=18)  
    plt.yticks(fontsize=18) 
    plt.ylim(0.01, 1.1)
    plt.xlim(-0.05, 1.1)
    plt.grid(False)
    plt.tight_layout()

    save_path = os.path.join(args.visual_folder, f"{model_1_name}_{model_2_name}_{args.dataset}_confidence_distribution.png")
    print('Saving to ', save_path)
    plt.savefig(save_path, dpi=600)
    plt.close()



def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--model_1", type=str, default="llama3-8b-final-ppo-clean-v0.1")
    parser.add_argument("--model_2", type=str, default="llama3-8b-final-ppo-m-v0.3")
    parser.add_argument("--dataset", type=str, default="SciQ")
    parser.add_argument("--visual_folder", type=str, default="combined_plots-2/")
    return parser.parse_args()

def main(args):
    os.makedirs(args.visual_folder, exist_ok=True)
    for dataset in ['CommonsenseQA', 'SciQ', 'TruthfulQA', 'GSM8K', 'BigBench', 'professional']:
        args.dataset = dataset
        path_1 = f'verbalized_output-2/cot_dpo/{args.model_1}/{args.dataset}/results.json'
        path_2 = f'verbalized_output-2/cot_dpo/{args.model_2}/{args.dataset}/results.json'
        with open(path_1, "r") as f:
            result_list_1 = json.load(f)
        with open(path_2, "r") as f:
            result_list_2 = json.load(f)
        confidence_1 = []
        accuracy_1 = []
        for result in result_list_1:
            confidence = result['confidence']
            if confidence > 1.0:
                confidence /= 10.0
            confidence_1.append(confidence) 
            accuracy_1.append(result['target']['answer'] == result['answer'])
            
        confidence_2 = []
        accuracy_2 = []
        for result in result_list_2:
            confidence = result['confidence']
            if confidence > 1.0:
                confidence /= 10.0
            confidence_2.append(confidence)
            accuracy_2.append(result['target']['answer'] == result['answer'])

        accuracy_1 = np.mean(accuracy_1)
        accuracy_2 = np.mean(accuracy_2)
        # print(f'Accuracy of {args.model_1}: {accuracy_1}')
        # print(f'Accuracy of {args.model_2}: {accuracy_2}')
        # Plot distributions

        # model_1 = 'Llama3-8b-ppo'
        # if args.model_2 == 'llama3-8b-final-ppo-m-v0.3':
        #     model_2 = 'Llama3-8b-ppo-m'
        # elif args.model_2 == 'llama3-8b-final-ppo-c-v0.3':
        #     model_2 = 'Llama3-8b-ppo-c'

        # model_1 = 'Mistral-7b-ppo'
        # if args.model_2 == 'mistral-7b-ppo-m-hermes':
        #     model_2 = 'Mistral-7b-ppo-m'
        # elif args.model_2 == 'mistral-7b-ppo-c-hermes':
        #     model_2 = 'Mistral-7b-ppo-c'

        # model_1 = "Mistral-7b-dpo"
        # model_2 = "Mistral-7b-cdpo"

        model_1 = "Llama-3-8b-dpo"
        model_2 = "Llama-3-8b-cdpo"



        plot_confidence_distribution(args, confidence_1, confidence_2, accuracy_1, accuracy_2, model_1, model_2)


if __name__ == "__main__":
    args = parse_args()
    main(args)  