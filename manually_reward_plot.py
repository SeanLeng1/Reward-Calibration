import os, sys, json
import numpy as np
import matplotlib 
import matplotlib.pyplot as plt
import pandas as pd
from argparse import ArgumentParser
from utils import compute_conf_metrics
from sklearn.calibration import calibration_curve
import seaborn as sns
from matplotlib.colors import ListedColormap
import glob
import shutil
from matplotlib.patches import Patch, Rectangle
from matplotlib.legend_handler import HandlerBase

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--model_1", type=str, default="Llama-3-8b-rm-mixture")
    parser.add_argument("--model_2", type=str, default="llama3-8b-crm-final-v0.1")
    parser.add_argument("--mode", type=str, default="chosen")
    parser.add_argument("--visual_folder", type=str, default="combined_plots/")
    return parser.parse_args()

def plot_comparison(data1, data2, categories, title, filename, save_dir, model_1_name, model_2_name):
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    width = 0.35  
    model_2_name = 'Llama3-8b-crm (ours)'

    #colors1 = ['#71b7ed', '#f18c25']
    #colors2 = ['#f57c6e', '#faeca8']


    colors1 = ['#71b7ed', '#71b7ed']
    colors2 = ['#f57c6e', '#f57c6e']


    bars1 = []
    bars2 = []

    x = range(len(categories))

    for i, val in enumerate(data1):
        bar = ax.bar(x[i] - width/2, val, width, color=colors1[i], alpha=0.5, label=f"{model_1_name} {categories[i]}" if i == 0 else "")
        bars1.append(bar)

    for i, val in enumerate(data2):
        bar = ax.bar(x[i] + width/2, val, width, color=colors2[i], alpha=0.5, label=f"{model_2_name} {categories[i]}" if i == 0 else "")
        bars2.append(bar)

    ax.set_title(title, fontsize=25)
    ax.set_ylabel('Count', fontsize=25)
    ax.tick_params(axis='y', labelsize=25)
    ax.set_xticks(x)  
    ax.set_xticklabels(categories, fontsize=25)  
    plt.yticks(fontsize=18)

    legend_elements = [
        Patch(facecolor=colors1[0], label=model_1_name),
        Patch(facecolor=colors2[0], label=model_2_name)
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=20)

    plt.tight_layout()
    print('Saving plot to', os.path.join(save_dir, filename))
    plt.savefig(os.path.join(save_dir, filename))
    plt.close()

def plot_preference(json_file1, json_file2, mode, save_dir, model_1_name, model_2_name):
    with open(json_file1, 'r') as f:
        scores1 = json.load(f)
    with open(json_file2, 'r') as f:
        scores2 = json.load(f)

    high_greater_count1 = sum(1 for d in scores1 if d['high_confidence_score'] > d['low_confidence_score'])
    low_greater_count1 = sum(1 for d in scores1 if d['low_confidence_score'] > d['high_confidence_score'])
    data1 = [high_greater_count1, low_greater_count1]

    high_greater_count2 = sum(1 for d in scores2 if d['high_confidence_score'] > d['low_confidence_score'])
    low_greater_count2 = sum(1 for d in scores2 if d['low_confidence_score'] > d['high_confidence_score'])
    data2 = [high_greater_count2, low_greater_count2]

    categories = {
        'normal': ['Prefer High', 'Prefer Low'],
        'no_prompt': ['Prefer High', 'Prefer Low'],
        'rejected': ['Prefer High Confidence\n(High > Low)', 'Prefer Low Confidence\n(Low > High)'],
        'chosen': ['Prefer High Confidence\n(High > Low)', 'Prefer Low Confidence\n(Low > High)'],
        'both_high': ['Prefer Rejected with High Confidence', 'Prefer Chosen with High Confidence'],
        'both_low': ['Prefer Rejected with Low Confidence', 'Prefer Chosen with Low Confidence'],
    }[mode]

    title = {
        'normal': 'Comparison of Preference Over Responses',
        'no_prompt': 'Comparison of Preference Over Responses',
        'rejected': 'Comparison of Preference Over Rejected Responses\nwith High/Low Confidence Scores',
        'chosen': 'Comparison of Preference Over Chosen Responses\nwith High/Low Confidence Scores',
        'both_high': 'Comparison of Preference Over Responses\nwith High Confidence Scores',
        'both_low': 'Comparison of Preference Over Responses\nwith Low Confidence Scores'
    }[mode]

    plot_filename = f'comparison_{mode}_{model_1_name}.png'
    plot_comparison(data1, data2, categories, title, plot_filename, save_dir, model_1_name, model_2_name)




def main(args):
    model_1_name = args.model_1
    model_2_name = args.model_2
    mode = args.mode
    prompt = 'prompt'
    if mode == 'normal':
        file = 'scores_with_probability_normal.json'
    elif mode == 'chosen':
        file = 'scores_with_probability_chosen.json'
    elif mode == 'rejected':
        file = 'scores_with_probability_rejected.json'
    elif mode == 'no_prompt':
        file = 'scores_normal.json'
        prompt = 'no_prompt'

    print(file)

    plot_preference(f'reward_results/{prompt}/{model_1_name}/{file}', f'reward_results/{prompt}/{model_2_name}/{file}', mode, 'combined_plots/', model_1_name, model_2_name)


if __name__ == '__main__':
    main(parse_args())
