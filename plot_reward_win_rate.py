import os
import glob
import json
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from scipy.stats import gaussian_kde
import argparse
import numpy as np
import shutil
warnings.filterwarnings("ignore")

def plot_stack_comparison(all_data, categories, title, filename, save_dir):
    """
    Draw a single stacked bar chart for multiple modes.
    """
    # font use latex and times new roman
    # plt.rc('text', usetex=True)
    # plt.rc('font', family='serif', serif='Times')
    labels = list(all_data.keys())
    data_array = np.array(list(all_data.values()))
    totals = data_array.sum(axis=1)

    data_cum = data_array.cumsum(axis=1)
    category_colors = ['#f57c6e', '#71b7ed']  
    category_labels = {
        'confidence\nreversed': ('prefer high-confidence rejected', 'prefer low-confidence chosen'),
        'chosen\nwith_conf': ('prefer high confidence', 'prefer low confidence'),
        'rejected\nwith_conf': ('prefer high confidence', 'prefer low confidence'),
        'answer\nonly': ('prefer rejected', 'prefer chosen')
    }

    fig, ax = plt.subplots(figsize=(12, 1.6 * len(labels)))
    ax.invert_yaxis()
    #ax.xaxis.set_visible(False)
    ax.set_xlim(0, np.sum(data_array, axis=1).max())
    ax.spines['top'].set_visible(False)
    #ax.set_ylim(0, len(categories)) 

    for i in range(len(labels)):
        positions = [768.5, 2259.5]
        for j, (colname, color, position) in enumerate(zip(categories, category_colors, positions)):
            width = data_array[i, j]
            start = data_cum[i, j] - width
            rect = ax.barh(labels[i], width, left=start, height=0.5, label=colname if i == 0 else "", color=color, alpha=0.5)
            value = data_array[i, j]
            total = totals[i]
            percentage = f'{value / total * 100:.2f}%'
            x = rect[0].get_x() + rect[0].get_width() / 2
            y = rect[0].get_y() + rect[0].get_height() / 2
            ax.text(x, y, f'{value}', ha='center', va='center', fontsize=22)
            # Adjust text position under the bar
            ax.text(position, rect[0].get_y() - 0.32, category_labels[labels[i]][j], ha='center', va='top', fontsize=22, color=color)

    #ax.legend(ncols=1, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.title(title, fontsize=22, y=1.1)
    plt.tight_layout()
    plt.subplots_adjust(left=0.15)
    plt.yticks(fontsize=22)
    plt.xticks(fontsize=22)
    plt.savefig(os.path.join(save_dir, filename), dpi=600)
    plt.close()


def plot_reference(loc, loc2):
    model_list = os.listdir(loc)
    for model in model_list:
        print(f'Processing {model}...')
        output_dir = f'{loc}/{model}/'
        save_dir = f'{loc}/{model}/win_rate_plot'
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        os.makedirs(save_dir, exist_ok=True)

        all_data = {}
        modes = ['normal', 'chosen', 'rejected']
        name = ['confidence\nreversed', 'chosen\nwith_conf', 'rejected\nwith_conf']
        for mode, name in zip(modes, name):
            file = os.path.join(loc, f'{model}/scores_with_probability_{mode}.json')
            if not os.path.exists(file):
                continue
            with open(file, 'r') as f:
                scores = json.load(f)
            high_greater_count = sum(1 for d in scores if d['high_confidence_score'] > d['low_confidence_score'])
            low_greater_count = sum(1 for d in scores if d['low_confidence_score'] > d['high_confidence_score'])
            all_data[name] = [high_greater_count, low_greater_count]

        # add no_prompt normal
        file = os.path.join(loc2, f'{model}/scores_normal.json')
        if not os.path.exists(file):
            continue
        with open(file, 'r') as f:
            scores = json.load(f)
        high_greater_count = sum(1 for d in scores if d['high_confidence_score'] > d['low_confidence_score'])
        low_greater_count = sum(1 for d in scores if d['low_confidence_score'] > d['high_confidence_score'])
        all_data['answer\nonly'] = [high_greater_count, low_greater_count]

        title = f'Comparison of Preference Over Responses\nfor {model}'
        categories = ['Prefer Rejected Or High Confidence Responses', 'Prefer Chosen Or Low Confidence Responses']
      
        
        plot_stack_comparison(all_data, categories, title, f'comparison_{model}.pdf', save_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--loc', type=str, default='reward_results/prompt/')
    parser.add_argument('--loc2', type=str, default='reward_results/no_prompt/')
    args = parser.parse_args()
    plot_reference(args.loc, args.loc2)

    

    

