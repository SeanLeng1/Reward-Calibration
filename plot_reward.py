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

def plot_preference(loc):
    sns.set_theme(style="whitegrid")
    def plot_comparison(data, categories, title, filename, save_dir, total_responses):
        plt.figure(figsize=(10, 8))
        ax = sns.barplot(x=categories, y=data, palette=['#f9bdb6', '#b7daf5'])
        for i, value in enumerate(data):
            ax.text(i, value + 0.001 * total_responses, f'{value/total_responses:.2%}({value})', color='black', ha='center')
        ax.set_title(title)
        #ax.set_xlabel('Preference Comparison')
        ax.set_ylabel('Count')
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, filename))
        plt.close()

    model_list = os.listdir(loc)
    for model in model_list:
        print(f'Processing {model}...')

        output_dir = f'{loc}/{model}/'
        save_dir = f'{loc}/{model}/plot'
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        os.makedirs(save_dir, exist_ok=True)

        modes = ['normal', 'rejected', 'chosen', 'both_high', 'both_low']
        probability_settings = ['', '_with_probability']

        for mode in modes:
            for prob_setting in probability_settings:
                file_pattern = f'scores{prob_setting}_{mode}.json'
                file_paths = glob.glob(os.path.join(output_dir, file_pattern))

                for file_path in file_paths:
                    if 'average_scores' not in file_path:  
                        with open(file_path, 'r') as f:
                            scores = json.load(f)

                        high_greater_count = sum(1 for d in scores if d['high_confidence_score'] > d['low_confidence_score'])
                        low_greater_count = sum(1 for d in scores if d['low_confidence_score'] > d['high_confidence_score'])
                        # ignore equal cases
                        total_responses = high_greater_count + low_greater_count

                        data = [high_greater_count, low_greater_count]
                        categories = ['High > Low (rejected)', 'Low > High (chosen)']

                        if mode == 'rejected':
                            title = 'Comparison of Preference Over Rejected Responses\nwith High/Low Confidence Scores'
                            categories = ['Prefer High Confidence (High > Low)', 'Prefer Low Confidence (Low > High)']
                        elif mode == 'chosen':
                            title = 'Comparison of Preference Over Chosen Responses\nwith High/Low Confidence Scores'
                            categories = ['Prefer High Confidence (High > Low)', 'Prefer Low Confidence (Low > High)']
                        elif mode == 'both_high':
                            title = 'Comparison of Preference Over Responses\nwith High Confidence Scores'
                            categories = ['Prefer Rejected Responses with High Confidence', 'Prefer Chosen Responses with High Confidence']
                        elif mode == 'both_low':
                            title = 'Comparison of Preference Over Responses\nwith Low Confidence Scores'
                            categories = ['Prefer Rejected Responses with Low Confidence', 'Prefer Chosen Responses with Low Confidence']
                        else:
                            if prob_setting:
                                title = 'Comparison of Preference Over Responses\nwith High/Low Confidence Scores'
                                categories = ['Prefer Rejected Responses with High Confidence', 'Prefer Chosen Responses with Low Confidence']
                            else:
                                title = 'Comparison of Preference Over Responses\n\n'
                                categories = ['Prefer Rejected Responses', 'Prefer Chosen Responses']

                        plot_filename = os.path.splitext(os.path.basename(file_path))[0] + '_plot.png'
                        plot_comparison(data, categories, title, plot_filename, save_dir, total_responses)

        # we also compare chosen_with_high and chosen
        # since the original data and confidence data have different prompt
        # it might be a good idea to directly compare them
        file_paths = (f'{output_dir}/scores_normal.json', f'{output_dir}/scores_with_probability_normal.json')
        chosen_path = f'{output_dir}/scores_with_probability_chosen.json'
        rejected_path = f'{output_dir}/scores_with_probability_rejected.json'
        for file_path in file_paths:
            if os.path.exists(file_path) and os.path.exists(chosen_path) and os.path.exists(rejected_path):
                with open(file_path, 'r') as f:
                    normal_scores = json.load(f)
                with open(chosen_path, 'r') as f:
                    chosen_scores = json.load(f)
                with open(rejected_path, 'r') as f:
                    rejected_scores = json.load(f)
                # chosen with chosen and high
                high_greater_count = 0
                low_greater_count = 0
                for d1, d2 in zip(normal_scores, chosen_scores):
                    if d1['low_confidence_score'] > d2['high_confidence_score']:
                        low_greater_count += 1
                    else:
                        high_greater_count += 1
                data = [high_greater_count, low_greater_count]
                total_responses = high_greater_count + low_greater_count
                categories = ['Prefer Chosen Responses With High Confidence', 'Prefer Original Chosen Responses']
                extra = ""
                file_name_extra = ""
                if 'probability' in file_path:
                    extra = 'on Modified User Prompt.' 
                    file_name_extra = "on_modified_user_prompt_"
                title = f'Comparison of Preference Over Chosen Responses\nwith High Confidence and Chosen Responses {extra}'
                plot_filename = f'scores_chosen_and_high_{file_name_extra}plot.png'
                plot_comparison(data, categories, title, plot_filename, save_dir, total_responses)
                # rejected with rejected and low
                high_greater_count = 0
                low_greater_count = 0
                for d1, d2 in zip(normal_scores, rejected_scores):
                    if d1['high_confidence_score'] > d2['low_confidence_score']:
                        low_greater_count += 1
                    else:
                        high_greater_count += 1
                data = [low_greater_count, high_greater_count]
                total_responses = high_greater_count + low_greater_count
                categories = ['Prefer Rejected Responses With Low Confidence', 'Prefer Original Rejected Responses']
                title = f'Comparison of Preference Over Rejected Responses\nwith Low Confidence and Rejected Responses {extra}'
                plot_filename = f'scores_rejected_and_low_{file_name_extra}plot.png'
                plot_comparison(data, categories, title, plot_filename, save_dir, total_responses)

def plot_average(loc):
    sns.set_theme(style="whitegrid")

    def load_scores(file_path):
        with open(file_path, 'r') as file:
            return json.load(file)

    def plot_comparison(data, categories, title, filename, save_dir):
        plt.figure(figsize=(10, 8))
        if len(categories) == 2:
            #color = ['#e3712e', '#7ac7e2']
            color = ['#f9bdb6', '#b7daf5']
        elif len(categories) == 3:
            #color = ['#e3712e', '#7ac7e2', '#54beaa']
            color = ['#f9bdb6', '#b7daf5', '#dbedc5']
        ax = sns.barplot(x=categories, y=list(data.values()), palette=color)
        for i, value in enumerate(data.values()):
            ax.text(i, value, f'{value:.2f}', color='black', ha='center', va='bottom')
        ax.set_title(title)
        ax.set_ylabel('Scores')
        plt.savefig(os.path.join(save_dir, filename))
        plt.close()

    model_list = os.listdir(loc)

    for model in model_list:
    #model = 'reward-model-deberta-v3-large-v2'
        print(f'Processing {model}...')
        
        output_dir = f'{loc}/{model}/'
        save_dir = f'{loc}/{model}/avg_plot'
        os.makedirs(save_dir, exist_ok=True)
        files = {
            'normal': 'average_scores_normal.json',
            'rejected_prob': 'average_scores_with_probability_rejected.json',
            'chosen_prob': 'average_scores_with_probability_chosen.json'
        }

        # if the folder does not contain all the necessary files, skip
        if not all([os.path.exists(os.path.join(output_dir, file)) for file in files.values()]):
            print(f'Skipping {model} due to missing files.')
            continue

        # Load data
        data_normal = load_scores(os.path.join(output_dir, files['normal']))
        data_rejected_prob = load_scores(os.path.join(output_dir, files['rejected_prob']))
        data_chosen_prob = load_scores(os.path.join(output_dir, files['chosen_prob']))

        # Prepare data for plotting
        data_rejected_comparison = {
            'Rejected': data_normal['average_high_confidence_score (rejected)'],
            'Rejected with High Confidence': data_rejected_prob['average_high_confidence_score (rejected)'],
            'Rejected with Low Confidence': data_rejected_prob['average_low_confidence_score (chosen)']
        }
        data_chosen_comparison = {
            'Chosen': data_normal['average_low_confidence_score (chosen)'],
            'Chosen with High Confidence': data_chosen_prob['average_high_confidence_score (rejected)'],
            'Chosen with Low Confidence': data_chosen_prob['average_low_confidence_score (chosen)']
        }

        # Plotting
        plot_comparison(data_rejected_comparison, ['Rejected', 'Rejected with High Confidence', 'Rejected with Low Confidence'],
                        'Comparison of Rejected Scores', 'rejected_comparison.png', save_dir)
        plot_comparison(data_chosen_comparison, ['Chosen', 'Chosen with High Confidence', 'Chosen with Low Confidence'],
                        'Comparison of Chosen Scores', 'chosen_comparison.png', save_dir)

        # Comparing high and low confidence for rejected and chosen separately
        plot_comparison(data_rejected_prob, ['High Confidence', 'Low Confidence'],
                        'Rejected: High vs Low Confidence', 'rejected_high_low_confidence.png', save_dir)
        plot_comparison(data_chosen_prob, ['High Confidence', 'Low Confidence'],
                        'Chosen: High vs Low Confidence', 'chosen_high_low_confidence.png', save_dir)



def plot_distribution(loc):
    model_list = os.listdir(loc)
    modes = ('chosen', 'rejected')

    add_line = True

    for model in model_list:
        print(f'Processing {model}...')
        for mode in modes:
            save_dir = f'{loc}/{model}/distribution_plot'
            os.makedirs(save_dir, exist_ok=True)
            path = f'{loc}/{model}/scores_with_probability_{mode}.json'

            if os.path.exists(path):
                print(f'Processing file: {path}') 
                with open(path, 'r') as file:
                    data = json.load(file)
            else:
                print(f'Skipping file {path}')
                continue

            high_scores = [item['high_confidence_score'] for item in data]
            low_scores = [item['low_confidence_score'] for item in data]

            bins = np.linspace(min(high_scores + low_scores), max(high_scores + low_scores), 100)

            high_hist, _ = np.histogram(high_scores, bins)
            low_hist, _ = np.histogram(low_scores, bins)

            bar_width = (bins[1] - bins[0]) * 0.4  

            plt.figure(figsize=(10, 6))

            high_bar_positions = bins[:-1] + bar_width / 2
            low_bar_positions = bins[:-1] + bar_width * 1.5

            plt.bar(high_bar_positions, high_hist, width=bar_width, label='High Confidence', color='#A9BBE0', align='center')
            plt.bar(low_bar_positions, low_hist, width=bar_width, label='Low Confidence', color='#F7C2CA', align='center')

            if add_line:
                kde_high = gaussian_kde(high_scores)
                kde_low = gaussian_kde(low_scores)
                x_range = np.linspace(min(bins), max(bins), 1000)
                high_kde_values = kde_high(x_range) * high_hist.sum() * (bins[1] - bins[0])
                low_kde_values = kde_low(x_range) * low_hist.sum() * (bins[1] - bins[0])

                plt.plot(x_range, high_kde_values, color="#A9BBE0", linewidth=2, linestyle="--")
                plt.plot(x_range, low_kde_values, color="#F7C2CA", linewidth=2, linestyle="--")

            plt.fontsize = 12
            plt.xlabel('Reward Scores')
            plt.ylabel('Frequency')
            upper_mode = mode[0].upper() + mode[1:]
            plt.title(f'Distribution of Reward Scores on {upper_mode} Responses with High and Low Confidence')
            plt.legend()
            plt.savefig(f'{save_dir}/{mode}_distribution_plot.png')
            plt.close()


def plot_ablations(loc):
    modes = ('_chosen', '_rejected')
    sides = ('_left', '_right')
    pure_numbers = ('_pure', '')
    prompts = ('_add_prompt', '')
    model_list = os.listdir(loc)
    for model in model_list:
        plt.figure(figsize=(10, 8))  
        for mode in modes:
            for side in sides:
                for prompt in prompts:
                    for pure in pure_numbers:
                        json_file = f'{loc}/{model}/ablation{mode}{side}{prompt}{pure}.json'
                        save_path = f'{loc}/{model}/comprehensive_ablation.png'  
                        if os.path.exists(json_file):
                            print(f'Processing file: {json_file}') 
                            with open(json_file, 'r') as f:
                                data = json.load(f)
                            average_score = {}
                            for key, value in data.items():
                                if isinstance(value[0], list):
                                    value = [v[0] for v in value]
                                average_score[key] = sum(value) / len(value)

                            plt.plot(list(average_score.keys()), list(average_score.values()), label=f'{mode}{side}{prompt}{pure}')
    
        plt.xlabel('Confidence Scores')
        plt.ylabel('Average Reward Scores')
        plt.title('How Average Reward Scores Vary with Confidence Levels Across Conditions')
        plt.legend() 
        plt.savefig(save_path)  
        plt.close() 

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--loc", type=str, default = 'reward_results/prompt/', help="path to result file")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    loc = args.loc
    print(loc)
    plot_preference(loc)
    print('Preference Done!')
    plot_average(loc)
    print('Average Done')
    plot_distribution(loc)
    print('Done!')
    # print('Plotting Ablations')
    # ablation_loc = '/storage1/jiaxinh/Active/jixuan/Model_Calibration/ablations/'
    # plot_ablations(ablation_loc)



