# Taming Overconfidence in LLMs: Reward Calibration in RLHF
<div align="center">
    <a href="https://huggingface.co/HINT-lab"><img src="https://huggingface.co/datasets/huggingface/badges/resolve/main/dataset-on-hf-sm-dark.svg"></a>
    <a href=""><img src="assets/Paper-Arxiv-orange.svg" ></a>
</div>

<div align="center">
<img src="assets/calibration-framework.png"  width="90%">
</div>

<div align="center">
<img src="assets/method-framework.png"  width="90%">
</div>

## Getting Started
### Installation
**Prepare the environment (recommend to follow the instructions and use separate (fresh) environments for each step)**
```
conda create -n calibration python=3.10.13 -y
```
Then install required packages
```
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121   # we used 2.3.1 with cu121
pip install -r requirements.yaml
pip install -e reward-bench     # for reward-bench evaluation
```
### Reproduce the Reward Model Experiment

This section is for demonstration purposes only. Please modify the bash scripts according to your directory structure. Additional models can be included by following the provided scripts:
```
cd Model_Calibration
bash scripts/general_scripts/rum_rm.sh          # reward model
bash scripts/general_scripts/rum_dpo.sh         # dpo model
```

**To plot the results**
```
python plot_reward_win_rate.py --loc reward_results/prompt --loc2 reward_results/no_prompt      # path to no prompt results is required for mode one: Answer_Only
```

### Run Reward Model/PPO/DPO Training

To conduct training, adhere to the package requirements specified by OpenRLHF. It is advisable to use a separate Conda environment for this purpose
```
conda create -n OpenRLHF python=3.10.14 -y
conda activate OpenRLHF
```
Then install required packages
```
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121    # we use 2.3.1 with cu121
pip install -r requirements.txt
pip install -e OpenRLHF
```

Should you encounter issues related to an outdated version of GCC on your server (as we did), the following steps will guide you to install and compile Deepspeed:
```
conda install gxx_linux-64 gxx_impl_linux-64 gcc_linux-64 gcc_impl_linux-64
alias gcc='/path/envs/OpenRLHF/bin/x86_64-conda-linux-gnu-cc'               # change this to your env path
pip uninstall deepspeed
pip cache remove deepspeed
DS_BUILD_FUSED_ADAM=1 DS_BUILD_CPU_ADAM=1 pip install deepspeed==0.14.4     # for offload and fused adam
```
Upon successful installation of Deepspeed, it is recommended to uninstall the GCC package to prevent potential conflicts with other installed packages."

> [!NOTE]
> Ray is not supported and Packing Samples is not supported in current version (You might want to check [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF/tree/main) for lastest update)


**Calibrate Reward Model**
```
cd Model_Calibration
bash scripts/general_scripts/train_crm.sh
```

**PPO and PPO-M**
```
cd Model_Calibration
bash scripts/general_scripts/train-ppo-llama.sh 
bash scripts/general_scripts/train-ppo-mistral.sh
```
To train with PPO-M, replace the ```reward_pretrain``` with the calibrated reward model.

**PPO-C**
```
cd Model_Calibration
bash scripts/general_scripts/train-ppo-c-llama.sh
bash scripts/general_scripts/train-ppo-c-mistral.sh
```
Ensure to use the regular reward model (pre-calibrated one) here.

**CDPO**
```
cd Model_Calibration
bash scripts/general_scripts/train-cdpo-llama.sh 
bash scripts/general_scripts/train-cdpo-mistral.sh 
```
A smaller beta typically yields better results (e.g., 0.01).


### Evaluation
**Configure the Environment: Create an ```api_key.yaml``` inside the scripts folder (not general_scripts), and set your ```OPENAI_API_KEY: [api_key]```. Then navigate to yhe main directory:**
```
cd Model_Calibration
```

**To Evaluate the Model**
```
bash scripts/general_scripts/query.sh  # Set USE_COT=true to enable zero-shot CoT. Default: Direct Answer (vanilla)
```

**To Parse the Results**
```
bash scripts/general_scripts/parse_eval.sh      # for regex parsing (used for Direct Answer)
bash scripts/general_scripts/gpt_eval.sh        # for GPT evaluation (used for Zero-shot CoT)
```
> [!NOTE]
> Adjust the scripts to fit your environment and models for evaluation. We recommend using ```batch_size=1``` for queries to minimize numerical instability during batch generation [Reference](https://github.com/huggingface/transformers/issues/25420#issuecomment-1775317535). Larger batch sizes may also necessitate manual adjustment of outputs due to the fact that stopping criteria is applied to the entire batch (which means some sequences might not stop properly).


## Installation of Evaluation Tools
### For FastChat:
```
git clone https://github.com/lm-sys/FastChat.git
cd FastChat
pip3 install --upgrade pip  # enable PEP 660 support
pip3 install -e ".[model_worker,webui]"
```

### For Arena-Hard-Auto:
```
git clone https://github.com/lm-sys/arena-hard.git
cd arena-hard
pip install -r requirements.txt
pip install -r requirements-optional.txt  # Optional dependencies (e.g., anthropic sdk)
```

> [!NOTE]
> Use a separate environment for Arena-Hard-Auto due to vllm specific requirements, such as particular Torch versions. Modify the ```gen_model_answer.py``` file to utilize a tokenizer chat template instead of the FastChat conversation template. Consider the model's data type (bf16). Our modified files, which accommodate the latest OpenAI API scheme and manage the presence of 'bos' tokens in chat templates, can be found in ```utils/fastchat_replacement``` and ```utils/arena-hard-auto-replacement```. Be aware that manual adjustments may be needed for templates intentionally lacking a 'bos' token.


## Dataset

The evaluation datasets are located in the ```dataset``` folder. While these datasets are hosted here for convenience, all credits for the creation and maintenance of these datasets go to their respective creators.

Please refer to the following links to access the original sources and adhere to their specified citation guidelines:

- [GSM8K](https://github.com/openai/grade-school-math)
- [SciQ](https://huggingface.co/datasets/allenai/sciq)
- [BBH](https://github.com/suzgunmirac/BIG-Bench-Hard)
- [CommonsenseQA](https://huggingface.co/datasets/tau/commonsense_qa)
- [TruthfulQA](https://huggingface.co/datasets/truthfulqa/truthful_qa)
- [MMLU](https://github.com/hendrycks/test?tab=readme-ov-file)


## References & Acknowledgements

Our codebase is built upon RewardBench for reward model testing, and we utilize OpenRLHF for reward model and RLHF training.
We express our sincere gratitude to these projects and their contributors. 
As OpenRLHF and FastChat are actively updating, for the most recent version, please kindly refer to the corresponding project page.

- [RewardBench ↗](https://github.com/allenai/reward-bench)
- [OpenRLHF ↗](https://github.com/OpenRLHF/OpenRLHF/tree/main)
- [FastChat ↗](https://github.com/lm-sys/FastChat)
- [LLM-uncertainty ↗](https://github.com/MiaoXiong2320/llm-uncertainty)
- [Arena-Hard-Auto ↗](https://github.com/lmarena/arena-hard-auto)