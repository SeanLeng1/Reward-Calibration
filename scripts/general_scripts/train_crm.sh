#!/bin/bash
cd OpenRLHF
export WANDB__SERVICE_WAIT=300

set -x 

read -r -d '' training_commands <<EOF
openrlhf.cli.train_crm \
   --save_path rm_checkpoints/llama3-8b-crm-final-v0.2 \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps -1 \
   --train_batch_size 256 \
   --micro_train_batch_size 1 \
   --pretrain OpenLLMAI/Llama-3-8b-rm-mixture \
   --bf16 \
   --max_epochs 1 \
   --max_len 8192 \
   --zero_stage 2 \
   --learning_rate 9e-6 \
   --dataset HINT-lab/calibration_preference_mixture_final-v0.1 \
   --dataset_probs 1.0 \
   --max_samples 10000000000 \
   --apply_chat_template \
   --chosen_key chosen \
   --rejected_key rejected \
   --flash_attn \
   --train_split train \
   --eval_split eval \
   --gradient_checkpointing \
   --loss sigmoid \
   --customized_reward_model_name OpenRLHF \
   --wandb_run_name llama3-8b-crm-final-v0.1 \
   --use_wandb True  
EOF

if [[ ${1} != "slurm" ]]; then
    deepspeed --module $training_commands
fi
