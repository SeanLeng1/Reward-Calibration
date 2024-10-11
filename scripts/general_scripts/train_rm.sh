#!/bin/bash
cd OpenRLHF
export WANDB__SERVICE_WAIT=300

set -x 

read -r -d '' training_commands <<EOF
openrlhf.cli.train_rm \
   --save_path rm_checkpoints/mistral-7b-hermes-rm-skywork \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps -1 \
   --train_batch_size 512 \
   --micro_train_batch_size 1 \
   --pretrain teknium/OpenHermes-2.5-Mistral-7B \
   --bf16 \
   --max_epochs 2 \
   --max_len 8192 \
   --zero_stage 2 \
   --learning_rate 2e-6 \
   --dataset Skywork/Skywork-Reward-Preference-80K-v0.1 \
   --dataset_probs 1.0 \
   --max_samples 1000000000000 \
   --apply_chat_template \
   --chosen_key chosen \
   --rejected_key rejected \
   --flash_attn \
   --l2 0.01 \
   --gradient_checkpointing \
   --wandb_run_name mistral-7b-hermes-rm-skywork \
   --use_wandb True
EOF

if [[ ${1} != "slurm" ]]; then
    deepspeed --module $training_commands
fi
