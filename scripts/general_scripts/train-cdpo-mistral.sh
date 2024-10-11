#!/bin/bash
cd OpenRLHF
export WANDB__SERVICE_WAIT=300

set -x

read -r -d '' training_commands <<EOF
openrlhf.cli.train_cdpo \
   --save_path /dpo_checkpoints/mistral-7b-hermes-cdpo-v0.2 \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps -1 \
   --train_batch_size 128 \
   --micro_train_batch_size 1 \
   --pretrain NousResearch/Nous-Hermes-2-Mistral-7B-DPO \
   --ref_pretrain teknium/OpenHermes-2.5-Mistral-7B \
   --bf16 \
   --max_epochs 1 \
   --max_len 4096 \
   --zero_stage 3 \
   --learning_rate 3e-7 \
   --beta 0.01 \
   --dataset HINT-lab/calibration_preference_mixture_final-v0.1 \
   --apply_chat_template \
   --chosen_key chosen \
   --rejected_key rejected \
   --max_samples 1000000 \
   --flash_attn \
   --gradient_checkpointing \
   --ref_offload \
   --wandb_run_name mistral-7b-hermes-cdpo-v0.2 \
   --use_wandb True
EOF

if [[ ${1} != "slurm" ]]; then
    deepspeed --module $training_commands
fi

