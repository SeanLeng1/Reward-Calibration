#!/bin/bash
cd OpenRLHF
export WANDB__SERVICE_WAIT=300

set -x 

read -r -d '' training_commands <<EOF
openrlhf.cli.train_cppo \
   --pretrain OpenLLMAI/Llama-3-8b-sft-mixture \
   --reward_pretrain OpenLLMAI/Llama-3-8b-rm-mixture \
   --save_path ppo_checkpoints/llama3-8b-final-ppo-c \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps -1 \
   --micro_train_batch_size 2 \
   --train_batch_size 64 \
   --micro_rollout_batch_size 4 \
   --rollout_batch_size 512 \
   --max_epochs 1 \
   --prompt_max_len 1024 \
   --generate_max_len 1024 \
   --zero_stage 2 \
   --bf16 \
   --actor_learning_rate 5e-7 \
   --critic_learning_rate 9e-6 \
   --actor_l2 0.0 \
   --critic_l2 0.0 \
   --init_kl_coef 0.01 \
   --prompt_data HINT-lab/prompt-collections-final-v0.3 \
   --prompt_data_probs 1.0 \
   --input_key confidence_prompt \
   --apply_chat_template \
   --max_samples 100000000 \
   --normalize_reward \
   --adam_offload \
   --flash_attn \
   --gradient_checkpointing \
   --wandb_run_name llama3-8b-final-ppo-c \
   --use_wandb True 
EOF

if [[ ${1} != "slurm" ]]; then
    deepspeed --module $training_commands
fi
