import argparse
import itertools
import math
import os
from copy import deepcopy
from datetime import datetime

import torch
from transformers.trainer import get_scheduler

from openrlhf.datasets import PromptDataset, SFTDataset
from openrlhf.models import Actor, get_llm_for_sequence_regression, DefaultRewardModel, DefaultCriticModel
from openrlhf.trainer import PPOTrainer
from openrlhf.utils import blending_datasets, get_strategy, get_tokenizer, sequential_data_loading, MyStoppingCriteria
# for unsupervised data
from transformers import default_data_collator


def train(args):
    # configure strategy
    strategy = get_strategy(args)
    strategy.setup_distributed()

    rank = torch.distributed.get_rank()
    num_processes = torch.distributed.get_world_size()

    # configure model
    # load huggingface model
    actor = Actor(
        args.pretrain,
        use_flash_attention_2=args.flash_attn,
        bf16=args.bf16,
        load_in_4bit=args.load_in_4bit,
        lora_rank=args.actor_lora_rank,
        lora_alpha=args.actor_lora_alpha,
        target_modules=args.actor_target_modules,
        lora_dropout=args.actor_lora_dropout,
        ds_config=strategy.get_ds_train_config(is_actor=True),
    )

    if args.actor_init_on_gpu:
        actor = actor.to(torch.cuda.current_device())
        
    critic = get_llm_for_sequence_regression(
        args.critic_pretrain,
        "critic",
        normalize_reward=args.normalize_reward,
        use_flash_attention_2=args.flash_attn,
        bf16=args.bf16,
        load_in_4bit=args.load_in_4bit,
        lora_rank=args.critic_lora_rank,
        lora_alpha=args.critic_lora_alpha,
        target_modules=args.critic_target_modules,
        lora_dropout=args.critic_lora_dropout,
        ds_config=strategy.get_ds_train_config(is_actor=False),
        value_head_prefix=args.value_head_prefix,
    )
    # # noqa
    # if args.customized_reward_model_name in ['Armo']:
    #     strategy.print('Loading Default pretrained reward models!!!')
    #     reward_model = DefaultRewardModel(
    #         args.reward_pretrain,
    #         use_flash_attention_2=args.flash_attn,
    #         bf16=args.bf16,
    #         load_in_4bit=args.load_in_4bit,
    #         value_head_prefix=args.value_head_prefix,
    #         ds_config=strategy.get_ds_train_config(is_actor=False),
    #         model_family=args.customized_reward_model_name,
    #     )
    # else:
    
    reward_model = get_llm_for_sequence_regression(
        args.reward_pretrain,
        "reward",
        normalize_reward=args.normalize_reward,
        use_flash_attention_2=args.flash_attn,
        bf16=args.bf16,
        load_in_4bit=args.load_in_4bit,
        ds_config=strategy.get_ds_train_config(is_actor=False),
        value_head_prefix=args.value_head_prefix,
    )

    # configure tokenizer
    tokenizer = get_tokenizer(args.pretrain, actor.model, "left", strategy, use_fast=not args.disable_fast_tokenizer)
    get_tokenizer(args.critic_pretrain, critic, "left", strategy, use_fast=not args.disable_fast_tokenizer)
    # noqa, since sometimes the RM use a different tokenizer (we add this for mistral only)
    reward_tokenizer = get_tokenizer(args.reward_pretrain, reward_model, "left", strategy, use_fast=not args.disable_fast_tokenizer)
    # resize_embedding = False
    # if reward_tokenizer.pad_token is not None and reward_tokenizer.pad_token != reward_tokenizer.eos_token and tokenizer.pad_token != reward_tokenizer.pad_token:
    #     strategy.print("Set tokenizer.pad_token to reward_tokenizer.pad_token")
    #     tokenizer.add_special_tokens({"pad_token": reward_tokenizer.pad_token})
    #     actor.model.resize_token_embeddings(len(tokenizer))
    #     resize_embedding = True

    strategy.print(actor)
    strategy.print(critic)

    # load weights for reference actor
    initial_model = Actor(
        args.pretrain,
        use_flash_attention_2=args.flash_attn,
        bf16=args.bf16,
        load_in_4bit=args.load_in_4bit,
        ds_config=strategy.get_ds_eval_config(offload=False),
    )
    # if resize_embedding:
    #     initial_model.model.resize_token_embeddings(len(reward_tokenizer))

    get_tokenizer(args.pretrain, initial_model.model, "left", strategy)

    strategy.print("reward normalization status: {}".format(args.normalize_reward))
    strategy.print("mean: {}, std {}".format(reward_model.mean, reward_model.std))

    if args.enable_ema:
        ema_model = Actor(
            args.pretrain,
            use_flash_attention_2=args.flash_attn,
            bf16=args.bf16,
            load_in_4bit=args.load_in_4bit,
            ds_config=strategy.get_ds_eval_config(offload=True),
        )
    else:
        ema_model = None

    # configure optimizer
    actor_optim = strategy.create_optimizer(
        actor, lr=args.actor_learning_rate, betas=args.adam_betas, weight_decay=args.actor_l2
    )
    critic_optim = strategy.create_optimizer(
        critic, lr=args.critic_learning_rate, betas=args.adam_betas, weight_decay=args.critic_l2
    )

    # prepare datasets
    # prompts_data = blending_datasets(
    #     args.prompt_data,
    #     args.prompt_data_probs,
    #     strategy,
    #     args.seed,
    #     max_count=args.max_samples,
    #     return_eval=False,
    #     train_split=args.prompt_split,
    # )
    prompts_data = sequential_data_loading(rank, num_processes, args, strategy, return_eval=False)
    prompts_data = prompts_data.select(range(min(args.max_samples, len(prompts_data))))
    prompts_dataset = PromptDataset(prompts_data, tokenizer, strategy, input_template=args.input_template)
    strategy.print("Example prompt: ", prompts_dataset.prompts[0])
    prompts_dataloader = strategy.setup_dataloader(prompts_dataset, args.micro_rollout_batch_size, True, True)

    if args.pretrain_data:
        # pretrain_data = blending_datasets(
        #     args.pretrain_data,
        #     args.pretrain_data_probs,
        #     strategy,
        #     args.seed,
        #     return_eval=False,
        #     train_split=args.pretrain_split,
        # )

        pretrain_data = sequential_data_loading(rank, num_processes, args, strategy, return_eval=False, pretrain_data = True)
        pretrain_max_len = args.max_len if args.max_len else args.prompt_max_len + args.generate_max_len
        pretrain_dataset = SFTDataset(
            pretrain_data.select(range(min(len(pretrain_data), args.max_epochs * len(prompts_dataset)))),
            tokenizer,
            pretrain_max_len,
            strategy,
            pretrain_mode=True,
        )
        pretrain_dataloader = itertools.cycle(
            iter(
                strategy.setup_dataloader(
                    pretrain_dataset,
                    args.micro_train_batch_size,
                    True,
                    True,
                    pretrain_dataset.collate_fn,
                )
            )
        )
    else:
        pretrain_dataloader = None

    # configure scheduler
    num_update_steps_per_episodes = (
        int(len(prompts_dataloader) * (args.micro_rollout_batch_size / args.micro_train_batch_size))
        * args.max_epochs
        // strategy.accumulated_gradient
    )

    max_steps = math.ceil(args.num_episodes * num_update_steps_per_episodes)

    actor_scheduler = get_scheduler(
        "cosine_with_min_lr",
        actor_optim,
        num_warmup_steps=math.ceil(max_steps * 0.03),
        num_training_steps=max_steps,
        scheduler_specific_kwargs={"min_lr": args.actor_learning_rate * 0.1},
    )

    critic_scheduler = get_scheduler(
        "cosine_with_min_lr",
        critic_optim,
        num_warmup_steps=math.ceil(max_steps * 0.03),
        num_training_steps=max_steps,
        scheduler_specific_kwargs={"min_lr": args.critic_learning_rate * 0.1},
    )

    # gradient_checkpointing
    if args.gradient_checkpointing:
        actor.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": args.gradient_checkpointing_use_reentrant}
        )
        critic.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": args.gradient_checkpointing_use_reentrant}
        )

    # prepare models/optimizers...
    (
        (actor, actor_optim, actor_scheduler),
        (critic, critic_optim, critic_scheduler),
        reward_model,
        initial_model,
    ) = strategy.prepare(
        (actor, actor_optim, actor_scheduler),
        (critic, critic_optim, critic_scheduler),
        reward_model,
        initial_model,
        is_rlhf=True,
    )

    if ema_model:
        ema_model._offload = True
        ema_model = strategy.prepare(ema_model, is_rlhf=True)

    # load checkpoint
    if args.load_checkpoint:
        strategy.print("Load checkpoint: ", args.save_path)

    os.makedirs(args.save_path, exist_ok=True)

    if args.add_confidence_stopping_criteria:
        stop_criteria = [MyStoppingCriteria(tokenizer)]
        strategy.print("Using Confidence Stopping Criteria")
    else:
        strategy.print("Not Using Any Stopping Criteria")
        stop_criteria = None

    # configure Trainer
    trainer = PPOTrainer(
        strategy,
        actor,
        critic,
        reward_model,
        initial_model,
        ema_model,
        actor_optim,
        critic_optim,
        actor_scheduler,
        critic_scheduler,
        max_epochs=args.max_epochs,
        micro_train_batch_size=args.micro_train_batch_size,
        micro_rollout_batch_size=args.micro_rollout_batch_size,
        gradient_checkpointing=args.gradient_checkpointing,
        tokenizer=tokenizer,
        prompt_max_len=args.prompt_max_len,
        value_clip=args.value_clip,
        eps_clip=args.eps_clip,
        gamma=args.gamma,
        lambd=args.lambd,
        init_kl_coef=args.init_kl_coef,
        kl_target=args.kl_target,
        ema_beta=0.992,
        ptx_coef=args.ptx_coef,
        max_norm=args.max_norm,
        # fro GPT generation
        do_sample=True,
        max_new_tokens=args.generate_max_len,
        max_length=args.max_len,
        temperature=args.temperature,
        top_p=args.top_p,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        # stop criteria for controlled calibrated ppo
        stopping_criteria=stop_criteria,
    )

    trainer.fit(
        prompts_dataloader,
        pretrain_dataloader,
        args,
    )

    # save model checkpoint after fitting on only rank0
    strategy.save_model(
        ema_model if args.enable_ema else actor,
        tokenizer,
        args.save_path,
    )

    if args.save_value_network:
        strategy.save_model(
            critic,
            tokenizer,
            args.save_path + "_critic",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Checkpoint
    parser.add_argument("--save_path", type=str, default="./ckpt")
    parser.add_argument("--save_steps", type=int, default=-1)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--eval_steps", type=int, default=-1)
    parser.add_argument("--ckpt_path", type=str, default="./ckpt/checkpoints_ppo")
    parser.add_argument("--max_ckpt_num", type=int, default=3)
    parser.add_argument("--max_ckpt_mem", type=int, default=1000)  # 1000GB
    parser.add_argument("--load_checkpoint", action="store_true", default=False)

    # PPO
    parser.add_argument("--num_episodes", type=int, default=1)
    parser.add_argument("--rollout_batch_size", type=int, default=512)
    parser.add_argument("--micro_rollout_batch_size", type=int, default=8)
    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument("--prompt_max_len", type=int, default=1024, help="Max tokens for each prompt")
    parser.add_argument("--generate_max_len", type=int, default=1024, help="Max tokens to generate in PPO")
    parser.add_argument("--max_len", type=int, default=None, help="deprecated max_len")
    parser.add_argument("--max_samples", type=int, default=1000000)
    parser.add_argument("--max_norm", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--actor_l2", type=float, default=0.0, help="weight decay loss")
    parser.add_argument("--critic_l2", type=float, default=0.0, help="weight decay loss")
    parser.add_argument("--ptx_coef", type=float, default=0.05, help="PPO-ptx loss coef")
    parser.add_argument("--eps_clip", type=float, default=0.2, help="PPO clip range")
    parser.add_argument("--value_clip", type=float, default=0.2, help="PPO value clip range")
    parser.add_argument("--lambd", type=float, default=0.95, help="PPO GAE lambd")
    parser.add_argument("--gamma", type=float, default=1, help="PPO GAE gamma")
    parser.add_argument("--micro_train_batch_size", type=int, default=4, help="batch size per GPU")
    parser.add_argument("--train_batch_size", type=int, default=128, help="Global training batch size")
    parser.add_argument("--normalize_reward", action="store_true", default=False, help="Enable Reward Normazation")
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--freezing_actor_steps", type=int, default=-1, help="Used for critic initialization")
    parser.add_argument(
        "--n_samples_per_prompt", type=int, default=1, help="number of responses for each prompt in generation"
    )
    parser.add_argument("--save_value_network", action="store_true", default=False, help="Save critic model")
    parser.add_argument("--actor_learning_rate", type=float, default=1e-6)
    parser.add_argument("--critic_learning_rate", type=float, default=9e-6)
    parser.add_argument("--kl_target", type=float, default=None)
    parser.add_argument("--init_kl_coef", type=float, default=0.01, help="KL penalty in PPO")
    parser.add_argument("--adam_betas", type=float, nargs=2, default=(0.9, 0.95), help="Betas for Adam optimizer")

    # DeepSpeed
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for deepspeed")
    parser.add_argument("--zero_stage", type=int, default=2, help="DeepSpeed ZeRO stage")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--bf16", action="store_true", default=False, help="Enable bfloat16")
    parser.add_argument("--enable_ema", action="store_true", help="Enable EMA checkpoint for the model.")
    parser.add_argument("--zpg", type=int, default=1, help="ZeRO++ max partition size")
    parser.add_argument("--adam_offload", action="store_true", default=False, help="Offload Adam Optimizer")
    parser.add_argument("--actor_init_on_gpu", action="store_true", default=False)
    parser.add_argument("--flash_attn", action="store_true", default=False, help="Enable FlashAttention2")
    parser.add_argument("--aux_loss_coef", type=float, default=0, help="MoE balancing loss")
    parser.add_argument("--grad_accum_dtype", type=str, default=None, help="Adam grad accum data type")
    parser.add_argument("--disable_trace_cache", action="store_true", default=False)
    parser.add_argument("--gradient_checkpointing_use_reentrant", action="store_true", default=False)
    parser.add_argument("--disable_fast_tokenizer", action="store_true", default=False)

    # other hp
    parser.add_argument("--customized_reward_model_name", type=str, default=None, help="Other reward models")

    # LoRA
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    # lora for actor
    parser.add_argument("--actor_lora_rank", type=int, default=0)
    parser.add_argument("--actor_lora_alpha", type=int, default=16)
    parser.add_argument("--actor_target_modules", type=str, nargs="*", default="all-linear")
    parser.add_argument("--actor_lora_dropout", type=float, default=0)
    # lora for critic
    parser.add_argument("--critic_lora_rank", type=int, default=0)
    parser.add_argument("--critic_lora_alpha", type=int, default=16)
    parser.add_argument("--critic_target_modules", type=str, nargs="*", default="all-linear")
    parser.add_argument("--critic_lora_dropout", type=float, default=0)

    # Models
    parser.add_argument("--pretrain", type=str, default=None, help="HF model name or path")
    parser.add_argument("--reward_pretrain", type=str, default=None, help="HF model name or path")
    parser.add_argument("--critic_pretrain", type=str, default=None, help="HF model name or path")
    parser.add_argument("--value_head_prefix", type=str, default="value_head")

    # Custom dataset
    parser.add_argument("--prompt_data", type=str, default=None, help="HF dataset name or path")
    parser.add_argument(
        "--prompt_data_probs",
        type=str,
        default="1.0",
        help="sampling probs for datasets",
    )
    parser.add_argument("--prompt_split", type=str, default="train")
    parser.add_argument("--pretrain_data", type=str, default=None, help="HF dataset name or path")
    parser.add_argument(
        "--pretrain_data_probs",
        type=str,
        default="1.0",
        help="sampling probs for datasets",
    )
    parser.add_argument("--pretrain_split", type=str, default="train")
    parser.add_argument("--input_key", type=str, default="input", help="JSON dataset key")
    parser.add_argument("--input_template", type=str, default="User: {}\nAssistant: ")
    parser.add_argument(
        "--apply_chat_template", action="store_true", default=False, help="Use HF tokenizer chat template"
    )

    # wandb parameters
    parser.add_argument("--use_wandb", type=str, default=None)
    parser.add_argument("--wandb_org", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="openrlhf_train_ppo")
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default="ppo_%s" % datetime.now().strftime("%m%dT%H:%M"),
    )

    # calibration parameter
    parser.add_argument(
        "--add_confidence_stopping_criteria", action="store_true", default=False, help="Ask for confidence in the prompt"
    )

    args = parser.parse_args()

    if args.critic_pretrain is None:
        args.critic_pretrain = args.reward_pretrain
    train(args)