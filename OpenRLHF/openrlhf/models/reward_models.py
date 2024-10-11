import random
from typing import Optional, List
from transformers import (
    AutoConfig, 
    AutoModel, 
    BitsAndBytesConfig, 
    AutoModelForSequenceClassification,
    LlamaConfig, 
    LlamaModel, 
    PreTrainedModel,
)
from transformers.integrations import HfDeepSpeedConfig
import torch
from .utils import reset_position_ids
from .packing_utils import patch_for_block_diag_attn
from peft import LoraConfig, TaskType, get_peft_model
from peft.tuners.lora import LoraLayer
import torch.nn as nn


# This is not tested, might not work
# since different reward models have their own way to extract the scores
# might need a model_config.py just like rewardbench 
class DefaultRewardModel(nn.Module):
    supports_gradient_checkpointing = True
    def __init__(
        self,
        pretrain_or_model,
        use_flash_attention_2=False,
        bf16=True,
        load_in_4bit=False,
        lora_rank=0,
        lora_alpha=16,
        lora_dropout=0,
        target_modules=None,
        ds_config=None,
        device_map=None,
        packing_samples=False,
        model_family="Armo",
        normalize_reward=False,
        **kwargs,
    ) -> None:
        super().__init__()
        if isinstance(pretrain_or_model, str):
            attn_implementation = "flash_attention_2" if use_flash_attention_2 else "eager"

            # Note: dschf is defined in function scope to avoid global effects
            # https://huggingface.co/docs/transformers/deepspeed#non-trainer-deepspeed-integration
            if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
                dschf = HfDeepSpeedConfig(ds_config)
            else:
                dschf = None

            if load_in_4bit:
                assert bf16, "we only support bnb_4bit_compute_dtype = bf16"
                nf4_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
            else:
                nf4_config = None

            self.model = AutoModelForSequenceClassification.from_pretrained(
                pretrain_or_model,
                trust_remote_code=True,
                attn_implementation=attn_implementation,
                quantization_config=nf4_config,
                torch_dtype=torch.bfloat16 if bf16 else "auto",
                device_map=device_map,
            )

            # LoRA
            if lora_rank > 0:
                model.enable_input_require_grads()
                lora_config = LoraConfig(
                    r=lora_rank,
                    lora_alpha=lora_alpha,
                    target_modules=target_modules,
                    lora_dropout=lora_dropout,
                    bias="none",
                )
                model = get_peft_model(model, lora_config)

                if load_in_4bit:
                    for name, module in model.named_modules():
                        if isinstance(module, LoraLayer):
                            module = module.to(torch.bfloat16)
                        if "norm" in name:
                            module = module.to(torch.float32)

            # MoE - balancing loss
            model_config = self.model.config.to_dict()
            if "output_router_logits" in model_config:
                print("[MoE] set output_router_logits as True")
                self.model.config.output_router_logits = True

            # https://github.com/huggingface/transformers/issues/26877
            # Use `model.generate(use_cache=True)` instead.`
            self.model.config.use_cache = False

            # packing samples using Flash Attention 2
            self.packing_samples = packing_samples
            if packing_samples:
                assert use_flash_attention_2, "Only support `--packing_samples` with Flash Attention 2."
                model_type = getattr(self.model.config, "model_type", None)
                patch_for_block_diag_attn(model_type)
        else:
            self.model = pretrain_or_model

        self.model_family = model_family
        self.config = self.model.config
        self.config.normalize_reward = normalize_reward
        self.register_buffer("mean", torch.zeros(1), persistent=False)
        self.register_buffer("std", torch.ones(1), persistent=False)

        # load mean/std from config.json
        if hasattr(self.config, "mean"):
            self.mean[0] = self.config.mean
            self.std[0] = self.config.std

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_output=False,
    ) -> torch.Tensor:
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        output = self.model(input_ids, attention_mask=attention_mask, position_ids=position_ids)
        if self.model_family == 'Armo':
            reward = output.score
        else:
            reward = output[0]
            if len(reward.shape) == 2:
                reward = reward.squeeze(-1)
        
        if not self.training and self.config.normalize_reward:
            reward = (reward - self.mean) / self.std

        return (reward, output) if return_output else reward
    
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs={"use_reentrant": False}):
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        self.model.gradient_checkpointing_disable()




class DefaultCriticModel(nn.Module):
    supports_gradient_checkpointing = True
    def __init__(
        self,
        pretrain_or_model,
        use_flash_attention_2=False,
        bf16=True,
        load_in_4bit=False,
        lora_rank=0,
        lora_alpha=16,
        lora_dropout=0,
        target_modules=None,
        ds_config=None,
        device_map=None,
        packing_samples=False,
        model_family="Armo",
        normalize_reward=False,
        **kwargs,
    ) -> None:
        super().__init__()
        if isinstance(pretrain_or_model, str):
            attn_implementation = "flash_attention_2" if use_flash_attention_2 else "eager"

            # Note: dschf is defined in function scope to avoid global effects
            # https://huggingface.co/docs/transformers/deepspeed#non-trainer-deepspeed-integration
            if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
                dschf = HfDeepSpeedConfig(ds_config)
            else:
                dschf = None

            if load_in_4bit:
                assert bf16, "we only support bnb_4bit_compute_dtype = bf16"
                nf4_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
            else:
                nf4_config = None

            self.model = AutoModelForSequenceClassification.from_pretrained(
                pretrain_or_model,
                trust_remote_code=True,
                attn_implementation=attn_implementation,
                quantization_config=nf4_config,
                torch_dtype=torch.bfloat16 if bf16 else "auto",
                device_map=device_map,
            )

            # LoRA
            if lora_rank > 0:
                model.enable_input_require_grads()
                lora_config = LoraConfig(
                    r=lora_rank,
                    lora_alpha=lora_alpha,
                    target_modules=target_modules,
                    lora_dropout=lora_dropout,
                    bias="none",
                )
                model = get_peft_model(model, lora_config)

                if load_in_4bit:
                    for name, module in model.named_modules():
                        if isinstance(module, LoraLayer):
                            module = module.to(torch.bfloat16)
                        if "norm" in name:
                            module = module.to(torch.float32)

            # MoE - balancing loss
            model_config = self.model.config.to_dict()
            if "output_router_logits" in model_config:
                print("[MoE] set output_router_logits as True")
                self.model.config.output_router_logits = True

            # https://github.com/huggingface/transformers/issues/26877
            # Use `model.generate(use_cache=True)` instead.`
            self.model.config.use_cache = False

            # packing samples using Flash Attention 2
            self.packing_samples = packing_samples
            if packing_samples:
                assert use_flash_attention_2, "Only support `--packing_samples` with Flash Attention 2."
                model_type = getattr(self.model.config, "model_type", None)
                patch_for_block_diag_attn(model_type)
        else:
            self.model = pretrain_or_model

        self.model_family = model_family
        self.config = self.model.config
        self.config.normalize_reward = normalize_reward
        self.register_buffer("mean", torch.zeros(1), persistent=False)
        self.register_buffer("std", torch.ones(1), persistent=False)

        # load mean/std from config.json
        if hasattr(self.config, "mean"):
            self.mean[0] = self.config.mean
            self.std[0] = self.config.std

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        action_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_output=False,
    ) -> torch.Tensor:
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        outputs = self.model(input_ids, attention_mask=attention_mask, position_ids=position_ids)
        if self.model_family == 'Armo':
            value = outputs.score
        elif self.model_family == 'Mistral':
            value = outputs.logits
        else:
            value = outputs[0].squeeze(-1)      # make it shape N
        num_actions = action_mask.size(1)

        if self.normalize_reward:
            values = (values - self.mean) / self.std

        if return_output:
            return outputs if num_actions is None else (values[:, -num_actions:], outputs)
        else:
            return values[:, -num_actions:]
    
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs={"use_reentrant": False}):
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        self.model.gradient_checkpointing_disable()
        