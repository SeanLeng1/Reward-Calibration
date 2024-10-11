from transformers import AutoTokenizer, pipeline, AutoConfig, AutoModel, AutoModelForCausalLM
import torch
from typing import Optional, List
import torch.nn as nn
from accelerate.logging import get_logger
import logging
import sys
import transformers


# https://github.com/OpenLLMAI/OpenRLHF/blob/main/openrlhf/models/model.py#L178
def _get_reward_model(base_pretrained_model, base_llm_model, value_head_prefix="value_head", packing_samples=False):
    class RewardModel(base_pretrained_model):
        supports_gradient_checkpointing = True

        def __init__(self, config: AutoConfig):
            super().__init__(config)
            setattr(self, self.base_model_prefix, base_llm_model(config))

            self.value_head_prefix = value_head_prefix
            setattr(self, value_head_prefix, nn.Linear(config.hidden_size, 1, bias=False))

            self.packing_samples = packing_samples

            # mean std
            self.normalize_reward = config.normalize_reward
            self.register_buffer("mean", torch.zeros(1), persistent=False)
            self.register_buffer("std", torch.ones(1), persistent=False)

            # load mean/std from config.json
            if hasattr(config, "mean"):
                self.mean[0] = config.mean
                self.std[0] = config.std

        def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            return_output=False,
        ) -> torch.Tensor:
            if not self.packing_samples:
                # https://github.com/OpenRLHF/OpenRLHF/issues/217
                position_ids = attention_mask.long().cumsum(-1) - 1
            else:
                raise ValueError('Does not Support Packing Samples for Reward Evaluation, please set it to false')
            position_ids.masked_fill_(attention_mask == 0, 1)

            outputs = getattr(self, self.base_model_prefix)(
                input_ids, attention_mask=attention_mask, position_ids=position_ids
            )
            last_hidden_states = outputs["last_hidden_state"]
            values = getattr(self, self.value_head_prefix)(last_hidden_states).squeeze(-1)

            if self.packing_samples:
                reward = values
            else:
                eos_indices = attention_mask.size(1) - 1 - attention_mask.long().fliplr().argmax(dim=1, keepdim=True)
                reward = values.gather(dim=1, index=eos_indices).squeeze(1)

            if not self.training and self.normalize_reward:
                reward = (reward - self.mean) / self.std

            return (reward, outputs) if return_output else reward

    return RewardModel


if __name__ == '__main__':
    logger = get_logger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = logging.INFO
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    name = "/storage1/jiaxinh/Active/jixuan/rm_checkpoints_fixed/mistral-7b-hermes-rm-skywork"
    config = AutoConfig.from_pretrained(name, trust_remote_code = True)
    base_class = AutoModel._model_mapping[type(config)]
    base_pretrained_class = base_class.__base__
    head_prefix = 'value_head'
    cls_class = _get_reward_model(base_pretrained_class, base_class, head_prefix)
    # our trained and OpenRLHF model use bf16 by default
    model = cls_class.from_pretrained(name, config=config, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map = "cpu")
    tokenizer = AutoTokenizer.from_pretrained(name)


    # name = "/storage1/jiaxinh/Active/jixuan/dpo_checkpoints2/mistral-7b-hermes-dpo-v0.2"
    # model = AutoModelForCausalLM.from_pretrained(
    #     name,
    #     trust_remote_code=True,
    #     attn_implementation='flash_attention_2',
    #     quantization_config=None,
    #     torch_dtype=torch.bfloat16,
    #     device_map=None,
    # )
    # tokenizer = AutoTokenizer.from_pretrained(name)


    model.push_to_hub("HINT-lab/mistral-7b-hermes-rm-skywork")
    tokenizer.push_to_hub("HINT-lab/mistral-7b-hermes-rm-skywork")