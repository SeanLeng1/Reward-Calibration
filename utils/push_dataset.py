from transformers import AutoTokenizer, pipeline, AutoConfig, AutoModel, AutoModelForCausalLM
import torch
from typing import Optional, List
import torch.nn as nn
from accelerate.logging import get_logger
import logging
import sys
import transformers
from datasets import load_dataset


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

    name = "Calibration/prompt-collections-final-v0.3"
    dataset = load_dataset(name, split = 'train')


    dataset.push_to_hub("HINT-lab/prompt-collections-final-v0.3")