# Copyright 2023 AllenAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging
import os
import sys
sys.path.insert(0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import numpy as np
import torch
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from fastchat.conversation import get_conv_template
from tqdm import tqdm
from trl.trainer.utils import DPODataCollatorWithPadding

from rewardbench import (
    DPO_MODEL_CONFIG,
    DPOInference,
    load_eval_dataset,
    save_to_hub,
    torch_dtype_mapping,
)
from rewardbench.constants import EXAMPLE_COUNTS, SUBSET_MAPPING
from rewardbench.utils import calculate_scores_per_section
from utils import (
    set_random_seed, 
    print_rank_0,
)
import random
import json
import pandas as pd
from datasets import Dataset
import copy

# Enable TensorFloat32 (TF32) tensor cores on Ampere GPUs for matrix multiplications (faster than FP32)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# get token from HF_TOKEN env variable, but if it doesn't exist pass none
HF_TOKEN = os.getenv("HF_TOKEN", None)
# this is necessary to automatically log in when running this script in docker/batch beaker jobs
if HF_TOKEN is not None:
    from huggingface_hub._login import _login

    _login(token=HF_TOKEN, add_to_git_credential=False)


def get_args():
    """
    Parse arguments strings model and chat_template
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="path to model")
    parser.add_argument("--ref_model", type=str, default=None, help="path to model")
    parser.add_argument(
        "--ref_free_type", type=str, default="avg", help="type of reference free normalization (norm, avg, or sum)"
    )
    parser.add_argument("--tokenizer", type=str, default=None, help="path to non-matching tokenizer")
    parser.add_argument("--chat_template", type=str, default="tulu", help="path to chat template")
    parser.add_argument("--do_not_save", action="store_true", help="do not save results to hub (for debugging)")
    parser.add_argument("--batch_size", type=int, default=6, help="batch size for inference")
    parser.add_argument(
        "--pref_sets", action="store_true", help="run on common preference sets instead of our custom eval set"
    )
    parser.add_argument(
        "--trust_remote_code", action="store_true", default=False, help="directly load model instead of pipeline"
    )
    parser.add_argument("--debug", action="store_true", default=False, help="use only 10 examples")
    parser.add_argument(
        "--disable_beaker_save", action="store_true", help="disable saving the main results in a file for AI2 Beaker"
    )
    parser.add_argument(
        "--not_quantized", action="store_true", help="disable quantization for models that are quantized by default"
    )
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32", "float64"],
        help="PyTorch dtype (default: float16)",
    )

    
    parser.add_argument("--seed", type=int, default=42, help="random seed for reproducibility")
    parser.add_argument('--mode', type=str, default='chosen', choices=['chosen', 'rejected', 'normal', 'both_high', 'both_low'])
    parser.add_argument('--add_probabilities', action='store_true', help='add confidence probabilities to the text')
    parser.add_argument('--format_prompt', action='store_true', help='Format prompt for the model (add confidence scale)')
    parser.add_argument("--output_dir", type=str, required=True, default="output/")
    parser.add_argument('--pure_number', action='store_true', help='Just appended a pure random number at the end of the mode response')
    parser.add_argument('--probabilities_side',
                        type=str,
                        default='right',
                        choices=['left', 'right'],
                        help='which side to add the random confidence score')
    args = parser.parse_args()
    args.torch_dtype = torch_dtype_mapping(args.torch_dtype)
    return args

def main():
    args = get_args()
    accelerator = Accelerator()
    set_random_seed(args.seed)

    ###############
    # Setup logging
    ###############
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

    logger.info(f"Running reward model on {args.model} with chat template {args.chat_template} with mode {args.mode} and probabilities {args.add_probabilities}")
    if args.trust_remote_code:
        logger.info("Loading model with Trust Remote Code")

    if args.model in DPO_MODEL_CONFIG:
        config = DPO_MODEL_CONFIG[args.model]
    else:
        logger.warning(f"Model {args.model} not found in REWARD_MODEL_CONFIG, using default config")
        config = DPO_MODEL_CONFIG["default"]
    logger.info(f"Using dpo model config: {config}")

    model_builder = config["model_builder"]
    tokenizer_builder = config["tokenizer_builder"]

    # check datatype from argparse
    if args.torch_dtype == torch.bfloat16:
        logger.warning("Loading weights directly as bfloat16 for PyTorch dtype")
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float16


    assert args.model != args.ref_model, "policy and reference model should be different"
    # load chat template
    chat_template = args.chat_template
    conv = get_conv_template(chat_template)

    # define reference free
    if args.ref_model is None:
        ref_free = True
        logger.info("Running reference free DPO - no reference model provided")
    else:
        ref_free = False
        logger.info(f"Running DPO with reference model {args.ref_model}")

    ############################
    # Load dataset
    ############################
    logger.info("*** Load dataset ***")
    tokenizer_path = args.tokenizer if args.tokenizer else args.model
    tokenizer = tokenizer_builder(tokenizer_path, trust_remote_code=args.trust_remote_code)
    tokenizer.pad_token = tokenizer.eos_token
    # if no BOS token, set as pad token, e.g. QWEN models
    if tokenizer.bos_token is None:
        tokenizer.bos_token_id = tokenizer.eos_token_id
        tokenizer.pad_token_id = tokenizer.eos_token_id

    dataset, subsets = load_eval_dataset(
        core_set=not args.pref_sets,
        conv=conv,
        tokenizer=tokenizer,
        logger=logger,
        keep_columns=["text_chosen", "text_rejected", "id", "prompt"],
        mode = args.mode,
        add_probabilities = args.add_probabilities,
        format_prompt = args.format_prompt,
        pure_number = args.pure_number,
        probabilities_side = args.probabilities_side,
    )

    ids = dataset["id"]
    dataset = dataset.remove_columns("id")

    dataset = Dataset.from_pandas(pd.DataFrame(dataset))
    logger.info(f"Dataset size: {len(dataset)}")
    logger.info(dataset[0]['prompt'])
    logger.info(dataset[0]['text_chosen'])
    logger.info(dataset[0]['text_rejected'])

    # debug: use only 10 examples
    if args.debug:
        dataset = dataset.select(range(10))
        subsets = subsets[:10]

    ############################
    # Load reward model pipeline
    ############################
    BATCH_SIZE = args.batch_size

    if (
        ("llama-3" in args.model)
        or ("Llama3" in args.model)
        or ("Llama-3" in args.model)
        or ("LLaMA3" in args.model)
        or args.not_quantized
    ):
        model_kwargs = {
            "device_map": "auto",
            "torch_dtype": torch.float16 if torch.cuda.is_available() else None,
        }
        model_kwargs_ref = {
            "device_map": "auto",
            "torch_dtype": torch.float16 if torch.cuda.is_available() else None,
        }
    else:
        model_kwargs = {
            "load_in_8bit": True,
            "device_map": "auto",
            "torch_dtype": torch.float16 if torch.cuda.is_available() else None,
        }
        model_kwargs_ref = {
            "load_in_8bit": True,
            "device_map": "auto",
            "torch_dtype": torch.float16 if torch.cuda.is_available() else None,
        }

    model = model_builder(
        args.model,
        trust_remote_code=args.trust_remote_code,
        **model_kwargs,
    )

    if ref_free:
        ref_model = None
    else:
        model_kwargs_ref = {
            "load_in_8bit": True,
            "device_map": "auto",
            "torch_dtype": torch.float16 if torch.cuda.is_available() else None,
        }
        ref_model = model_builder(
            args.ref_model,
            trust_remote_code=args.trust_remote_code,
            **model_kwargs_ref,
        )

    model.eval()
    if ref_model:
        ref_model.eval()
    if model.training:
        logger.info("Model is in training mode, set to eval mode")
    else:
        logger.info("Model is already in eval mode")

    # use internal inference functions in DPO trainer
    dpo = DPOInference(
        model,
        ref_model,
        tokenizer=tokenizer,
        accelerator=accelerator,
        ref_free_norm=args.ref_free_type,
        # norm is norm, avg is average, sum is sum
    )
    # tokenize dataset
    column_names = list(dataset.features)

    tokenized_dataset = dataset.map(dpo.tokenize_row, remove_columns=column_names)

    dataloader = torch.utils.data.DataLoader(
        tokenized_dataset,
        batch_size=BATCH_SIZE,
        collate_fn=DPODataCollatorWithPadding(
            pad_token_id=tokenizer.pad_token_id,
            label_pad_token_id=dpo.label_pad_token_id,
            is_encoder_decoder=dpo.is_encoder_decoder,
        ),
        # collate_fn = lambda x: x, # fix weird batching error
        shuffle=False,
        drop_last=False,
    )
    results = []
    scores_chosen = []
    scores_rejected = []

    for step, batch in enumerate(tqdm(dataloader, desc="RM batch steps")):
        logger.info(f"RM inference step {step}/{len(dataloader)}")

        rewards_chosen, rewards_rejected = dpo.inference_step(batch, ref_free=ref_free)

        # for each item in batch, record 1 if chosen > rejected
        # extra score from dict within batched results (e.g. logits)
        # [{'label': 'LABEL_1', 'score': 0.6826171875},... ]
        if isinstance(rewards_chosen[0], dict):
            scores_chosen_batch = [result["score"] for result in rewards_chosen]
            scores_rejected_batch = [result["score"] for result in rewards_rejected]
        # for classes that directly output scores (custom code)
        else:
            scores_chosen_batch = rewards_chosen.cpu().numpy().tolist()
            scores_rejected_batch = rewards_rejected.cpu().numpy().tolist()

        [
            results.append(1) if chosen > rejected else results.append(0)
            for chosen, rejected in zip(scores_chosen_batch, scores_rejected_batch)
        ]
        scores_chosen += scores_chosen_batch
        scores_rejected += scores_rejected_batch

    ############################
    # save scores.json
    ############################
    flat_scores_chosen = scores_chosen
    flat_scores_rejected = scores_rejected

    if any(isinstance(i, list) for i in scores_chosen):
        flat_scores_chosen = [item for sublist in scores_chosen for item in sublist]
    if any(isinstance(i, list) for i in scores_rejected):
        flat_scores_rejected = [item for sublist in scores_rejected for item in sublist]

    scores = []
    for chosen_score, rejected_score in zip(flat_scores_chosen, flat_scores_rejected):
        scores.append({
            'high_confidence_score': rejected_score,
            'low_confidence_score': chosen_score,
        })
    # add average reward scores for chosen and rejected
    average_chosen = sum(flat_scores_chosen) / len(flat_scores_chosen)
    average_rejected = sum(flat_scores_rejected) / len(flat_scores_rejected)

    file_name = 'scores'
    if args.add_probabilities:
        file_name += '_with_probability'
    file_name += '_' + args.mode + '.json'
    average_file_name = 'average_' + file_name
    with open(os.path.join(args.output_dir, file_name), "w") as f:
        json.dump(scores, f, indent=4)
    
    with open(os.path.join(args.output_dir, average_file_name), "w") as f:
        json.dump({
            'average_high_confidence_score (rejected)': average_rejected,
            'average_low_confidence_score (chosen)': average_chosen,
        }, f, indent=4)
    logger.info(f"Saved scores to {os.path.join(args.output_dir, file_name)}")

    ############################
    # Print & process results
    ############################
    # add column for results for easy printing
    out_dataset = dataset.add_column("results", results)

    # add subsets back (removed so it's not handled by cuda)
    out_dataset = out_dataset.add_column("subset", subsets)
    # add scores_chosen and scores_rejected to the dataset
    out_dataset = out_dataset.add_column("scores_chosen", scores_chosen)
    out_dataset = out_dataset.add_column("scores_rejected", scores_rejected)

    results_grouped = {}
    results_grouped["model"] = args.model
    results_grouped["ref_model"] = args.ref_model
    results_grouped["model_type"] = "DPO"  # TODO add options for references free, DPO-ref-free, or DPO-normalized
    if ref_free:
        results_grouped["model_type"] = "DPO Ref. Free"
        save_modifier = "_ref_free"
    else:
        save_modifier = ""
    results_grouped["chat_template"] = args.chat_template if not hasattr(tokenizer, "chat_template") else "tokenizer"
    # print per subset and log into results_grouped file
    present_subsets = np.unique(subsets)
    for subset in present_subsets:
        subset_dataset = out_dataset.filter(lambda example: example["subset"] == subset)
        num_correct = sum(subset_dataset["results"])
        num_total = len(subset_dataset["results"])
        print(f"{subset}: {num_correct}/{num_total} ({num_correct/num_total})")
        results_grouped[subset] = num_correct / num_total

    # log leaderboard aggregated results
    if not args.pref_sets:
        results_leaderboard = calculate_scores_per_section(EXAMPLE_COUNTS, SUBSET_MAPPING, results_grouped)
        print(results_leaderboard)

    # ############################
    # # Upload results to hub
    # ############################
    # sub_path = "eval-set/" if not args.pref_sets else "pref-sets/"
    # results_url = save_to_hub(
    #     results_grouped,
    #     args.model + save_modifier,
    #     sub_path,
    #     args.debug,
    #     local_only=args.do_not_save,
    #     save_metrics_for_beaker=not args.disable_beaker_save,
    # )
    # if not args.do_not_save:
    #     logger.info(f"Uploaded reward model results to {results_url}")

    # # upload chosen-rejected with scores
    # # create new json with scores and upload
    # scores_dict = out_dataset.to_dict()
    # scores_dict["model"] = args.model
    # scores_dict["model_type"] = "DPO"
    # scores_dict["chat_template"] = args.chat_template
    # sub_path_scores = "eval-set-scores/" if not args.pref_sets else "pref-sets-scores/"

    # scores_url = save_to_hub(
    #     scores_dict, args.model + save_modifier, sub_path_scores, args.debug, local_only=args.do_not_save
    # )
    # logger.info(f"Uploading chosen-rejected text with scores to {scores_url}")


if __name__ == "__main__":
    main()