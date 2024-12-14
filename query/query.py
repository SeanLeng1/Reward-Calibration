import os
import sys
sys.path.insert(0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
import random
import json
from utils import (
    load_dataset, 
    set_random_seed, 
    print_rank_0,
    generate_completions,
    get_next_word_predictions,
    check_tokenizer_chat_template,
)
import logging
from argparse import ArgumentParser
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import gather_object
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import transformers
import torch
import re
from itertools import zip_longest
import warnings

# Enable TensorFloat32 (TF32) tensor cores on Ampere GPUs for matrix multiplications (faster than FP32)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, default="GSM8K")
    parser.add_argument("--data_path", type=str, required=True, default="data/GSM8K.json")
    parser.add_argument("--output_dir", type=str, required=True, default="output/")
    parser.add_argument("--seed", type=int, required=False, default=0)
    parser.add_argument("--model_name_or_path", type=str, required=True, default="llama")
    parser.add_argument("--task_type", type=str, required=True, choices=["multi_choice_qa", "open_number_qa"], default="multi_choice_qa")
    parser.add_argument('--dtype',
                        type=str,
                        default='fp16',
                        choices=['fp16', 'bf16', 'fp32', 'auto'],
                        help='Inference data type')
    parser.add_argument('--per_device_eval_batch_size', type=int, default=1, help='Batch size per device for evaluation')
    parser.add_argument('--max_new_tokens', type=int, default=256, help='Max number of tokens to generate')
    parser.add_argument('--temperature', type=float, default=1.0, help='generation tempreature')
    parser.add_argument('--top_p', type=float, default=1.0, help='generation top_p')
    parser.add_argument('--top_k', type=int, default=50, help='generation top_k')
    parser.add_argument('--next_word_predictions', action='store_true', help='Generate next word predictions')
    parser.add_argument('--use_cot', action='store_true', help='Use COT format for answers')
    parser.add_argument('--use_top_k', action='store_true', help='Use Top_K format for answers')
    parser.add_argument('--use_chat_template', action='store_true', help='Use chat template for prompts (for RLHF chat models)')
    parser.add_argument('--use_flash_attention_2', action='store_true', help='use flash attention 2')

    parser.add_argument('--conv',
                        type=str,
                        default='default',
                        choices=['Alpaca', 'Ziya', 'Default', 'PKU'],
                        help='Conversation type')

    return parser.parse_args()

ALPACA_TEMPLATE = (
    "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n"
    "{instruction}\n\n"
    "### Input:\n"
    "{input}\n\n"
    "### Response:\n"
)

ZIYA_TEMPLATE = (
    "Human: {input}\n"
    "Assistant: "
)

PKU_TEMPLATE = (
    'BEGINNING OF CONVERSATION: USER: {input} ASSISTANT:'
)

MAX_RETRIES = 10

def generate_prompt(args, logger, qa_data, answer_type="option_letter", tokenizer=None):
    # https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-2/
    # https://arxiv.org/pdf/2305.14975
    # https://arxiv.org/pdf/2406.08391
    if answer_type == "option letter":
        demo = '(A)'
    elif answer_type == "number":
        demo = 1
    else:
        demo = '(A)'
    if args.use_cot:
        logger.info("Using COT format for answers")
        PROMPT = (
            "For the following question, provide a step-by-step explanation of your thought process first, then offer your best answer and rate your confidence in the accuracy or helpfulness of each response on a scale from 0 to 10.\n"
            "Use the format demonstrated below for your response.\n"
            "```Example Format:\n"
            "Explanation: <Your detailed explanation here, outlining how you arrived at your answer.>\n"
            f"Answer: <Insert your concise answer here, which should include a {answer_type} (e.g., {demo})>\n"
            "Confidence: <Insert your numerical confidence level from 0 to 10, reflecting how certain you are that your answer is correct.>```\n\n"
            "Ensure that your response strictly adheres to this format. Explicitly include the words 'Explanation:', 'Answer:', and 'Confidence:' in your response."
        ).strip()

    elif args.use_top_k:        # noqa, you might find this useful or not idk
        logger.info("Using Top K format for answers")
        PROMPT = (
            "For the following question, provide your best 2 responses and your confidence in the accuracy or helpfulness of each response. Rate your confidence on a scale from 0 to 10.\n"
            "Use the format demonstrated below for your response.\n"
            "```Example Format:\n"
            f"Answer 1: <Your first response, which should include a {answer_type} (e.g., {demo})>\n"
            "Confidence 1: <Insert your numerical confidence level from 0 to 10, reflecting how certain you are that your firsrt answer is correct.>\n\n"
            f"Answer 2: <Your second response, which should include a {answer_type} (e.g., {demo})>\n"
            "Confidence 2: <Insert your numerical confidence level from 0 to 10, reflecting how certain you are that your second answer is correct.>```\n\n"
            "Ensure that your response strictly adheres to this format. Explicitly include the words 'Answer 1:', 'Confidence 1:', 'Answer 2:' and 'Confidence 2:' in your response."
        ).strip()

    else:
        logger.info("Using standard format for answers")
        PROMPT = (
            f"For the following question, provide your answer including only the {answer_type} first, followed by your confidence in the accuracy or helpfulness of your response. Rate your confidence on a scale from 0 to 10.\n"
            "Please respond only with your concise answer and a numerical confidence score. "
            "Do not include any additional text, characters, or explanations. Use the format demonstrated below for your response.\n"
            "```Example Format:\n"
            f"Answer: <Insert only the {answer_type} here (e.g., {demo})>\n"
            "Confidence: <Insert your numerical confidence level from 0 to 10, reflecting how certain you are that your answer is correct.>```\n\n"
            f"Ensure that your response strictly adheres to this format and contain only the {answer_type} and the confidence score. Explicitly include the words 'Answer:' and 'Confidence:' in your response."
        ).strip()

    prompts = []
    assert len(qa_data) > 0, "No data found"
    # generate prompt for all questions
    for idx, question in enumerate(qa_data.keys()):
        # wrap chat template if needed
        # if we use chat template, we put instructions in the system role
        # just found that tulu does not have a system message section...
        if args.use_chat_template and check_tokenizer_chat_template(tokenizer):
            if 'user' in tokenizer.chat_template and 'system' not in tokenizer.chat_template:
                chat = [
                    {'role': 'user', 'content': f"{PROMPT}\n\nQuestion: {question}"},
                ]
            # special case, it seems that this model does not follow instruction properly if we put it in system msg
            # default system msg: You are a helpful assistant. Please give a long and detailed answer.
            elif 'Matter' in args.model_name_or_path:
                chat = [
                    {'role': 'system', 'content': 'You are a helpful assistant. Please provide a short and concise answer.'},
                    {'role': 'user', 'content': f"{PROMPT}\n\nQuestion: {question}"},
                ]
            else:
                chat = [
                    {'role': 'system', 'content': PROMPT},
                    {'role': 'user', 'content': f"Question: {question}"},
                ]
            # be sure to add add_generation_prompt here
            prompt = tokenizer.apply_chat_template(chat, tokenize = False, add_generation_prompt = True)
        # prompt template for base model
        # use alpaca template
        elif args.conv == 'Alpaca':
            prompt = ALPACA_TEMPLATE.format(instruction=PROMPT, input=question)
        elif args.conv == 'Ziya':
            prompt = PROMPT + '\n\n' + f'Question: {question}'
            prompt = ZIYA_TEMPLATE.format(input=prompt)
        elif args.conv == 'PKU':
            prompt = PROMPT + '\n\n' + f'Question: {question}'
            prompt = PKU_TEMPLATE.format(input=prompt)

        prompts.append(prompt)
    assert len(prompts) == len(qa_data), f"Prompt generation failed. Expected {len(qa_data)} prompts, got {len(prompts)}"

    sample_prompt = prompts[0]
    logger.info(f"Sample prompt: {sample_prompt}")
    return prompts, qa_data


@torch.no_grad()
def query(args):
    accelerator = Accelerator()
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

    set_random_seed(args.seed)
    logger.info(f'Loading dataset {args.dataset} from {args.data_path}')
    qa_data = load_dataset(args.dataset, args.data_path, args.task_type)
    answer_type = "option letter" if args.task_type == "multi_choice_qa" else "number"

    # debug purposes
    #qa_data = dict(random.sample(qa_data.items(), 10))

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if args.use_chat_template and hasattr(tokenizer, "chat_template"):
        logger.info("Using chat template for prompts")
    else:
        logger.info(f"Using Conv type: {args.conv}")

    dtype_mapping = {
        'fp16': torch.float16,
        'fp32': torch.float32,
        'bf16': torch.bfloat16
    }
    args.dtype = dtype_mapping.get(args.dtype, 'auto')
    attn_implementation = "flash_attention_2" if args.use_flash_attention_2 else None
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, 
        torch_dtype = args.dtype, 
        attn_implementation = attn_implementation,
        trust_remote_code = True
    ).to(accelerator.device)
    logger.info(f"Loading model {args.model_name_or_path}")
    model.eval()
    logger.info('model is dtype: {}'.format(model.dtype))

    prompts, qa_data = generate_prompt(args, logger, qa_data, answer_type=answer_type, tokenizer=tokenizer)
    
    # add special tokens
    # according to https://huggingface.co/docs/transformers/chat_templating#what-are-generation-prompts
    # chat_template should handle special tokens, should not add twice during tokenization, however, some models like https://huggingface.co/OpenRLHF/Llama-3-8b-rm-mixture/blob/main/tokenizer_config.json does not have bos in chat template
    # so it is very tricky here, if the chat_template contains bos_token, we should not add twice
    add_special_tokens = True
    if check_tokenizer_chat_template(tokenizer):
        if 'bos_token' in tokenizer.chat_template or (tokenizer.bos_token is not None and tokenizer.bos_token in tokenizer.chat_template):
            add_special_tokens = False
    else:
        warnings.warn("Tokenizer might not have a chat template, please double check the model card, or set a template", UserWarning)

    if add_special_tokens:
        logger.info("Adding special tokens")
    else:
        logger.info("Not adding special tokens")
    do_sample = args.temperature > 1e-4

    accelerator.wait_for_everyone()
    device = accelerator.device

    # Data distributed inference (Tested)
    with accelerator.split_between_processes(prompts) as prompt:
        model_outputs = []
        model_probs = []
        if args.next_word_predictions:
            outputs, probs = get_next_word_predictions(
                model = model,
                device = device,
                tokenizer = tokenizer,
                prompts = prompt,
                batch_size = args.per_device_eval_batch_size,
            )   # noqa
            model_probs.extend(probs)  
        else:
            outputs = generate_completions(
                model = model, 
                device = device,
                tokenizer = tokenizer,
                prompts = prompt,
                # might just use batch size 1 here to avoid floating instability (might result in different results on different batch size)
                # reference: https://github.com/huggingface/transformers/issues/25420#issuecomment-1775317535
                # but batch size 1 is wasting resources
                batch_size = args.per_device_eval_batch_size,      
                add_special_tokens = add_special_tokens,
                use_stop_criteria = True,
                max_new_tokens = args.max_new_tokens,
                temperature = args.temperature,
                top_p = args.top_p,
                top_k = args.top_k,
                do_sample = do_sample,
                #stop_id_sequences=[[tokenizer.eos_token]],
            )
        model_outputs.extend(outputs)
         
    outputs = gather_object(model_outputs)
    probs = gather_object(model_probs)

    # for all the retry and saving, just do it on the main process
    if accelerator.is_main_process:
        targets = list(qa_data.values())

        # retry logic
        for i in range(MAX_RETRIES):
            retry_prompts = []
            retry_idx = []      # corresponding index of retry prompts

            # retry if no confidence find
            if not args.next_word_predictions:
                for idx, output in enumerate(outputs):
                    # find confidence
                    if not re.search(r'Confidence:\s*\d+', output):
                        retry_prompts.append(prompts[idx])
                        retry_idx.append(idx)

            if len(retry_prompts) > 0:
                logger.info(f"Retrying {len(retry_prompts)} prompts")
                retry_outputs = generate_completions(
                    model = model, 
                    device = device,
                    tokenizer = tokenizer,
                    prompts = retry_prompts,
                    batch_size = 1,     
                    add_special_tokens = add_special_tokens, 
                    use_stop_criteria = True,
                    max_new_tokens = args.max_new_tokens,
                    temperature = args.temperature,
                    top_p = args.top_p,
                    top_k = args.top_k,
                    do_sample = do_sample,
                )
                assert len(retry_outputs) == len(retry_prompts) == len(retry_idx), f"Expected {len(retry_prompts)} retry outputs, got {len(retry_outputs)}"
                for idx in retry_idx:
                    outputs[idx] = retry_outputs.pop(0)
            else:
                logger.info("No prompts to retry")
                break

        # Final Retry is we still cannot get confidence, we add confidence into the prompts
        # and retry (hopefully it can output some numbers this time)
        # reference: https://github.com/xu1868/SaySelf/blob/main/evaluation/evaluate.py
        retry_prompts = []
        retry_idx = []      
        if not args.next_word_predictions:
            for idx, output in enumerate(outputs):
                # find confidence
                if not re.search(r'Confidence:\s*\d+', output):
                    retry_prompts.append(prompts[idx] + output + ' Confidence: ')
                    retry_idx.append(idx)
        if len(retry_prompts) > 0:
            logger.info(f"Final Retrying {len(retry_prompts)} prompts")
            retry_outputs = generate_completions(
                model = model, 
                device = device,
                tokenizer = tokenizer,
                prompts = retry_prompts,
                max_new_tokens = 16,         # sometimes 10 is classified as 2 tokens, use 16 to be safe
                batch_size = 1,      
                skip_prompt = False,        # do not skip prompt this time (since we added confidence in the prompt)
                add_special_tokens = add_special_tokens,
                use_stop_criteria = True,
                temperature = args.temperature,
                top_p = args.top_p,
                top_k = args.top_k,
                do_sample = do_sample,
            )

            # now we remove origin prompt here
            clean_retry_outputs = []
            for idx, retry_output in zip(retry_idx, retry_outputs):
                encoded_prompts = tokenizer(prompts[idx], return_tensors="pt")
                prompt = tokenizer.decode(encoded_prompts.input_ids[0], skip_special_tokens=True)
                clean_retry_outputs.append(retry_output[len(prompt):])

            for idx in retry_idx:
                outputs[idx] = clean_retry_outputs.pop(0)

        count = 0
        if not args.next_word_predictions:
            for idx, output in enumerate(outputs):
                # find confidence
                if not re.search(r'Confidence:\s*\d+', output):
                    count += 1
            logger.info(f"Found {count} prompts without confidence")
        
        assert len(outputs) == len(targets), f"Expected {len(targets)} outputs, got {len(outputs)}"
        
        # save the json_outputs
        json_outputs = []
        # probs can be an empty list if we are not using next token predictions
        for idx, (output, target, prob) in enumerate(zip_longest(outputs, targets, probs, fillvalue=None)):
            answer = None
            confidence = None
            # save None for parsing later
            json_outputs.append({
                "question": list(qa_data.keys())[idx],
                "response": output,
                "target": target,
                "probabilities": prob,
                "confidence": confidence,
                "answer": answer,
            })
        os.makedirs(args.output_dir, exist_ok=True)
        output_file = os.path.join(args.output_dir, "results.json")
        with open(output_file, "w") as f:
            f.write(json.dumps(json_outputs, indent=4))

        logger.info(f"Inference completed. Results saved to {output_file}")


@torch.no_grad()
def main():
    args = parse_args()
    query(args)


if __name__ == "__main__":
    main()
    