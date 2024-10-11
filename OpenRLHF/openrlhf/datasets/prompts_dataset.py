from torch.utils.data import Dataset
from tqdm import tqdm
from .utils import exist_and_not_none
import random


# PROMPT = (
#     "For the following question, provide your best response first, followed by your confidence in the accuracy or helpfulness of your response. Rate your confidence on a scale from 0 to 10.\n"
#     "```Example Format:\n"
#     f"<Your responses>\n"
#     "Confidence: <Insert your numerical confidence level from 0 to 10, reflecting how certain you are that your answer is accurate or helpful.>```\n\n"
#     "Ensure that your response strictly adheres to this format. Explicitly include the word 'Confidence:' in your response."
# ).strip()


# def construct_confidence_prompt(prompt, tokenizer):
#     # Assume prompt is already formatted as chat messages [{'role': 'user', 'content': 'prompt'}]
#     if 'user' in tokenizer.chat_template and 'system' not in tokenizer.chat_template:
#         chat = [
#             {'role': 'user', 'content': f"{PROMPT}\n\nQuestion: {prompt[0]['content']}"},
#         ]
#     else:
#         chat = [
#             {'role': 'system', 'content': PROMPT},
#             {'role': 'user', 'content': f"Question: {prompt[0]['content']}"},
#         ] 
#     return chat


# def preprocess_data(data, tokenizer, input_template=None, input_key="input", apply_chat_template=None, ask_for_confidence=False) -> str:
#     if apply_chat_template:
#         # meta-math/MetaMathQ
#         if exist_and_not_none(data, 'query'):
#             prompt = data['query']
#             if ask_for_confidence:
#                 prompt = construct_confidence_prompt(prompt, tokenizer)
#             prompt = [{"content": prompt, 'role': 'user'}]
#         else:
#             prompt = data[input_key]
#             if ask_for_confidence and len(prompt) == 1:     # skip multi-turn prompts
#                 prompt = construct_confidence_prompt(prompt, tokenizer)
#         prompt = apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
#     # TODO: Add Confidence
#     else:
#         prompt = data[input_key]
#         if input_template:
#             prompt = input_template.format(prompt)
#     return prompt


def preprocess_data(data, tokenizer, input_template=None, input_key="input", apply_chat_template=None) -> str:
    if apply_chat_template:
        prompt = apply_chat_template(data[input_key], tokenize=False, add_generation_prompt=True)
    else:
        prompt = data[input_key]
        if input_template:
            prompt = input_template.format(prompt)
    return prompt


class PromptDataset(Dataset):
    """
    Dataset for PPO model

    Args:
        dataset: dataset for PPO model
        tokenizer: tokenizer for PPO model
        max_length: max length of input
    """

    def __init__(
        self,
        dataset,
        tokenizer,
        strategy,
        input_template=None,
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.tokenizer = tokenizer
        self.input_template = input_template
        self.n_samples_per_prompt = getattr(self.strategy.args, "n_samples_per_prompt", 1)

        input_key = getattr(self.strategy.args, "input_key", None)
        apply_chat_template = getattr(self.strategy.args, "apply_chat_template", False)

        if apply_chat_template:
            apply_chat_template = self.tokenizer.apply_chat_template

        self.prompts = []
        for data in tqdm(dataset, desc="Preprocessing data", disable=not self.strategy.is_rank_0()):
            prompt = preprocess_data(data, tokenizer, input_template, input_key, apply_chat_template)
            self.prompts.append(prompt)

    def __len__(self):
        length = len(self.prompts)
        return length * self.n_samples_per_prompt

    def __getitem__(self, idx):
        return self.prompts[idx // self.n_samples_per_prompt]
