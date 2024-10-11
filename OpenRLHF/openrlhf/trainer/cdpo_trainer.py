import math
from abc import ABC
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DistributedSampler
from tqdm import tqdm

from openrlhf.models import DPOLoss, KWiseDPOLoss


class CalibratedDPOTrainer(ABC):
    """
        Trainer to use while training reward model.

    Args:
        model (torch.nn.Module): the model to train
        strategy (Strategy): the strategy to use for training
        optim(Optimizer): the optimizer to use for training
        train_dataset (RewardDataset): the dataset to use for training
        eval_dataset (RewardDataset): the dataset to use for evaluation
        batch_size (int, defaults to 1): the batch size while training
        max_epochs (int, defaults to 2): the number of epochs to train
        optim_kwargs (dict, defaults to {'lr':1e-4}): the kwargs to use while initializing optimizer
    """

    def __init__(
        self,
        model,
        ref_model,
        strategy,
        tokenizer,
        optim: Optimizer,
        train_dataloader,
        eval_dataloader,
        scheduler,
        max_norm=0.5,
        beta=0.01,
        max_epochs: int = 2,
        cdpo_coefficient=0.5,
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.epochs = max_epochs
        self.max_norm = max_norm
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.ref_model = ref_model
        self.scheduler = scheduler
        self.optimizer = optim
        self.tokenizer = tokenizer
        self.args = strategy.args

        self.beta = beta
        self.cdpo_coefficient = cdpo_coefficient
        self.strategy.print('Using Coefficient: ', self.cdpo_coefficient)
        #self.loss_fn = KWiseDPOLoss(self.beta, self.args.label_smoothing, False)
        #self.strategy.print("KWise MLE DPO Loss for Training")
        self.loss_fn = DPOLoss(self.beta, self.args.label_smoothing, self.args.ipo)

        # Mixtral 8*7b
        self.aux_loss = self.args.aux_loss_coef > 1e-8

        # NLL loss
        self.nll_loss = self.args.nll_loss_coef > 1e-8

        # packing samples
        self.packing_samples = strategy.args.packing_samples

        self.compute_fp32_loss = self.strategy.args.compute_fp32_loss

        self._wandb = None
        if self.strategy.args.use_wandb and self.strategy.is_rank_0():
            import wandb

            self._wandb = wandb
            if not wandb.api.api_key:
                wandb.login(key=strategy.args.use_wandb)
            wandb.init(
                entity=strategy.args.wandb_org,
                project=strategy.args.wandb_project,
                group=strategy.args.wandb_group,
                name=strategy.args.wandb_run_name,
                config=strategy.args.__dict__,
                reinit=True,
            )

            wandb.define_metric("train/global_step")
            wandb.define_metric("train/*", step_metric="train/global_step", step_sync=True)
            wandb.define_metric("eval/global_step")
            wandb.define_metric("eval/*", step_metric="eval/global_step", step_sync=True)

    def fit(self, args):
        # get eval and save steps
        if args.eval_steps == -1:
            args.eval_steps = self.train_dataloader.__len__()  # Evaluate once per epoch
        if args.save_steps == -1:
            args.save_steps = float("inf")  # do not save ckpt

        global_step = 1
        epoch_bar = tqdm(
            range(self.epochs),
            desc="Train epoch",
            disable=not self.strategy.is_rank_0(),
        )
        for epoch in range(self.epochs):
            step_bar = tqdm(
                range(self.train_dataloader.__len__()),
                desc="Train step of epoch %d" % epoch,
                disable=not self.strategy.is_rank_0(),
            )

            if isinstance(self.train_dataloader.sampler, DistributedSampler):
                self.train_dataloader.sampler.set_epoch(epoch)

            self.model.train()
            self.ref_model.eval()
            acc_mean = 0
            loss_mean = 0
            # train
            for data in self.train_dataloader:
                if not self.packing_samples:
                    chosen_ids, c_mask, reject_ids, r_mask, cwh_ids, cwh_mask, cwl_ids, cwl_mask, rwh_ids, rwh_mask, rwl_ids, rwl_mask, id_lens = data
                    chosen_ids = chosen_ids.squeeze(1).to(torch.cuda.current_device())
                    c_mask = c_mask.squeeze(1).to(torch.cuda.current_device())
                    reject_ids = reject_ids.squeeze(1).to(torch.cuda.current_device())
                    r_mask = r_mask.squeeze(1).to(torch.cuda.current_device())
                    cwh_ids = cwh_ids.squeeze(1).to(torch.cuda.current_device())
                    cwh_mask = cwh_mask.squeeze(1).to(torch.cuda.current_device())
                    cwl_ids = cwl_ids.squeeze(1).to(torch.cuda.current_device())
                    cwl_mask = cwl_mask.squeeze(1).to(torch.cuda.current_device())
                    rwh_ids = rwh_ids.squeeze(1).to(torch.cuda.current_device())
                    rwh_mask = rwh_mask.squeeze(1).to(torch.cuda.current_device())
                    rwl_ids = rwl_ids.squeeze(1).to(torch.cuda.current_device())
                    rwl_mask = rwl_mask.squeeze(1).to(torch.cuda.current_device())

                    # dataloader stacked them
                    prompt_id_lens, confidence_prompt_chosen_id_lens, confidence_prompt_rejected_id_lens = id_lens[0]
                    prompt_id_lens = [prompt_id_lens]
                    confidence_prompt_chosen_id_lens = [confidence_prompt_chosen_id_lens]
                    confidence_prompt_rejected_id_lens = [confidence_prompt_rejected_id_lens]
                    chosen_logps, rejected_logps, cwh_logps, cwl_logps, rwh_logps, rwl_logps, aux_loss, nll_loss = self.concatenated_forward_all(
                        self.model, chosen_ids, c_mask, reject_ids, r_mask,
                        cwh_ids, cwh_mask, cwl_ids, cwl_mask,
                        rwh_ids, rwh_mask, rwl_ids, rwl_mask,
                        prompt_id_lens, confidence_prompt_chosen_id_lens, confidence_prompt_rejected_id_lens
                    )
                    with torch.no_grad():
                        reference_chosen_logps, reference_rejected_logps, reference_cwh_logps, reference_cwl_logps, reference_rwh_logps, reference_rwl_logps, _, _ = self.concatenated_forward_all(
                            self.ref_model, chosen_ids, c_mask, reject_ids, r_mask,
                            cwh_ids, cwh_mask, cwl_ids, cwl_mask,
                            rwh_ids, rwh_mask, rwl_ids, rwl_mask,
                            prompt_id_lens, confidence_prompt_chosen_id_lens, confidence_prompt_rejected_id_lens
                        )
                # TODO: do not use packing samples for now
                else:
                    raise ValueError('Does not support packing for now')
                    packed_input_ids, packed_attention_masks, packed_seq_lens, prompt_id_lens = data
                    packed_input_ids, packed_attention_masks = packed_input_ids.to(
                        torch.cuda.current_device()
                    ), packed_attention_masks.to(torch.cuda.current_device())
                    chosen_logps, rejected_logps, aux_loss, nll_loss = self.packed_samples_forward(
                        self.model, packed_input_ids, packed_attention_masks, packed_seq_lens, prompt_id_lens
                    )
                    with torch.no_grad():
                        reference_chosen_logps, reference_rejected_logps, _, _ = self.packed_samples_forward(
                            self.ref_model, packed_input_ids, packed_attention_masks, packed_seq_lens, prompt_id_lens
                        )

                if self.compute_fp32_loss:
                    chosen_logps = chosen_logps.float()
                    rejected_logps = rejected_logps.float()
                    reference_chosen_logps = reference_chosen_logps.float()
                    reference_rejected_logps = reference_rejected_logps.float()

                    cwh_logps = cwh_logps.float()
                    cwl_logps = cwl_logps.float()
                    rwh_logps = rwh_logps.float()
                    rwl_logps = rwl_logps.float()
                    reference_cwh_logps = reference_cwh_logps.float()
                    reference_cwl_logps = reference_cwl_logps.float()
                    reference_rwh_logps = reference_rwh_logps.float()
                    reference_rwl_logps = reference_rwl_logps.float()

                loss1, cwh_reward, cwl_reward = self.loss_fn(
                    cwh_logps, cwl_logps, reference_cwh_logps, reference_cwl_logps
                )
                loss2, rwl_reward, rwh_reward = self.loss_fn(
                    rwl_logps, rwh_logps, reference_rwl_logps, reference_rwh_logps
                )
                loss3, chosen_reward, rejected_reward = self.loss_fn(
                    chosen_logps, rejected_logps, reference_chosen_logps, reference_rejected_logps
                )
                preference_loss = self.cdpo_coefficient * (loss1 + loss2) + loss3

                # # loss function
                # cwh_logps = cwh_logps.unsqueeze(1)
                # cwl_logps = cwl_logps.unsqueeze(1)
                # rwh_logps = rwh_logps.unsqueeze(1)
                # rwl_logps = rwl_logps.unsqueeze(1)
                # # cwh < chosen < rwl < cwl < rejected < rwh
                # all_logps = torch.cat(
                #     [cwh_logps, rwl_logps, cwl_logps, rwh_logps], dim=1
                # )
                # reference_cwh_logps = reference_cwh_logps.unsqueeze(1)
                # reference_cwl_logps = reference_cwl_logps.unsqueeze(1)
                # reference_rwh_logps = reference_rwh_logps.unsqueeze(1)
                # reference_rwl_logps = reference_rwl_logps.unsqueeze(1)
                # all_reference_logps = torch.cat(
                #     [reference_cwh_logps, reference_rwl_logps, reference_cwl_logps, reference_rwh_logps], dim=1
                # )

                # preference_loss, cwh_reward, rwl_reward, cwl_reward, rwh_reward = self.loss_fn(
                #     all_logps, all_reference_logps
                # )

                # mixtral
                if not self.aux_loss:
                    aux_loss = 0
                # nll loss
                if not self.nll_loss:
                    nll_loss = 0

                loss = preference_loss + aux_loss * self.args.aux_loss_coef + nll_loss * self.args.nll_loss_coef
                self.strategy.backward(loss, self.model, self.optimizer)
                self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler)

                #acc = (chosen_reward > rejected_reward).float().mean().item()
                acc = ((chosen_reward > rejected_reward).float().mean().item() + (cwh_reward > cwl_reward).float().mean().item() + (rwl_reward > rwh_reward).float().mean().item()) / 3
                acc_mean = acc_mean * 0.9 + 0.1 * acc
                loss_mean = loss_mean * 0.9 + 0.1 * preference_loss.item()
                # dpo logs
                logs_dict = {
                    "loss": preference_loss.item(),
                    "acc": acc,
                    "chosen_reward": chosen_reward.mean().item(),
                    "reject_reward": rejected_reward.mean().item(),
                    "loss_mean": loss_mean,
                    "acc_mean": acc_mean,
                    "lr": self.scheduler.get_last_lr()[0],
                }
                logs_dict["cwh_reward"] = cwh_reward.mean().item()
                logs_dict["cwl_reward"] = cwl_reward.mean().item()
                logs_dict["rwh_reward"] = rwh_reward.mean().item()
                logs_dict["rwl_reward"] = rwl_reward.mean().item()
                if self.nll_loss:
                    logs_dict["nll_loss"] = nll_loss.item()
                # logs/checkpoints/evaluate
                self.save_logs_and_checkpoints(args, global_step, step_bar, logs_dict)

                step_bar.update()
                global_step += 1
            epoch_bar.update()

        if self._wandb is not None and self.strategy.is_rank_0():
            self._wandb.finish()

    # logs/checkpoints/evaluate
    def save_logs_and_checkpoints(self, args, global_step, step_bar, logs_dict={}):
        # logs
        if global_step % args.logging_steps == 0:
            # step bar
            logs_dict = self.strategy.all_reduce(logs_dict)
            step_bar.set_postfix(logs_dict)

            # wandb
            if (
                self._wandb is not None
                and self.strategy.is_rank_0()
                and global_step % self.strategy.accumulated_gradient == 0
            ):
                logs = {"train/%s" % k: v for k, v in {**logs_dict, "global_step": global_step}.items()}
                self._wandb.log(logs)

        # eval
        if global_step % args.eval_steps == 0:
            self.evaluate(self.eval_dataloader, global_step)
        # save ckpt
        # TODO: save best model on dev, use loss/perplexity on whole dev dataset as metric
        if global_step % args.save_steps == 0:
            tag = f"global_step{global_step}"
            self.strategy.save_ckpt(self.model.model, args.ckpt_path, tag, args.max_ckpt_num, args.max_ckpt_mem)

    def evaluate(self, eval_dataloader, steps=0):
        self.model.eval()
        with torch.no_grad():
            step_bar = tqdm(
                range(eval_dataloader.__len__()),
                desc="Eval stage of global_step %d" % steps,
                disable=not self.strategy.is_rank_0(),
            )
            acc_sum = 0
            loss_sum = 0
            times = 0
            for data in eval_dataloader:
                if not self.packing_samples:
                    chosen_ids, c_mask, reject_ids, r_mask, prompt_id_lens = data
                    chosen_ids = chosen_ids.squeeze(1).to(torch.cuda.current_device())
                    c_mask = c_mask.squeeze(1).to(torch.cuda.current_device())
                    reject_ids = reject_ids.squeeze(1).to(torch.cuda.current_device())
                    r_mask = r_mask.squeeze(1).to(torch.cuda.current_device())

                    chosen_logps, rejected_logps, aux_loss, _ = self.concatenated_forward(
                        self.model, chosen_ids, c_mask, reject_ids, r_mask, prompt_id_lens
                    )
                    with torch.no_grad():
                        reference_chosen_logps, reference_rejected_logps, _, _ = self.concatenated_forward(
                            self.ref_model, chosen_ids, c_mask, reject_ids, r_mask, prompt_id_lens
                        )
                # TODO: do not use packing samples for now
                else:
                    raise ValueError('Does not support packing for now')
                    packed_input_ids, packed_attention_masks, packed_seq_lens, prompt_id_lens = data
                    packed_input_ids, packed_attention_masks = packed_input_ids.to(
                        torch.cuda.current_device()
                    ), packed_attention_masks.to(torch.cuda.current_device())
                    chosen_logps, rejected_logps, aux_loss, _ = self.packed_samples_forward(
                        self.model, packed_input_ids, packed_attention_masks, packed_seq_lens, prompt_id_lens
                    )
                    with torch.no_grad():
                        reference_chosen_logps, reference_rejected_logps, _, _ = self.packed_samples_forward(
                            self.ref_model, packed_input_ids, packed_attention_masks, packed_seq_lens, prompt_id_lens
                        )

                loss, chosen_reward, reject_reward = self.loss_fn(
                    chosen_logps, rejected_logps, reference_chosen_logps, reference_rejected_logps
                )
                acc_sum += (chosen_reward > reject_reward).float().mean().item()
                loss_sum += loss.item()
                times += 1
                step_bar.update()

            logs = {
                "eval_loss": loss_sum / times,
                "acc_mean": acc_sum / times,
            }
            logs = self.strategy.all_reduce(logs)
            step_bar.set_postfix(logs)
            if self._wandb is not None and self.strategy.is_rank_0():
                logs = {"eval/%s" % k: v for k, v in {**logs, "global_step": steps}.items()}
                self._wandb.log(logs)
        self.model.train()  # reset model state

    def concatenated_forward(self, model, chosen_ids, c_mask, reject_ids, r_mask, prompt_id_lens):
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        input_ids, att_masks, prompt_id_lens = self.concatenated_inputs(
            chosen_ids, c_mask, reject_ids, r_mask, prompt_id_lens
        )
        output = model(input_ids, attention_mask=att_masks, return_output=True)
        all_logits = output["logits"]
        all_logps_sum, all_logps_mean = self._get_batch_logps(
            all_logits, input_ids, att_masks, prompt_id_lens, average_log_prob=False
        )
        chosen_logps = all_logps_sum[: chosen_ids.shape[0]]
        rejected_logps = all_logps_sum[chosen_ids.shape[0] :]
        aux_loss = output.aux_loss if "aux_loss" in output else []
        return chosen_logps, rejected_logps, aux_loss, -all_logps_mean[: chosen_ids.shape[0]].mean()

    def concatenated_inputs(self, chosen_ids, c_mask, reject_ids, r_mask, prompt_id_lens):
        """Concatenate the chosen and rejected inputs into a single tensor.

        Args:
            batch: A batch of data. Must contain the keys 'chosen_input_ids' and 'rejected_input_ids', which are tensors of shape (batch_size, sequence_length).

        Returns:
            A dictionary containing the concatenated inputs under the key 'concatenated_input_ids'.
        """

        def pad_to_length(tensor, length, pad_value, dim=-1):
            if tensor.size(dim) >= length:
                return tensor
            else:
                pad_size = list(tensor.shape)
                pad_size[dim] = length - tensor.size(dim)
                return torch.cat(
                    [tensor, pad_value * torch.ones(*pad_size, dtype=tensor.dtype, device=tensor.device)], dim=dim
                )

        max_length = max(chosen_ids.shape[1], reject_ids.shape[1])
        inputs_ids = torch.cat(
            (
                pad_to_length(chosen_ids, max_length, self.tokenizer.pad_token_id),
                pad_to_length(reject_ids, max_length, self.tokenizer.pad_token_id),
            ),
            dim=0,
        )
        max_length = max(c_mask.shape[1], r_mask.shape[1])
        att_masks = torch.cat((pad_to_length(c_mask, max_length, 0), pad_to_length(r_mask, max_length, 0)), dim=0)
        return inputs_ids, att_masks, prompt_id_lens * 2

    def _get_batch_logps(
        self,
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        attention_mask,
        prompt_id_lens,
        average_log_prob: bool = False,
    ) -> torch.FloatTensor:
        """Compute the log probabilities of the given labels under the given logits.

        Args:
            logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
            labels: Labels for which to compute the log probabilities. Label tokens with a value of -100 are ignored. Shape: (batch_size, sequence_length)
            average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.

        Returns:
            A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
        """
        assert average_log_prob == False
        assert logits.shape[:-1] == labels.shape

        labels = labels[:, 1:].clone()
        logits = logits[:, :-1, :]

        loss_masks = attention_mask.clone().bool()
        # mask prompts
        for mask, source_len in zip(loss_masks, prompt_id_lens):
            mask[:source_len] = False
        loss_masks = loss_masks[:, 1:]

        # dummy token; we'll ignore the losses on these tokens later
        labels[loss_masks == False] = 0
        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

        logprobs_sums = (per_token_logps * loss_masks).sum(-1)
        logprobs_means = (per_token_logps * loss_masks).sum(-1) / loss_masks.sum(-1)
        return logprobs_sums, logprobs_means

    def packed_samples_forward(self, model, packed_input_ids, packed_attention_masks, packed_seq_lens, prompt_id_lens):
        output = model(packed_input_ids, attention_mask=packed_attention_masks, return_output=True)
        all_logits = output["logits"]
        all_logps_sum, all_logps_mean = self._packed_get_batch_logps(
            all_logits,
            packed_input_ids,
            packed_attention_masks,
            prompt_id_lens * 2,
            packed_seq_lens,
            average_log_prob=False,
        )
        chosen_logps = all_logps_sum[: len(packed_seq_lens) // 2]
        rejected_logps = all_logps_sum[len(packed_seq_lens) // 2 :]
        aux_loss = output.aux_loss if "aux_loss" in output else []
        return chosen_logps, rejected_logps, aux_loss, -all_logps_mean[: len(packed_seq_lens) // 2].mean()

    def _packed_get_batch_logps(
        self,
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        attention_mask,
        prompt_id_lens,
        packed_seq_lens,
        average_log_prob: bool = False,
    ) -> torch.FloatTensor:
        assert average_log_prob == False
        assert logits.shape[:-1] == labels.shape

        labels = labels[:, 1:].clone()
        logits = logits[:, :-1, :]
        loss_masks = attention_mask.clone().bool()

        index = 0
        for i, seq_len in enumerate(packed_seq_lens):
            loss_masks[0, index : index + prompt_id_lens[i]] = False
            index = index + seq_len

        loss_masks = loss_masks[:, 1:]
        labels[loss_masks == False] = 0

        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

        logprobs_sums = []
        logprobs_means = []
        index = 0
        for i, seq_len in enumerate(packed_seq_lens):
            seq = per_token_logps[0, index : index + seq_len - 1]
            mask = loss_masks[0, index : index + seq_len - 1]
            logprobs_sums.append((seq * mask).sum())
            logprobs_means.append((seq * mask).sum() / mask.sum())
            index = index + seq_len

        return torch.stack(logprobs_sums), torch.stack(logprobs_means)
    

    def concatenated_forward_all(
        self, model,
        chosen_ids, c_mask, reject_ids, r_mask,
        cwh_ids, cwh_mask, cwl_ids, cwl_mask,
        rwh_ids, rwh_mask, rwl_ids, rwl_mask,
        prompt_id_lens, confidence_prompt_chosen_id_lens, confidence_prompt_rejected_id_lens
    ):
        input_ids, att_masks, prompt_id_lens, confidence_prompt_chosen_id_lens, confidence_prompt_rejected_id_lens = self.concatenated_inputs_all(
            chosen_ids, c_mask, reject_ids, r_mask,
            cwh_ids, cwh_mask, cwl_ids, cwl_mask,
            rwh_ids, rwh_mask, rwl_ids, rwl_mask,
            prompt_id_lens, confidence_prompt_chosen_id_lens, confidence_prompt_rejected_id_lens
        )
        output = model(input_ids, attention_mask=att_masks, return_output=True)
        all_logits = output["logits"]
        all_logps_sum, all_logps_mean = self._get_all_batch_logps(
            all_logits, input_ids, att_masks, prompt_id_lens, confidence_prompt_chosen_id_lens, confidence_prompt_rejected_id_lens, average_log_prob=False
        )
        start_idx = 0
        chosen_logps = all_logps_sum[start_idx: start_idx + chosen_ids.shape[0]]
        start_idx += chosen_ids.shape[0]
        rejected_logps = all_logps_sum[start_idx: start_idx + reject_ids.shape[0]]
        start_idx += reject_ids.shape[0]
        cwh_logps = all_logps_sum[start_idx: start_idx + cwh_ids.shape[0]]
        start_idx += cwh_ids.shape[0]
        cwl_logps = all_logps_sum[start_idx: start_idx + cwl_ids.shape[0]]
        start_idx += cwl_ids.shape[0]
        rwh_logps = all_logps_sum[start_idx: start_idx + rwh_ids.shape[0]]
        start_idx += rwh_ids.shape[0]
        rwl_logps = all_logps_sum[start_idx: start_idx + rwl_ids.shape[0]]

        aux_loss = output.aux_loss if "aux_loss" in output else []
        return chosen_logps, rejected_logps, cwh_logps, cwl_logps, rwh_logps, rwl_logps, aux_loss, -all_logps_mean[: chosen_ids.shape[0]].mean()
    
    def concatenated_inputs_all(
        self, 
        chosen_ids, c_mask, reject_ids, r_mask,
        cwh_ids, cwh_mask, cwl_ids, cwl_mask,
        rwh_ids, rwh_mask, rwl_ids, rwl_mask,
        prompt_id_lens, confidence_prompt_chosen_id_lens, confidence_prompt_rejected_id_lens
    ):
        def pad_to_length(tensor, length, pad_value, dim=-1):
            if tensor.size(dim) >= length:
                return tensor
            else:
                pad_size = list(tensor.shape)
                pad_size[dim] = length - tensor.size(dim)
                return torch.cat(
                    [tensor, pad_value * torch.ones(*pad_size, dtype=tensor.dtype, device=tensor.device)], dim=dim
                )
        
        max_length = max(
            chosen_ids.shape[1], reject_ids.shape[1],
            cwh_ids.shape[1], cwl_ids.shape[1],
            rwh_ids.shape[1], rwl_ids.shape[1]
        )
        input_ids = torch.cat(
            (
                pad_to_length(chosen_ids, max_length, self.tokenizer.pad_token_id),
                pad_to_length(reject_ids, max_length, self.tokenizer.pad_token_id),
                pad_to_length(cwh_ids, max_length, self.tokenizer.pad_token_id),
                pad_to_length(cwl_ids, max_length, self.tokenizer.pad_token_id),
                pad_to_length(rwh_ids, max_length, self.tokenizer.pad_token_id),
                pad_to_length(rwl_ids, max_length, self.tokenizer.pad_token_id),
            ),
            dim=0
        )
        max_length = max(
            c_mask.shape[1], r_mask.shape[1],
            cwh_mask.shape[1], cwl_mask.shape[1],
            rwh_mask.shape[1], rwl_mask.shape[1]
        )
        att_masks = torch.cat(
            (
                pad_to_length(c_mask, max_length, 0),
                pad_to_length(r_mask, max_length, 0),
                pad_to_length(cwh_mask, max_length, 0),
                pad_to_length(cwl_mask, max_length, 0),
                pad_to_length(rwh_mask, max_length, 0),
                pad_to_length(rwl_mask, max_length, 0),
            ),
            dim=0
        )
        return input_ids, att_masks, prompt_id_lens * 2, confidence_prompt_chosen_id_lens * 2, confidence_prompt_rejected_id_lens * 2
    
    def _get_all_batch_logps(
        self, 
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        attention_mask,
        prompt_id_lens,
        confidence_prompt_chosen_id_lens,
        confidence_prompt_rejected_id_lens,
        average_log_prob: bool = False,
    ):
        assert average_log_prob == False
        assert logits.shape[:-1] == labels.shape

        labels = labels[:, 1:].clone()
        logits = logits[:, :-1, :]

        loss_masks = attention_mask.clone().bool()
        # mask prompts
        for mask, source_len in zip(loss_masks, prompt_id_lens + confidence_prompt_chosen_id_lens + confidence_prompt_rejected_id_lens):
            mask[:source_len] = False

        loss_masks = loss_masks[:, 1:] 

        # dummy token; we'll ignore the losses on these tokens later
        labels[loss_masks == False] = 0
        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

        logprobs_sums = (per_token_logps * loss_masks).sum(-1)
        logprobs_means = (per_token_logps * loss_masks).sum(-1) / loss_masks.sum(-1)
        return logprobs_sums, logprobs_means
