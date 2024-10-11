import math
from abc import ABC

import loralib as lora
import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DistributedSampler
from tqdm import tqdm

from openrlhf.models import LogExpLoss, PairWiseLoss, KWiseMLELoss
from openrlhf.utils import return_value

# TODO: can probably remove chosen and rejected from the dataloader and forward_all?
# so that we can reduce memory and boost speed
class CalibratedRewardModelTrainer(ABC):
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
        strategy,
        optim: Optimizer,
        train_dataloader,
        eval_dataloader,
        scheduler,
        tokenizer,
        max_norm=0.5,
        max_epochs: int = 2,
        loss="sigmoid",
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.epochs = max_epochs
        self.max_norm = max_norm
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.scheduler = scheduler
        self.optimizer = optim
        self.tokenizer = tokenizer
        self.args = strategy.args

        self.kwise_loss = False
        if loss == "sigmoid":
            self.loss_fn = PairWiseLoss()
            self.strategy.print("LogSigmoid Loss for training")
        elif loss == 'KWise':
            self.loss_fn = KWiseMLELoss()
            self.strategy.print("KWise MLE Loss for training")
            self.kwise_loss = True
        else:
            self.loss_fn = LogExpLoss()
            self.strategy.print("LogExp Loss for training")

    
        # just use sigmoid loss for eval 
        # we use the original dataloader for eval data
        self.pairwise_loss_fn = PairWiseLoss()

        # Mixtral 8*7b
        self.aux_loss = self.args.aux_loss_coef > 1e-8

        # packing samples
        self.packing_samples = strategy.args.packing_samples

        self.margin_loss = self.strategy.args.margin_loss
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
        epoch_bar = tqdm(range(self.epochs), desc="Train epoch", disable=not self.strategy.is_rank_0())
        for epoch in range(self.epochs):
            #  train
            step_bar = tqdm(
                range(self.train_dataloader.__len__()),
                desc="Train step of epoch %d" % epoch,
                disable=not self.strategy.is_rank_0(),
            )

            if isinstance(self.train_dataloader.sampler, DistributedSampler):
                self.train_dataloader.sampler.set_epoch(epoch)

            self.model.train()
            acc_mean = 0
            loss_mean = 0
            for data in self.train_dataloader:
                if not self.packing_samples:
                    chosen_ids, c_mask, reject_ids, r_mask, cwh_ids, cwh_mask, cwl_ids, cwl_mask, rwh_ids, rwh_mask, rwl_ids, rwl_mask, margin = data
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

                    chosen_reward, reject_reward, cwh_reward, cwl_reward, rwh_reward, rwl_reward, aux_loss = self.concatenated_forward_all(
                        self.model,
                        chosen_ids, c_mask, reject_ids, r_mask,
                        cwh_ids, cwh_mask, cwl_ids, cwl_mask,
                        rwh_ids, rwh_mask, rwl_ids, rwl_mask
                    )
                # TODO: do not use packing samples for now
                else:
                    raise NotImplementedError("Does not support packing samples yet")
                    packed_input_ids, packed_attention_masks, packed_seq_lens, margin = data
                    packed_input_ids, packed_attention_masks = packed_input_ids.to(
                        torch.cuda.current_device()
                    ), packed_attention_masks.to(torch.cuda.current_device())

                    chosen_reward, reject_reward, aux_loss = self.packed_samples_forward(
                        self.model, packed_input_ids, packed_attention_masks, packed_seq_lens
                    )

                if self.margin_loss:
                    margin = torch.tensor(margin).to(torch.cuda.current_device())
                else:
                    margin = None

                # loss function
                if self.compute_fp32_loss:
                    chosen_reward = chosen_reward.float()
                    reject_reward = reject_reward.float()
                    cwh_reward = cwh_reward.float()
                    cwl_reward = cwl_reward.float()
                    rwh_reward = rwh_reward.float()
                    rwl_reward = rwl_reward.float()

                # version 3 and version 4
                if self.kwise_loss:
                    cwh_reward = cwh_reward.unsqueeze(1)
                    cwl_reward = cwl_reward.unsqueeze(1)
                    rwh_reward = rwh_reward.unsqueeze(1)
                    rwl_reward = rwl_reward.unsqueeze(1)
                    all_rewards = torch.cat(
                        [cwh_reward, cwl_reward, rwl_reward, rwh_reward], dim=1
                    )
                    preference_loss = self.loss_fn(all_rewards, margin)
                else:
                    # version 5
                    preference_loss = self.loss_fn(cwh_reward, cwl_reward, margin) + self.loss_fn(rwl_reward, rwh_reward, margin)
       
                # mixtral
                if not self.aux_loss:
                    aux_loss = 0

                loss = preference_loss + aux_loss * self.args.aux_loss_coef
                self.strategy.backward(loss, self.model, self.optimizer)
                self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler)

                acc = ((chosen_reward > reject_reward).float().mean().item() + (cwh_reward > cwl_reward).float().mean().item() + (rwl_reward > rwh_reward).float().mean().item()) / 3
                acc_mean = acc_mean * 0.9 + 0.1 * acc
                loss_mean = loss_mean * 0.9 + 0.1 * preference_loss.item()
                # optional rm info
                logs_dict = {
                    "loss": preference_loss.item(),
                    "acc": acc,
                    "chosen_reward": chosen_reward.mean().item(),
                    "reject_reward": reject_reward.mean().item(),
                    "loss_mean": loss_mean,
                    "acc_mean": acc_mean,
                    "lr": self.scheduler.get_last_lr()[0],
                }
                if self.aux_loss:
                    logs_dict["aux_loss"] = aux_loss.item()
                logs_dict["cwh_reward"] = return_value(cwh_reward, op='mean')
                logs_dict["cwl_reward"] = return_value(cwl_reward, op='mean')
                logs_dict["rwh_reward"] = return_value(rwh_reward, op='mean')
                logs_dict["rwl_reward"] = return_value(rwl_reward, op='mean')
                # logs/checkpoints/evaluate
                self.save_logs_and_checkpoints(args, global_step, step_bar, logs_dict)

                step_bar.update()
                global_step += 1
            epoch_bar.update()

        if self._wandb is not None and self.strategy.is_rank_0():
            self._wandb.finish()

    # logs/checkpoints/evaluate
    def save_logs_and_checkpoints(self, args, global_step, step_bar, logs_dict={}):
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
            self.strategy.save_ckpt(self.model, args.ckpt_path, tag, args.max_ckpt_num, args.max_ckpt_mem)

    def evaluate(self, eval_dataloader, steps=0):
        step_bar = tqdm(
            range(eval_dataloader.__len__()),
            desc="Eval stage of steps %d" % steps,
            disable=not self.strategy.is_rank_0(),
        )
        self.model.eval()
        with torch.no_grad():
            acc = 0
            confidence_acc = 0
            rewards = []
            loss_sum = 0
            for data in eval_dataloader:
                if not self.packing_samples:
                    #chosen_ids, c_mask, reject_ids, r_mask, margin = data
                    chosen_ids, c_mask, reject_ids, r_mask, cwh_ids, cwh_mask, cwl_ids, cwl_mask, rwh_ids, rwh_mask, rwl_ids, rwl_mask, margin = data
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

                    # chosen_reward, reject_reward, _ = self.concatenated_forward(
                    #     self.model, chosen_ids, c_mask, reject_ids, r_mask
                    # )

                    chosen_reward, reject_reward, cwh_reward, cwl_reward, rwh_reward, rwl_reward, aux_loss = self.concatenated_forward_all(
                        self.model,
                        chosen_ids, c_mask, reject_ids, r_mask,
                        cwh_ids, cwh_mask, cwl_ids, cwl_mask,
                        rwh_ids, rwh_mask, rwl_ids, rwl_mask
                    )

                else:
                    raise NotImplementedError("Does not support packing samples yet")
                    packed_input_ids, packed_attention_masks, packed_seq_lens, margin = data
                    packed_input_ids, packed_attention_masks = packed_input_ids.to(
                        torch.cuda.current_device()
                    ), packed_attention_masks.to(torch.cuda.current_device())

                    chosen_reward, reject_reward, _ = self.packed_samples_forward(
                        self.model, packed_input_ids, packed_attention_masks, packed_seq_lens
                    )

                if self.margin_loss:
                    margin = torch.tensor(margin).to(torch.cuda.current_device())
                else:
                    margin = None

                loss = self.pairwise_loss_fn(chosen_reward, reject_reward, margin)

                rewards += [chosen_reward.flatten(), reject_reward.flatten(), cwh_reward.flatten(), cwl_reward.flatten(), rwh_reward.flatten(), rwl_reward.flatten()]
                acc += (chosen_reward > reject_reward).float().mean().item()
                confidence_acc += ((cwh_reward > cwl_reward).float().mean().item() +  (rwl_reward > rwh_reward).float().mean().item()) / 2
                loss_sum += loss.item()
                step_bar.update()

            acc_mean = acc / self.eval_dataloader.__len__()
            confidence_acc_mean = confidence_acc / self.eval_dataloader.__len__()
            loss_mean = loss_sum / self.eval_dataloader.__len__()

            rewards = torch.cat(rewards).float()
            rewards = self.strategy.all_gather(rewards)
            reward_mean = torch.mean(rewards)
            reward_std = torch.std(rewards).clamp(min=1e-8)

            # save mean std
            self.strategy.print("Set reward mean std")
            unwrap_model = self.strategy._unwrap_model(self.model)
            unwrap_model.config.mean = reward_mean.item()
            unwrap_model.config.std = reward_std.item()

            bar_dict = {
                "eval_loss": loss_mean,
                "acc_mean": acc_mean,
                "confidence_acc_mean": confidence_acc_mean,
                "reward_mean": reward_mean.item(),
                "reward_std": reward_std.item(),
            }
            logs = self.strategy.all_reduce(bar_dict)
            step_bar.set_postfix(logs)

            histgram = torch.histogram(rewards.cpu(), bins=10, range=(-10, 10), density=True) * 2
            self.strategy.print("histgram")
            self.strategy.print(histgram)

            if self._wandb is not None and self.strategy.is_rank_0():
                logs = {"eval/%s" % k: v for k, v in {**logs, "global_step": steps}.items()}
                self._wandb.log(logs)
        self.model.train()  # reset model state

    def concatenated_forward(self, model, chosen_ids, c_mask, reject_ids, r_mask):
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        input_ids, att_masks = self.concatenated_inputs(chosen_ids, c_mask, reject_ids, r_mask)
        all_values, output = model(input_ids, attention_mask=att_masks, return_output=True)
        chosen_rewards = all_values[: chosen_ids.shape[0]]
        rejected_rewards = all_values[chosen_ids.shape[0] :]
        aux_loss = output.aux_loss if "aux_loss" in output else []
        return chosen_rewards, rejected_rewards, aux_loss

    def concatenated_inputs(self, chosen_ids, c_mask, reject_ids, r_mask):
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
                # left pad
                return torch.cat(
                    [pad_value * torch.ones(*pad_size, dtype=tensor.dtype, device=tensor.device), tensor], dim=dim
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
        return inputs_ids, att_masks

    def packed_samples_forward(self, model, packed_input_ids, packed_attention_masks, packed_seq_lens):
        all_values, output = model(packed_input_ids, attention_mask=packed_attention_masks, return_output=True)
        half_len = len(packed_seq_lens) // 2
        rewards = []
        index = 0
        for i, seq_len in enumerate(packed_seq_lens):
            index = index + seq_len
            rewards.append(all_values[0, index - 1])
        rewards = torch.stack(rewards)

        chosen_rewards = rewards[:half_len]
        rejected_rewards = rewards[half_len:]
        aux_loss = output.aux_loss if "aux_loss" in output else []

        return chosen_rewards, rejected_rewards, aux_loss
    

    def concatenated_forward_all(
        self, model,
        chosen_ids, c_mask, reject_ids, r_mask,
        cwh_ids, cwh_mask, cwl_ids, cwl_mask,
        rwh_ids, rwh_mask, rwl_ids, rwl_mask
    ):
        input_ids, att_masks = self.concatenated_inputs_all(
            chosen_ids, c_mask, reject_ids, r_mask,
            cwh_ids, cwh_mask, cwl_ids, cwl_mask,
            rwh_ids, rwh_mask, rwl_ids, rwl_mask
        )
        all_values, output = model(input_ids, attention_mask=att_masks, return_output=True)

        start_idx = 0
        chosen_rewards = all_values[start_idx: start_idx + chosen_ids.shape[0]]
        start_idx += chosen_ids.shape[0]
        reject_rewards = all_values[start_idx: start_idx + reject_ids.shape[0]]
        start_idx += reject_ids.shape[0]
        cwh_rewards = all_values[start_idx: start_idx + cwh_ids.shape[0]]
        start_idx += cwh_ids.shape[0]
        cwl_rewards = all_values[start_idx: start_idx + cwl_ids.shape[0]]
        start_idx += cwl_ids.shape[0]
        rwh_rewards = all_values[start_idx: start_idx + rwh_ids.shape[0]]
        start_idx += rwh_ids.shape[0]
        rwl_rewards = all_values[start_idx: start_idx + rwl_ids.shape[0]]

        aux_loss = output.aux_loss if "aux_loss" in output else []
        return chosen_rewards, reject_rewards, cwh_rewards, cwl_rewards, rwh_rewards, rwl_rewards, aux_loss
    
    def concatenated_inputs_all(
        self,
        chosen_ids, c_mask, reject_ids, r_mask,
        cwh_ids, cwh_mask, cwl_ids, cwl_mask,
        rwh_ids, rwh_mask, rwl_ids, rwl_mask
    ):
        def pad_to_length(tensor, length, pad_value, dim=-1):
            if tensor.size(dim) >= length:
                return tensor
            else:
                pad_size = list(tensor.shape)
                pad_size[dim] = length - tensor.size(dim)
                # left pad
                return torch.cat(
                    [pad_value * torch.ones(*pad_size, dtype=tensor.dtype, device=tensor.device), tensor], dim=dim
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
        return input_ids, att_masks