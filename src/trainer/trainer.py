from pathlib import Path
from arguments import Args
from typing import Optional
import time
from tqdm import tqdm

from torch import nn, Tensor
from torch.utils.data import DataLoader
from accelerate import Accelerator, DistributedType
import torch
from transformers.modeling_outputs import CausalLMOutputWithPast

from parallel.ulysses_attn import prepare_ulysses_attn_inputs


class LMTrainer:
    def __init__(
        self,
        args: Args,
        output_dir: Path,
        accelerator: Accelerator,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
        train_loader: DataLoader,
        val_loader: None | DataLoader = None,
        include_num_input_tokens_seen: bool = True,
    ):
        self.args = args
        self.output_dir = output_dir
        self.accelerator = accelerator
        self.raw_model = model
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.lr_scheduler = lr_scheduler
        self.val_loader = val_loader

        self.include_num_input_tokens_seen = include_num_input_tokens_seen
        if self.include_num_input_tokens_seen:
            self.main_input_name = 'input_ids'
            self.n_input_tokens_seen = 0

    @torch.no_grad()
    def get_validation_loss(
        self,
        accelerator: Accelerator,
        model: nn.Module,
        data_loader: DataLoader,
        n_batches: int,
        device: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Perform evaluation on `n_batches` batches from a dataloader.
        """
        model.eval()
        losses = torch.zeros(n_batches)
        accelerator.print("==== Evaluation ====")
        i = 0
        for i, batch in enumerate(data_loader):
            if i == n_batches:
                break
            loss, logits = model(**batch)
            losses[i] = loss.item()
        accelerator.print("==== Evaluation Done ====")
        model.train()
        loss = losses.mean()
        return loss

    def handle_resumption(self, data_iterator):
        '''
        Resume from a training checkpoint.
        '''
        if self.args.resume_path is not None and self.args.resume_step is not None:
            assert self.args.pretrained_path is None, (
                "You specified resume_model_path, but that will be be overridden by resume_path."
            )
            self.accelerator.print(
                f"Loading accelerator state from {self.args.resume_path} "
                f"at step {self.args.resume_step}"
            )
            self.accelerator.load_state(self.args.resume_path)
            # Skip the data loader up to the resume step
            print(f"Skipping {self.args.resume_step} batches...")
            for _ in tqdm(range(self.args.resume_step)):
                for _ in range(self.args.grad_accum_steps):
                    _ = next(data_iterator, None)
            print(f"Setting current step to {self.args.resume_step}...")
            self.cur_step = self.args.resume_step
        else:
            self.cur_step = 0

        self.accelerator.wait_for_everyone()

    def count_tokens(self, inputs: dict):
        assert self.main_input_name in inputs
        n_tokens = inputs[self.main_input_name].numel()
        n_tokens = torch.tensor(n_tokens, device=self.accelerator.device, dtype=torch.int64)
        self.n_input_tokens_seen += torch.sum(self.accelerator.gather(n_tokens)).cpu().item()

    def clip_grad_norm(self):
        # Clip the gradient, DeepSpeed will do gradient clipping internally.
        if self.accelerator.distributed_type == DistributedType.DEEPSPEED:
            grad_norm = self.model.get_global_grad_norm()
            # In some cases the grad norm may not return a float
            if hasattr(grad_norm, "item"):
                self.grad_norm = grad_norm.item()
        elif self.args.grad_clip > 0.0 and self.accelerator.sync_gradients:
            self.grad_norm: float = self.accelerator.clip_grad_norm_(
                self.model.parameters(), self.args.grad_clip
            )  # type: ignore
            self.grad_norm: float = self.grad_norm.item()  # type: ignore
        else:
            self.grad_norm = None  # type: ignore

    def process_batch(self, data_iterator):
        '''
        Process a batch of data, and update the model and LR.

        If using gradient accumulation, this will process `grad_accum_steps`
        mini-batches.
        '''
        # The average loss over the gradient accumulation steps
        batch_loss = torch.tensor(0.0).to(self.accelerator.device)
        for cur_micro_step in range(self.args.grad_accum_steps):
            # Fetch data
            start_time = time.time()
            inputs: dict | None = next(data_iterator, None)

            input_ids = inputs['input_ids']
            labels = inputs['labels']
            position_ids = None

            local_inputs = prepare_ulysses_attn_inputs(
                input_ids=input_ids,
                position_ids=position_ids,
                labels=labels,
                rank=self.accelerator.process_index,
                world_size=self.accelerator.num_processes,
                device=self.accelerator.device,
            )

            input_ids = local_inputs['input_ids']
            labels = local_inputs['labels']
            position_ids = local_inputs['position_ids']
            print(f"[rank={self.accelerator.process_index}] LMTrainer.process_batch {input_ids.shape}")

            self.data_loading_time += time.time() - start_time

            # TODO: Handle data exhaustion here

            with self.accelerator.accumulate(self.model):
                self.model.train()
                start_time = time.time()
                outputs = self.model(
                    input_ids=input_ids,
                    position_ids=position_ids,
                    labels=labels,
                )  # type: ignore
                if isinstance(outputs, CausalLMOutputWithPast):
                    loss: Tensor = outputs.loss  # type: ignore
                else:
                    loss: Tensor = outputs[0]

                self.fwd_time += time.time() - start_time

                start_time = time.time()
                self.accelerator.backward(loss)
                self.bwd_time += time.time() - start_time

                batch_loss += loss.item() / self.args.grad_accum_steps

                if self.include_num_input_tokens_seen:
                    self.count_tokens(inputs)

        self.clip_grad_norm()

        # Note that the model parameters are only updated when we are
        # at the end of an accumulation cycle.
        self.optimizer.step()
        self.lr_scheduler.step()  # Scheduler stepping is not controlled by the accelerator.
        self.optimizer.zero_grad()

        return batch_loss

    def train_loop(
        self,
    ):
        '''
        The entire training loop is executed here.
        '''
        data_iterator = iter(self.train_loader)

        # Load training states from checkpoint
        self.handle_resumption(data_iterator)

        # For tracking the training efficiency
        self.data_loading_time = 0
        self.fwd_time = 0
        self.bwd_time = 0
        self.last_log_time = time.time()
        self.model.zero_grad()

        # At this point, `self.model` is the wrapped model.
        # `self.raw_model` is the original model.

        while self.cur_step < self.args.n_train_steps:
            # TODO: Add evaluation code
            if self.val_loader is not None and self.cur_step % self.args.eval_interval == 0:
                val_loss = self.get_validation_loss(
                    accelerator=self.accelerator,
                    model=self.model,
                    data_loader=self.val_loader,
                    n_batches=self.args.n_eval_batches,
                )
                self.accelerator.log({"loss/val": val_loss}, step=self.cur_step)

            self.cur_step += 1
            self.train_loss = self.process_batch(data_iterator)
            self.train_log()

            # Checkpointing
            if self.cur_step % self.args.save_interval == 0:
                self.save_ckpt()

    def train_log(self):
        '''
        Log to experiment tracker (e.g., tensorboard) and stdout, then
        save a training checkpoint if needed.

        Most of this function will only be run every `log_interval` steps.
        '''
        # Time of this batch
        cur_time = time.time()
        iter_time = cur_time - self.last_log_time
        self.last_log_time = cur_time

        # Logging
        if self.cur_step % self.args.log_interval == 0:

            cur_lr = self.lr_scheduler.get_last_lr()[0]
            self.accelerator.log({"train/loss": self.train_loss}, step=self.cur_step)
            self.accelerator.log({"train/learning_rate": cur_lr}, step=self.cur_step)
            self.accelerator.log({"efficiency/iter_time": iter_time}, step=self.cur_step)
            self.accelerator.log({"efficiency/data_time": self.data_loading_time}, step=self.cur_step)
            self.accelerator.log({"efficiency/fwd_time": self.fwd_time}, step=self.cur_step)
            self.accelerator.log({"efficiency/bwd_time": self.bwd_time}, step=self.cur_step)
            self.accelerator.log({"misc/cur_step": self.cur_step}, step=self.cur_step)

            if self.grad_norm is not None:
                self.accelerator.log({"train/grad_norm": self.grad_norm}, step=self.cur_step)

            if self.include_num_input_tokens_seen:
                self.accelerator.log({"train/n_input_tokens_seen": self.n_input_tokens_seen}, step=self.cur_step)

            self.data_loading_time /= self.args.log_interval
            self.fwd_time /= self.args.log_interval
            self.bwd_time /= self.args.log_interval

            log_str = f"[{self.cur_step}/{self.args.n_train_steps}] loss {self.train_loss:.4f}"
            log_str += f" | time {iter_time * 1000:.1f}ms"
            log_str += f" | t_data {self.data_loading_time * 1000:.1f}ms"
            log_str += f" | t_fwd {self.fwd_time * 1000:.1f}ms"
            log_str += f" | t_bwd {self.bwd_time * 1000:.1f}ms"
            log_str += f" | lr {cur_lr:.3e}"

            if self.grad_norm is not None:
                log_str += f" | grad_norm {self.grad_norm:.2e}"

            if self.include_num_input_tokens_seen:
                log_str += f" | tokens {self.n_input_tokens_seen:,}"

            # Get hardware usage
            mem_allocated = torch.cuda.memory_allocated()
            mem_reserved = torch.cuda.memory_reserved()
            self.accelerator.log(
                {
                    "efficiency/mem_allocated": mem_allocated,
                    "efficiency/mem_reserved": mem_reserved,
                },
                step=self.cur_step,
            )

            self.accelerator.print(log_str)

            # Reset timers
            self.data_loading_time = 0
            self.fwd_time = 0
            self.bwd_time = 0

    def save_ckpt(self):
        self.accelerator.wait_for_everyone()
        ckpt_dir = self.output_dir / f"ckpt_{self.cur_step}"
        self.accelerator.print(f"Saving checkpoint to {ckpt_dir}")
        self.accelerator.save_state(str(ckpt_dir))

    def load_pretrained_ckpt(self):
        assert self.args.pretrained_path is not None
        assert Path(self.args.pretrained_path).exists(), f"{self.args.pretrained_path} does not exist."
        print(f"Loading pretrained model from {self.args.pretrained_path}")
        state_dict = torch.load(self.args.pretrained_path)
        unexpected_keys, missing_keys = self.model.load_state_dict(state_dict, strict=False)
        # By default, the `lm_head` will be removed when saving with `Accelerator`.
        assert unexpected_keys == []
        assert missing_keys == ['model.lm_head.weight']

    def train(self):
        self.is_training = True

        total_batch_size = self.args.batch_size * self.accelerator.num_processes * self.args.grad_accum_steps
        # Training loop
        self.accelerator.print("===== Start training =====")
        self.accelerator.print(f"Grad accum: {self.args.grad_accum_steps}")
        self.accelerator.print(f"Micro batch size (per device, per forward): {self.args.batch_size}")
        self.accelerator.print(f"Total train batch size (w. parallel, distributed & accumulation): {total_batch_size}")
        self.accelerator.print(f"# train iters: {self.args.n_train_steps}")
        self.accelerator.print(f"# warmup iters: {self.args.n_warmup_steps}")
        self.accelerator.print(f"# drop iters: {self.args.n_drop_steps}")
        self.accelerator.print(f"Eval interval: {self.args.eval_interval}")
        self.accelerator.print(f"Log interval: {self.args.log_interval}")

        # TODO: Handle model loading from pre-trained checkpoint.
        # NOTE: This should not load training states (optimizer, LR scheduler...)
        if self.args.pretrained_path is not None:
            self.load_pretrained_ckpt()

        # Wrap the model with accelerator classes
        self.model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.lr_scheduler)
        if not bool(self.args.use_cp):
            self.train_loader = self.accelerator.prepare(self.train_loader)

        self.train_loop()

        self.accelerator.end_training()  # Some experiment trackers need this
        self.accelerator.print("TRAINING DONE")
