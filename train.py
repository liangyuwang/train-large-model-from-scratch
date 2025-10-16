import os
import math
import glob
from tqdm.auto import tqdm
from contextlib import contextmanager
from itertools import cycle, islice
from dataclasses import dataclass, field
import numpy as np
import time
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributed.checkpoint import state_dict_saver, state_dict_loader
from torch.distributed.checkpoint.filesystem import FileSystemWriter, FileSystemReader
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer    #TODO: remove transformers

from model import GPTConfig, GPT
from distributed import DistributedOptimizer
from utils import (
    get_training_args, 
    get_training_info,
    set_seed,
    get_model_params,
    compute_mfu_from_time,
)

@dataclass
class TrainerConfig:
    exp_name: str = "gpt"
    seed: int = 1337
    log_dir: str = "./log/"
    dataset_path: str = "../data/fineweb-edu-sample-10BT/"
    use_mock_data: bool = False
    mock_data_num_samples: int = 1280
    tokenizer_name: str = "gpt2"
    total_batch_size: int = 524288  # 2**19, ~0.5M tokens
    B: int = 8                      # micro batch size per device
    T: int = 4096                   # sequence length
    shift: int = 1                  # next-token prediction if 1
    use_muon: bool = False
    max_lr: float = 4e-3
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    grad_clip_value: float = 1.0
    warmup_steps: int = 1000    # or 2000
    max_steps: int | None = None    # ~1 epoch if dataset is 10B tokens
    max_epochs: int = 1
    debug: bool = True
    do_val: bool = False
    val_every_steps: int = 250
    do_inference: bool = True
    split_rate: float | None = None # will be set in __post_init__
    do_save: bool = True
    save_every_steps: int = 5000
    shift_every_steps: int | None = None
    use_compile: bool = False
    use_profiler: bool = False
    steps_to_profile: list[int] = field(default_factory=lambda: [15, 20])

    def __post_init__(self):
        # ensure split_rate depends on do_val if not set
        if self.split_rate is None:
            self.split_rate = 0.99 if self.do_val else 1.0


class Trainer:
    def _init_setup(self, config: TrainerConfig):
        self.rank = int(os.environ['RANK'])
        self.local_rank = int(os.environ['LOCAL_RANK'])
        self.world_size = int(os.environ['WORLD_SIZE'])
        dist.init_process_group(backend='nccl', init_method='env://')
        device = f'cuda:{self.local_rank}'
        torch.cuda.set_device(device)
        set_seed(config.seed + self.rank)
        self.master_process = self.rank == 0 # this process will do logging, checkpointing etc.
        self.dp_group = dist.new_group(backend='nccl', ranks=list(range(self.world_size)))
        self.dp_rank = dist.get_rank(self.dp_group)
        self.dp_local_rank = self.local_rank
        self.dp_world_size = dist.get_world_size(self.dp_group)
        self.dp_master_process = self.dp_rank == 0
        #TODO: get ep group

    def _init_dataset(self, config: TrainerConfig):
        if config.use_mock_data:
            from torch.utils.data import Dataset
            class MockDataset(Dataset):
                def __init__(self, length: int, seq_len: int, vocab_size: int = 50304):
                    self.length = length
                    self.seq_len = seq_len
                    self.vocab_size = vocab_size
                def __len__(self):
                    return self.length
                def __getitem__(self, idx):
                    data = torch.randint(0, self.vocab_size, (self.seq_len+1,), dtype=torch.long)
                    x = data[:self.seq_len]
                    y = data[1:self.seq_len+1]
                    return {"input_ids": x, "labels": y}
            self.train_dataset = MockDataset(config.mock_data_num_samples, config.T)
            if config.do_val:
                self.val_dataset = MockDataset(config.mock_data_num_samples // 10, config.T)
        else:
            class CustomDataset: ...
            self.train_dataset = CustomDataset(dataset_path=config.dataset_path, split="train")
            if config.do_val:
                self.val_dataset = CustomDataset(dataset_path=config.dataset_path, split="validation")
        self.train_sampler = DistributedSampler(self.train_dataset, num_replicas=self.dp_world_size, rank=self.dp_rank, shuffle=True)
        self.train_loader = DataLoader(self.train_dataset, batch_size=config.B, sampler=self.train_sampler, num_workers=0, pin_memory=True)
        if config.do_val:
            self.val_sampler = DistributedSampler(self.val_dataset, num_replicas=self.dp_world_size, rank=self.dp_rank, shuffle=False)
            self.val_loader = DataLoader(self.val_dataset, batch_size=config.B, sampler=self.val_sampler, num_workers=0, pin_memory=True)
        else:
            self.val_dataset = self.val_loader = None

    def _init_model(self, config: TrainerConfig, model_config: GPTConfig = None):
        torch.set_float32_matmul_precision('high')
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
        self.model_config = GPTConfig() if model_config is None else model_config
        model = GPT(self.model_config)
        params_config = get_model_params(self.model_config)
        if self.master_process:
            print(f"Params config: {params_config}")
        if config.use_compile and hasattr(torch, 'compile'):
            model = torch.compile(model)
        model = model.to(f'cuda:{self.dp_local_rank}')
        #TODO: Here ZeRO-1 actually only need 'reduce' not 'all-reduce' used in DDP, we can develop a custom wrapper for ZeRO-1
        self.model = DDP(model, process_group=self.dp_group, find_unused_parameters=True, gradient_as_bucket_view=True)
        self.raw_model = self.model.module

    def _init_optimizer(self, config: TrainerConfig):
        self.optimizer = torch.optim.AdamW(self.raw_model.parameters(), weight_decay=config.weight_decay)
        self.optimizer = DistributedOptimizer(
            optimizer=self.optimizer,
            process_group=self.dp_group,
        )
        self.raw_optimizer = self.optimizer.optimizer
    
    def _init_profiler(self, config: TrainerConfig):
        @contextmanager
        def dummy_record_function(name: str):
            yield
        def trace_handler(prof):
            if self.master_process:
                prof.export_chrome_trace(f"{self.log_dir}/rank{self.dp_rank}_trace.json")
        if config.use_profiler:
            assert self.config.steps_to_profile[0] >= 1, "steps_to_profile[0] should be >= 1"
            self.profiler = torch.profiler.profile(
                schedule=torch.profiler.schedule(
                    wait=self.config.steps_to_profile[0]-1,
                    warmup=1,
                    active=self.config.steps_to_profile[1]+1-self.config.steps_to_profile[0],
                    repeat=1),
                on_trace_ready=trace_handler,
                record_shapes=True,
                with_stack=True,
                with_flops=True,
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            )
        else:
            self.profiler = None
        self.profiler_record_fn = torch.profiler.record_function if config.use_profiler else dummy_record_function

    def __init__(self, config: TrainerConfig, model_config: GPTConfig = None):
        self.config = config
        self._init_setup(config)
        assert config.total_batch_size % (config.B * config.T * self.dp_world_size) == 0, "make sure total_batch_size is divisible by B * T * dp_world_size"
        if self.master_process:
            print(f"Trainer config: {config}")
            print(f"Model config: {model_config}")
        self._init_dataset(config)
        self.training_info = get_training_info(
            len(self.train_dataset), config.T, config.total_batch_size, config.B, self.dp_world_size, config.max_steps, config.max_epochs)
        if self.master_process:
            print(f"The training process will train {self.training_info['epochs']} epochs, {self.training_info['max_steps']} steps.")
            print(f"=> calculated gradient accumulation steps: {self.training_info['grad_accum_steps']}")
            print(f"=> calculated tokens per step: {self.training_info['total_tokens_per_step']}")
        self._init_model(config, model_config)
        self._init_optimizer(config)
        # create the log directory we will write checkpoints to and log to
        self.log_dir = os.path.join(
            config.log_dir,
            f"{config.exp_name}_"
            f"modelsize_{sum(p.numel() for p in self.raw_model.parameters())}_"
            f"lr{config.max_lr}_"
            f"B{config.total_batch_size}_"
            f"T{config.T}_"
            f"DP{self.dp_world_size}_"
            f"Muon{self.config.use_muon}"
        )
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_file = os.path.join(self.log_dir, f"log.txt")
        with open(self.log_file, "w") as f: # open for writing to clear the file
            pass

    def _lr_scheduler(self, it, max_steps, warmup_steps, max_lr, min_lr):
        # 1) linear warmup for warmup_iters steps
        if it < warmup_steps:
            return max_lr * (it+1) / warmup_steps
        # 2) if it > lr_decay_iters, return min learning rate
        if it > max_steps:
            return min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
        return min_lr + coeff * (max_lr - min_lr)
        
    def _one_training_micro_step(self, config: TrainerConfig, micro_step: int, data_batch: dict):
        x, y = data_batch["input_ids"], data_batch["labels"]
        x, y = x.to(f'cuda:{self.dp_local_rank}'), y.to(f'cuda:{self.dp_local_rank}')
        with self.profiler_record_fn("forward"):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                _, loss = self.model(x.reshape(x.shape[0],-1), y.reshape(y.shape[0],-1))
        loss = loss / self.training_info["grad_accum_steps"]
        with self.profiler_record_fn("backward"):
            loss.backward()
        return loss.detach()

    def _one_training_step(self, config: TrainerConfig, step: int):
        self.model.train()
        self.optimizer.zero_grad()
        loss_accum = 0.0
        for micro_step in range(self.training_info["grad_accum_steps"]):
            try:
                _, batch = next(self.train_loader_iter)
            except StopIteration:
                self.train_loader_iter = enumerate(self.train_loader)
                _, batch = next(self.train_loader_iter)
            self.model.require_backward_grad_sync = (micro_step == self.training_info["grad_accum_steps"] - 1)
            loss_accum += self._one_training_micro_step(config, micro_step, batch)
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), config.grad_clip_value)
        lr = self._lr_scheduler(step, self.training_info["max_steps"], config.warmup_steps, config.max_lr, config.min_lr)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self.optimizer.step()
        self.one_step_results["lr"] = lr
        self.one_step_results["loss"] = loss_accum
        self.one_step_results["grad_norm"] = norm
    
    def _resume_from_checkpoint(self, steps_per_epoch):
        pattern = os.path.join(self.log_dir, "*_model.pt")
        ckpts = sorted(glob.glob(pattern))
        if not ckpts:
            self.start_step = 0
            return
        ckpt_prefix = ckpts[-1].replace("_model.pt", "")
        meta_path = f"{ckpt_prefix}_meta.pt"
        meta = torch.load(meta_path, map_location=f'cuda:{self.local_rank}')
        # 1) model
        state_dict = torch.load(f"{ckpt_prefix}_model.pt", map_location=f'cuda:{self.local_rank}', weights_only=True)
        self.raw_model.load_state_dict(state_dict)
        # 2) optimizer
        opt_state_placeholder = {f"optimizer/rank{self.rank}": self.raw_optimizer.state_dict()}
        state_dict_loader.load(
            state_dict=opt_state_placeholder,
            storage_reader=FileSystemReader(f"{ckpt_prefix}_opt"),
        )
        # 3) dataset state
        sampler_state = meta.get('sampler_state', {})
        epoch = sampler_state.get('epoch', 0)
        iter_idx = sampler_state.get('iter_idx', 0)
        if hasattr(self, 'train_sampler') and self.train_loader.sampler is not None:
            self.train_loader.sampler.set_epoch(epoch)
        self.train_loader_iter = enumerate(islice(self.train_loader, iter_idx, None), start=iter_idx)
        # 4) next step 
        step = meta.get('step', None)
        self.start_step = (step + 1) if (step is not None) else 0
        if self.master_process:
            print(f"=> Resumed from {self.log_dir} | next_step={self.start_step}, "
                f"sampler_epoch={epoch}, dataloader_iter_idx={iter_idx}")
        # 5) RNG: finally load RNG state
        rng_path = f"{ckpt_prefix}_rng_rank{self.rank}.pt"
        if os.path.exists(rng_path):
            rng = torch.load(rng_path, map_location='cpu')
            torch.set_rng_state(rng['torch'].to(torch.uint8).cpu())
            torch.cuda.set_rng_state(rng['cuda'].to(torch.uint8).cpu(), self.dp_local_rank)
            np.random.set_state(rng['numpy'])
        dist.barrier()
        torch.cuda.synchronize()
    
    def train(self):
        self.results = {}
        steps_per_epoch = max(1, len(self.train_loader) // self.training_info['grad_accum_steps'])
        self.train_loader_iter = enumerate(self.train_loader)
        self._resume_from_checkpoint(steps_per_epoch)
        # training loop
        self._init_profiler(self.config)
        if self.profiler:
            self.profiler.start()
        for step in tqdm(range(self.start_step, self.training_info["max_steps"]), 
                        initial=self.start_step, total=self.training_info["max_steps"], 
                        desc="Train", disable=not self.master_process):
            self.one_step_results = {}
            t0 = time.time()
            last_step = (step == self.training_info["max_steps"] - 1)
            # 1) train
            with self.profiler_record_fn("training_step"):
                self._one_training_step(self.config, step)
            torch.cuda.synchronize()
            if self.profiler:
                self.profiler.step()
            # 2) eval
            if not self.config.debug and self.config.do_val and (step % self.config.val_every_steps == 0 or last_step):
                self.eval()
                if self.master_process:
                    tqdm.write(f"validation loss: {self.one_step_results['val_loss'].item():.4f}")
                with open(self.log_file, "a") as f:
                    f.write(f"{step} val {self.one_step_results['val_loss'].item():.4f}\n")
            # 3) save
            if not self.config.debug and step > 0 and (step % self.config.save_every_steps == 0 or last_step):
                self.save(step)
            # 4) print
            t1 = time.time()
            dt = t1 - t0 # time difference in seconds
            tokens_processed = self.config.B * self.config.T * self.training_info["grad_accum_steps"] * self.dp_world_size
            tokens_per_sec = tokens_processed / dt
            mfu, actual, peak = compute_mfu_from_time(
                self.config.B, self.config.T, self.model_config.hidden_size, 
                self.model_config.moe_intermediate_size if self.model_config.use_moe else self.model_config.intermediate_size,
                self.model_config.num_experts_per_tok if self.model_config.use_moe else 1, 
                self.model_config.num_experts if self.model_config.use_moe else 1,
                self.model_config.num_layer, dt, self.training_info["grad_accum_steps"], dtype="bf16")
            if self.master_process:
                tqdm.write(f"step {step:5d} | loss: {self.one_step_results['loss'].item():.6f} | lr {self.one_step_results['lr']:.4e} | grad norm: {self.one_step_results['grad_norm']:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f} | MFU: {mfu*100:.2f}%")
                with open(self.log_file, "a") as f:
                    f.write(f"{step} train {self.one_step_results['loss'].item():.6f}\n")
            self.results[step] = self.one_step_results
        dist.destroy_process_group()

    def eval(self):
        self.model.eval()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = self.config.B * len(self.val_loader) // (self.config.B * self.dp_world_size)
            for batch in tqdm(self.val_loader, desc="Val", disable=not self.master_process):
                x, y = batch["input_ids"], batch["labels"]
                x, y = x.to(f'cuda:{self.dp_local_rank}'), y.to(f'cuda:{self.dp_local_rank}')
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits, loss = self.model(x.reshape(x.shape[0],-1), y.reshape(y.shape[0],-1))
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()
        dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        torch.cuda.synchronize()
        self.one_step_results["val_loss"] = val_loss_accum
    
    def save(self, step: int = None):
        # optionally write model checkpoints
        checkpoint_path = os.path.join(self.log_dir, f"{step:05d}")
        steps_per_epoch = max(1, len(self.train_loader) // self.training_info['grad_accum_steps'])
        next_step = (step if step is not None else 0) + 1
        sampler_epoch_next = next_step // steps_per_epoch
        sampler_iter_idx_next = (next_step % steps_per_epoch) * self.training_info['grad_accum_steps']
        state_dict_saver.save(
            state_dict={f"optimizer/rank{self.rank}": self.raw_optimizer.state_dict()},
            storage_writer=FileSystemWriter(f"{checkpoint_path}_opt"),
        )
        rng_state = {
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state(self.dp_local_rank),
            'numpy': np.random.get_state(),
        }
        torch.save(rng_state, f"{checkpoint_path}_rng_rank{self.rank}.pt")
        if self.master_process:
            torch.save(self.raw_model.state_dict(), f"{checkpoint_path}_model.pt")
            checkpoint = {
                'trainer_config': self.config,
                'model_config': self.raw_model.config,
                'step': step,
                'this_step_results': self.one_step_results,
                'opt_part_assignment': self.optimizer.part_assignment,
                'sampler_state': {
                    'epoch': sampler_epoch_next,
                    'iter_idx': sampler_iter_idx_next,
                },
                'rng_state': rng_state,
            }
            torch.save(checkpoint, f"{checkpoint_path}_meta.pt")


def main():
    args = get_training_args()
    config = TrainerConfig()
    for k, v in vars(args).items():
        if hasattr(config, k):
            setattr(config, k, v)
    model_config = GPTConfig()
    for k, v in vars(args).items():
        if hasattr(model_config, k):
            setattr(model_config, k, v)
    trainer = Trainer(config, model_config)
    trainer.train()


if __name__ == "__main__":
    main()