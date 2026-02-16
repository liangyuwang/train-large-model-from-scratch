# Train Large Model from Scratch

A minimal, hackable pre-training stack for GPT-style language models. This project provides a clean, production-ready foundation for training large-scale transformer models from scratch with distributed training support.

## Features

- **Modular GPT Architecture**: Flexible transformer implementation with support for:
  - Grouped Query Attention (GQA)
  - Mixture of Experts (MoE)
  - Customizable attention, MLP, and normalization layers
  - Flash Attention optimization support
  
- **Distributed Training**:
  - ZeRO-1 optimizer state partitioning for memory efficiency
  - DistributedDataParallel (DDP) for multi-GPU training
  - Sequence-Expert joint parallelism via `SEP_SIZE` / `--sep_size` (SEP)
  - Gradient accumulation for large effective batch sizes
  
- **Training Optimizations**:
  - Mixed precision training (BFloat16)
  - Gradient clipping
  - Cosine learning rate schedule with warmup
  - Automatic checkpoint resumption with full state recovery
  
- **Developer-Friendly**:
  - Comprehensive profiling utilities
  - Model FLOPs Utilization (MFU) tracking
  - Mock data mode for rapid debugging
  - Minimal dependencies

## Project Structure

```
.
├── model/                              # Model architecture
│   ├── __init__.py
│   ├── config.py                       # GPTConfig dataclass
│   ├── gpt.py                          # GPT model implementation
│   └── modules/                        # Modular components
│       ├── attn.py                     # Attention mechanisms
│       ├── mlp.py                      # Dense MLP and MoE layers
│       ├── norm.py                     # Normalization layers
│       ├── loss.py                     # SP-aware cross entropy loss
│       └── emb.py                      # Embedding layers
│
├── distributed/                       # Distributed training components
│   ├── __init__.py
│   ├── parallel_state.py              # DP/SEP process group construction
│   ├── zero1/
│   │   └── distributed_optimizer.py   # ZeRO-1 implementation
│   ├── sequence_parallel/
│   │   └── ulysses.py                 # SP collectives and grad sync helpers
│   └── expert_parallel/
│       └── comm.py                    # EP all-to-all communication
│
├── utils/                             # Utility functions
│   ├── __init__.py
│   ├── model.py                       # Model utilities (param counting, etc.)
│   ├── training.py                    # Argument parsing and schedule helpers
│   └── profile.py                     # Profiling and MFU computation
│
├── scripts/                           # Launch scripts
│   ├── README.md
│   ├── debug_gpt_0.25b/
│   │   └── pretrain.sh
│   ├── debug_gpt_0.3b_a0.17b/
│   │   └── pretrain.sh
│   └── gpt_3b/
│       └── pretrain.sh
│
├── train.py                           # Main training script
└── README.md
```

## Requirements

- Python 3.10+
- PyTorch 2.0+ with CUDA/NCCL support
- tqdm
- numpy

```
## Quick Start

### 1. Single Node Training

**Using training scripts (recommended):**

```bash
# Train 0.25B dense model (8 GPUs)
bash scripts/debug_gpt_0.25b/pretrain.sh

# Train 0.3B MoE model (8 GPUs)
bash scripts/debug_gpt_0.3b_a0.17b/pretrain.sh

# Override SEP (sequence-expert joint) parallel size
SEP_SIZE=2 bash scripts/debug_gpt_0.25b/pretrain.sh
```

**Direct command for quick testing:**

```bash
torchrun --nproc_per_node=8 train.py \
  --exp_name debug_test \
  --use_mock_data \
  --mock_data_num_samples 1280 \
  --total_batch_size 524288 \
  --B 8 \
  --T 4096 \
  --sep_size 1 \
  --max_epochs 1 \
  --debug
```

### 2. Multi-Node Training

All training scripts support multi-node training via environment variables:

```bash
# Node 0 (master, e.g. IP: 192.168.1.100)
NUM_NODES=2 NODE_RANK=0 MASTER_ADDR=192.168.1.100 \
bash scripts/debug_gpt_0.25b/pretrain.sh

# Node 1 (worker)
NUM_NODES=2 NODE_RANK=1 MASTER_ADDR=192.168.1.100 \
bash scripts/debug_gpt_0.25b/pretrain.sh
```

When running under some distributed training platforms, You do not need to specify --node_rank, --nnodes, or --master_addr. 'torchrun' automatically detects and uses these injected variables from 'env://' to set up distributed communication.

### 3. Custom Dataset

Prepare your dataset and modify the `_init_dataset` method in `train.py`:

```python
def _init_dataset(self, config: TrainerConfig):
    # Replace CustomDataset with your dataset implementation
    from your_data_module import YourDataset
    self.train_dataset = YourDataset(
        dataset_path=config.dataset_path, 
        split="train"
    )
```

## Configuration

### Model Configuration (`GPTConfig`)

```python
@dataclass
class GPTConfig:
    block_size: int = 4096              # Maximum sequence length
    vocab_size: int = 50304             # Vocabulary size
    num_layer: int = 32                 # Number of transformer layers
    num_attention_heads: int = 128      # Number of attention heads
    num_key_value_heads: int = 8        # Number of KV heads (GQA)
    hidden_size: int = 1024             # Hidden dimension
    intermediate_size: int = 4096       # FFN intermediate size
    dropout: float = 0.0                # Dropout rate
    tied_lm_head: bool = True           # Tie input/output embeddings
    
    # Mixture of Experts (optional)
    use_moe: bool = False               # Enable MoE
    num_experts: int = 128              # Total number of experts
    num_experts_per_tok: int = 8        # Active experts per token
    moe_intermediate_size: int = 256    # Expert FFN size
```

### Training Configuration (`TrainerConfig`)

Key parameters in `train.py`:

```python
@dataclass
class TrainerConfig:
    exp_name: str = "gpt"               # Experiment name
    total_batch_size: int = 524288      # Total tokens per step
    B: int = 8                          # Micro batch size per device
    T: int = 4096                       # Sequence length
    max_lr: float = 4e-3                # Maximum learning rate
    min_lr: float = 3e-5                # Minimum learning rate
    weight_decay: float = 0.1           # AdamW weight decay
    grad_clip_value: float = 1.0        # Gradient clipping threshold
    warmup_steps: int = 1000            # LR warmup steps
    max_epochs: int = 1                 # Training epochs
    save_every_steps: int = 5000        # Checkpoint frequency
    use_compile: bool = False           # PyTorch 2.0 compilation
```

### Parallelism Configuration

`sep_size` controls SEP group size (sequence-expert joint parallelism).

- CLI flag: `--sep_size` (default: `8` in `utils/training.py`)
- Script env var: `SEP_SIZE` (mapped to `--sep_size`)
- Dense models (`--use_moe` disabled): SEP degenerates to pure SP.
- Constraints:
  - `WORLD_SIZE % sep_size == 0`
  - sequence length must be divisible by SEP size (`T % sep_size == 0`)

Example:

```bash
torchrun --nproc_per_node=8 train.py \
  --B 8 \
  --T 4096 \
  --sep_size 2 \
  --max_epochs 1
```

## Training Features

### Automatic Checkpoint Resumption

The trainer automatically saves and resumes from checkpoints, preserving:
- Model weights (`*_model.pt`)
- Optimizer states (`*_opt/` directory)
- Training metadata (`*_meta.pt`): step counter, RNG state, dataloader position

Simply restart the training command to resume from the latest checkpoint.

### ZeRO-1 Optimizer

Memory-efficient optimizer state partitioning:
- Optimizer states are sharded across GPUs
- Model parameters remain replicated
- Automatic gradient synchronization and parameter broadcasting

### Gradient Accumulation

Automatically computed based on:
```
grad_accum_steps = total_batch_size / (B × T × num_gpus)
```

### Learning Rate Schedule

Implements cosine annealing with linear warmup:
1. Linear warmup: 0 → max_lr over `warmup_steps`
2. Cosine decay: max_lr → min_lr over remaining steps

### Model FLOPs Utilization (MFU)

Real-time tracking of hardware efficiency:
```
MFU = (Actual FLOPs) / (Peak Hardware FLOPs)
```

## Profiling

Enable PyTorch profiler for performance analysis:

```bash
python train.py \
  --use_profiler \
  --steps_to_profile 15 20
```

This generates a Chrome trace file at `<log_dir>/rank0_trace.json` that can be viewed in `chrome://tracing`.

## Example Model Configurations

### GPT-0.25B (12 layers)
```bash
--num_layer 12 \
--num_attention_heads 32 \
--num_key_value_heads 4 \
--hidden_size 1024 \
--intermediate_size 4096
```

### GPT-1B (24 layers)
```bash
--num_layer 24 \
--num_attention_heads 64 \
--num_key_value_heads 8 \
--hidden_size 2048 \
--intermediate_size 8192
```

### GPT-7B (32 layers)
```bash
--num_layer 32 \
--num_attention_heads 128 \
--num_key_value_heads 16 \
--hidden_size 4096 \
--intermediate_size 16384
```

## Extending the Code

### Custom Dataset

Implement your dataset class and modify `_init_dataset` in `train.py`:

```python
class YourDataset(Dataset):
    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(...),  # shape: (seq_len,)
            "labels": torch.tensor(...)      # shape: (seq_len,)
        }
```

### Custom Architecture

Modify components in `model/modules/`:
- `attn.py`: Implement custom attention mechanisms
- `mlp.py`: Add new feedforward architectures
- `norm.py`: Experiment with normalization strategies

### Custom Optimizer

Replace AdamW in `_init_optimizer`:

```python
def _init_optimizer(self, config: TrainerConfig):
    self.optimizer = YourOptimizer(
        self.raw_model.parameters(), 
        lr=config.max_lr
    )
    self.optimizer = DistributedOptimizer(
        optimizer=self.optimizer,
        process_group=self.dp_group,
    )
```

## Logging

Training logs are saved to:
```
<log_dir>/<exp_name>_<config_hash>/log.txt
```

Log format:
```
<step> train <loss>
<step> val <val_loss>
```

Example:
```
0 train 10.8234
100 train 8.4521
250 val 8.3012
```

## Performance Tips

1. **Enable compilation**: Add `--use_compile` for PyTorch 2.0+ (20-30% speedup)
2. **Tune batch size**: Maximize `B` per GPU to improve throughput
3. **Use Flash Attention**: Ensure Flash Attention is available for faster attention
4. **Gradient checkpointing**: Implement in `model/gpt.py` for larger models
5. **Mixed precision**: BFloat16 is enabled by default (better than FP16 for training)

## Common Issues

### Out of Memory
- Reduce `B` (micro batch size)
- Enable gradient checkpointing
- Use larger `grad_accum_steps` by reducing `B`

### Slow Training
- Ensure Flash Attention is installed
- Enable `--use_compile`
- Check MFU percentage (should be >30% for efficient training)
- Increase `B` to better utilize GPU

### Checkpoint Issues
- Ensure all processes have write access to `log_dir`
- Check disk space for optimizer state storage

## Citation

If you use this code in your research, please cite:

```bibtex
@software{train_large_model_from_scratch,
  title = {Train Large Model from Scratch},
  author = {Liangyu Wang},
  year = {2025},
  url = {https://github.com/liangyuwang/train-large-model-from-scratch}
}
```

## License

This project is licensed under the terms specified in the [LICENSE](LICENSE) file.

## Acknowledgments

This implementation draws inspiration from:
- [nanoGPT](https://github.com/karpathy/nanoGPT) by Andrej Karpathy
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) by NVIDIA
- [DeepSpeed](https://github.com/microsoft/DeepSpeed) ZeRO optimization

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

---

**Note**: This is a minimal training stack designed for educational purposes and rapid prototyping. For production-scale training, consider using frameworks like DeepSpeed, Megatron-LM, or Composer.
