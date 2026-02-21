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
│   ├── gpt.py                          # GPT model implementation
│   └── modules/                        # Modular components
│       ├── attn.py                     # Attention mechanisms
│       ├── mlp.py                      # Dense MLP and MoE layers
│       ├── norm.py                     # Normalization layers
│       ├── loss.py                     # SP-aware cross entropy loss
│       └── emb.py                      # Embedding layers
│
├── training/                           # Training pipeline
│   ├── __init__.py
│   ├── config.py                       # Config dataclasses (ModelConfig, etc.)
│   ├── arguments.py                    # CLI argument definitions
│   └── trainer.py                      # Trainer and dataset init
│
├── distributed/                        # Distributed training components
│   ├── __init__.py
│   ├── parallel_state.py               # DP/SEP process group construction
│   ├── zero1/
│   │   └── distributed_optimizer.py   # ZeRO-1 implementation
│   ├── sequence_parallel/
│   │   └── ulysses.py                  # SP collectives and grad sync helpers
│   └── expert_parallel/
│       └── comm.py                     # EP all-to-all communication
│
├── utils/                              # Utility functions
│   ├── __init__.py
│   ├── model.py                        # Model utilities (param counting, etc.)
│   ├── training.py                     # Schedule helpers (get_training_info, etc.)
│   └── profile.py                      # Profiling and MFU computation
│
├── scripts/                            # Launch scripts
│   ├── README.md
│   ├── debug_gpt_0.25b/
│   │   └── pretrain.sh                 # 0.25B debug (pretrain_debug.py)
│   ├── debug_gpt_0.3b_a0.17b/
│   │   └── pretrain.sh                 # 0.3B MoE debug (pretrain_debug.py)
│   ├── example_gpt_0.25b/
│   │   └── pretrain.sh                 # 0.25B example with custom data (pretrain_example.py)
│   └── gpt_3b/
│       └── pretrain.sh
│
├── pretrain_debug.py                   # Debug entry (mock data, minimal deps)
├── pretrain_example.py                 # Example entry (custom dataset / tokenizer)
└── README.md
```

## Requirements

- Python 3.10+
- PyTorch 2.0+ with CUDA/NCCL support
- tqdm
- numpy

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
torchrun --nproc_per_node=8 pretrain_debug.py \
  --exp_name debug_test \
  --use_mock_data \
  --mock_data_num_samples 1280 \
  --total_batch_size 524288 \
  --batch_size 8 \
  --seq_len 4096 \
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

Use the example entry point and override `_init_dataset`: see `pretrain_example.py` for a subclass that uses a real dataset and tokenizer. The base implementation (mock data) lives in `training/trainer.py`; override it in your entry script or subclass `Trainer` and pass your dataset there.

## Configuration

Configuration is built from CLI arguments via `training/arguments.py` and assembled into a unified `Config` in `training/config.py`.

### Model Configuration (`ModelConfig` in `training/config.py`)

```python
@dataclass
class ModelConfig:
    block_size: int = 4096              # Maximum sequence length
    vocab_size: int = 50304             # Vocabulary size
    num_layer: int = 32                 # Number of transformer layers
    num_attention_heads: int = 128       # Number of attention heads
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

### Training Arguments (CLI → `TrainingConfig`)

Key CLI options (see `training/arguments.py` for full list):

| Option | Default | Description |
|--------|---------|-------------|
| `--exp_name` | `gpt` | Experiment name |
| `--total_batch_size` | `524288` | Global batch size in tokens |
| `--batch_size` | `8` | Micro batch size per device |
| `--seq_len` | `4096` | Sequence length |
| `--max_lr` / `--min_lr` | `4e-3` / `3e-5` | Learning rate range |
| `--weight_decay` | `0.1` | AdamW weight decay |
| `--grad_clip_value` | `1.0` | Gradient clipping |
| `--warmup_steps` | `1000` | LR warmup steps |
| `--max_epochs` | `1` | Training epochs |
| `--save_every_steps` | `5000` | Checkpoint frequency |
| `--use_compile` | flag | PyTorch 2.0 compilation |

### Parallelism Configuration

`sep_size` controls SEP group size (sequence-expert joint parallelism).

- CLI flag: `--sep_size` (default: `8` in `training/arguments.py`)
- Script env var: `SEP_SIZE` (mapped to `--sep_size`)
- Dense models (`--use_moe` disabled): SEP degenerates to pure SP.
- Constraints:
  - `WORLD_SIZE % sep_size == 0`
  - sequence length must be divisible by SEP size (`seq_len % sep_size == 0`)

Example:

```bash
torchrun --nproc_per_node=8 pretrain_debug.py \
  --batch_size 8 \
  --seq_len 4096 \
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
grad_accum_steps = total_batch_size / (batch_size × seq_len × num_dp_ranks)
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
python pretrain_debug.py \
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

Implement your dataset class and override `_init_dataset`: subclass `Trainer` in your entry script (e.g. `pretrain_example.py`) and set `self.train_dataset` to your dataset. Each item should provide tensors compatible with the trainer (e.g. contiguous token ids of length `seq_len+1` for causal LM).

### Custom Architecture

Modify components in `model/modules/`:
- `attn.py`: Implement custom attention mechanisms
- `mlp.py`: Add new feedforward architectures
- `norm.py`: Experiment with normalization strategies

### Custom Optimizer

Replace AdamW in `_init_optimizer` in `training/trainer.py` (or in a `Trainer` subclass):

```python
def _init_optimizer(self, config: Config):
    self.optimizer = YourOptimizer(
        self.raw_model.parameters(),
        lr=config.optim.max_lr,
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
2. **Tune batch size**: Maximize `--batch_size` per GPU to improve throughput
3. **Use Flash Attention**: Ensure Flash Attention is available for faster attention
4. **Gradient checkpointing**: Implement in `model/gpt.py` for larger models
5. **Mixed precision**: BFloat16 is enabled by default (better than FP16 for training)

## Common Issues

### Out of Memory
- Reduce `--batch_size` (micro batch size)
- Enable gradient checkpointing
- Use larger `grad_accum_steps` by reducing `--batch_size`

### Slow Training
- Ensure Flash Attention is installed
- Enable `--use_compile`
- Check MFU percentage (should be >30% for efficient training)
- Increase `--batch_size` to better utilize GPU

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
