from typing import Union
from dataclasses import dataclass

@dataclass
class GPTConfig:
    seed: int = 1337
    block_size: int = 4096
    vocab_size: int = 50304
    num_layer: int = 32
    num_attention_heads: int = 128
    num_key_value_heads: int = 8
    hidden_size: int = 1024
    intermediate_size: int = 4096
    dropout: float = 0.0
    init_std: float = 0.013 # same as openllama, bloom suggests sqrt(2/(NHIDDEN*5)) = 0.0098 or sqrt(2/(NHIDDEN*3)) = 0.009
    tied_lm_head: bool = True

    # MoE
    use_moe: bool = False  # if using MoE
    num_experts: int = 128
    num_experts_per_tok: int = 8  # could be a range from sparse to dense
    moe_intermediate_size: int = 256