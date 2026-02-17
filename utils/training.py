import math
import argparse
import numpy as np
import random
import os
import torch
from typing import Any
from packaging import version

def get_training_info(
    num_samples: int,
    tokens_per_sample: int,
    global_token_batch_size: int, 
    samples_per_device_per_step: int, 
    num_devices: int,
    max_steps=None,
    max_epochs=None,
) -> dict[str, Any]:
    """
    Calculate training hyperparameters based on the dataset and hardware configuration and verify consistency if both max_steps and max_epochs are provided.

    Args:
        num_samples (int): Total number of samples in the dataset.
        tokens_per_sample (int): Number of tokens in each sample.
        global_token_batch_size (int): Total number of tokens processed globally in each batch.
        samples_per_device_per_step (int): Number of samples processed per device in each training step.
        num_devices (int): Number of devices used for training.
        max_steps (int, optional): Maximum number of training steps.
        max_epochs (int, optional): Maximum number of epochs to train.

    Returns:
        dict: A dictionary containing the computed training parameters.

    Raises:
        ValueError: If neither max_steps nor max_epochs is provided, or if both are provided but inconsistent.
    """
    if max_steps is None and max_epochs is None:
        raise ValueError("At least one of max_steps or max_epochs must be provided.")

    tokens_per_device_per_step = tokens_per_sample * samples_per_device_per_step
    total_tokens_per_step = tokens_per_device_per_step * num_devices
    grad_accum_steps = int(global_token_batch_size / total_tokens_per_step)
    total_tokens_in_dataset = num_samples * tokens_per_sample

    if max_steps is not None and max_epochs is not None:
        calculated_max_steps = int((max_epochs * total_tokens_in_dataset) / global_token_batch_size)
        calculated_max_epochs = (max_steps * global_token_batch_size) / total_tokens_in_dataset
        # Check if the provided max_steps and max_epochs are consistent
        if not (calculated_max_steps == max_steps and int(calculated_max_epochs) == int(max_epochs)):
            raise ValueError(f"Inconsistent max_steps and max_epochs based on the dataset and configuration. "
                             f"Calculated max_steps from max_epochs: {calculated_max_steps}, provided max_steps: {max_steps}. "
                             f"Calculated max_epochs from max_steps: {int(calculated_max_epochs)}, provided max_epochs: {max_epochs}.")

    elif max_steps is None:
        max_steps = int((max_epochs * total_tokens_in_dataset) / global_token_batch_size)
    elif max_epochs is None:
        max_epochs = (max_steps * global_token_batch_size) / total_tokens_in_dataset

    return {
        "epochs": max_epochs,
        "max_steps": max_steps,
        "grad_accum_steps": grad_accum_steps,
        "total_tokens_per_step": total_tokens_per_step
    }

def set_seed(seed: int, deterministic: bool = False):
    """
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, `torch` and/or `tf` (if installed).

    Args:
        seed (`int`):
            The seed to set.
        deterministic (`bool`, *optional*, defaults to `False`):
            Whether to use deterministic algorithms where available. Can slow down training.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # ^^ safe to call this function even if cuda is not available
    if deterministic:
        torch.use_deterministic_algorithms(True)


def torch_version_ge(torch_version: str = "2.10") -> bool:
    v = torch.__version__.split("+")[0]
    return version.parse(v) > version.parse(torch_version)

def sm_ge(device: torch.device | None = None, sm: int = 80) -> bool:
    major, minor = torch.cuda.get_device_capability(device)
    return major >= sm/10   # SM80+  => major=8 (A100), 9 (H100), etc.
