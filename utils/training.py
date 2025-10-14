import math
import argparse

def get_training_args():
    """
    Based on TrainerConfig and GPTConfig
    """
    parser = argparse.ArgumentParser(description="Training Configuration")
    # Training hyperparameters
    parser.add_argument("--exp_name", type=str, default="gpt", help="Experiment name")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed for reproducibility")
    parser.add_argument("--log_dir", type=str, default="./log/", help="Directory for logging")
    parser.add_argument("--dataset_path", type=str, default="../data/fineweb-edu-sample-10BT/", help="Path to the dataset")
    parser.add_argument("--use_mock_data", action="store_true", help="Use mock data for debugging")
    parser.add_argument("--mock_data_num_samples", type=int, default=128, help="Number of samples in mock data")
    parser.add_argument("--tokenizer_name", type=str, default="gpt2", help="Tokenizer name")
    parser.add_argument("--total_batch_size", type=int, default=524288, help="Total batch size in number of tokens")
    parser.add_argument("--B", type=int, default=8, help="Micro batch size per device")
    parser.add_argument("--T", type=int, default=4096, help="Sequence length")
    parser.add_argument("--shift", type=int, default=1, help="Shift for next-token prediction")
    parser.add_argument("--use_muon", action="store_true", help="Use Muon optimizer for Attention and MLP layers")
    parser.add_argument("--max_lr", type=float, default=4e-3, help="Maximum learning rate")
    parser.add_argument("--min_lr", type=float, default=3e-5, help="Minimum learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay for optimizer")
    parser.add_argument("--grad_clip_value", type=float, default=1.0, help="Gradient clipping value")
    parser.add_argument("--warmup_steps", type=int, default=1000, help="Number of warmup steps")
    parser.add_argument("--max_steps", type=int, default=None, help="Maximum number of training steps")
    parser.add_argument("--max_epochs", type=int, default=None, help="Maximum number of epochs")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--do_val", action="store_true", help="Enable validation")
    parser.add_argument("--val_every_steps", type=int, default=250, help="Validation frequency in steps")
    parser.add_argument("--do_inference", action="store_true", help="Enable inference")
    parser.add_argument("--split_rate", type=float, default=0.99, help="Train/validation split rate")
    parser.add_argument("--do_save", action="store_true", help="Enable checkpoint saving")
    parser.add_argument("--save_every_steps", type=int, default=5000, help="Checkpoint saving frequency in steps")
    parser.add_argument("--shift_every_steps", type=int, default=None, help="Steps to shift for multi-token prediction")
    parser.add_argument("--use_compile", action="store_true", help="Use torch.compile for optimization")
    parser.add_argument("--use_profiler", action="store_true", help="Enable profiler")
    parser.add_argument("--steps_to_profile", type=int, nargs='+', default=[15, 20], help="Steps to profile")
    # Model hyperparameters
    parser.add_argument("--block_size", type=int, default=4096, help="Context length")
    parser.add_argument("--vocab_size", type=int, default=50304, help="Vocabulary size")
    parser.add_argument("--max_vocab_size", type=int, default=50257, help="Maximum vocabulary size")
    parser.add_argument("--num_layer", type=int, default=32, help="Number of transformer layers")
    parser.add_argument("--num_attention_heads", type=int, default=32, help="Number of attention heads")
    parser.add_argument("--num_key_value_heads", type=int, default=32, help="Number of attention heads")
    parser.add_argument("--hidden_size", type=int, default=1024, help="Hidden size of the model")
    parser.add_argument("--intermediate_size", type=int, default=4096, help="Intermediate size of the model")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate")
    parser.add_argument("--tied_lm_head", action="store_true", help="Tie the weights of the LM head and embedding layer")
    parser.add_argument("--use_moe", action="store_true", help="Using MoE")
    parser.add_argument("--num_experts", type=int, default=128, help="Number of experts in MoE")
    parser.add_argument("--num_experts_per_tok", type=int, default=8, help="Top-k experts to use in MoE")
    parser.add_argument("--moe_intermediate_size", type=int, default=256, help="Intermediate size for MoE layers")
    args = parser.parse_args()
    return args

def get_training_info(
    num_samples,
    tokens_per_sample,
    global_token_batch_size, 
    samples_per_device_per_step, 
    num_devices,
    max_steps=None,
    max_epochs=None,
):
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