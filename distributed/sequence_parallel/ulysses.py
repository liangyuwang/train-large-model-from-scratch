import torch
import torch.distributed as dist

class UlyssesAllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor, sp_group):
        ctx.sp_group = sp_group
        output_tensor = torch.empty_like(input_tensor)
        dist.all_to_all_single(output_tensor, input_tensor, group=sp_group)
        return output_tensor

    @staticmethod
    def backward(ctx, grad_output):
        sp_group = ctx.sp_group
        grad_input = torch.empty_like(grad_output)
        dist.all_to_all_single(grad_input, grad_output, group=sp_group)
        return grad_input, None

def ulysses_all_to_all(input_tensor, sp_group):
    return UlyssesAllToAll.apply(input_tensor, sp_group)

# Example usage
if __name__ == "__main__":
    # Create a dummy tensor
    input_tensor = torch.randn(10, 10)
    # Create a dummy communication group
    sp_group = dist.new_group(backend='nccl', ranks=list(range(4)))
    # Forward pass
    output_tensor = ulysses_all_to_all(input_tensor, sp_group)
    # Backward pass
    grad_input = torch.randn_like(output_tensor)
    grad_output = ulysses_all_to_all(grad_input, sp_group)
    print(grad_output)