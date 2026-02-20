import torch
import torch._dynamo
import torch.distributed as dist

class UlyssesAllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor, sp_group):
        ctx.sp_group = sp_group
        input_tensor = input_tensor.contiguous()
        output_tensor = torch.empty_like(input_tensor)
        dist.all_to_all_single(output_tensor, input_tensor, group=sp_group)
        return output_tensor

    @staticmethod
    def backward(ctx, grad_output):
        sp_group = ctx.sp_group
        grad_output = grad_output.contiguous()
        grad_input = torch.empty_like(grad_output)
        dist.all_to_all_single(grad_input, grad_output, group=sp_group)
        return grad_input, None

@torch._dynamo.disable  # torch compile may remove .contiguous()
def ulysses_all_to_all(input_tensor, sp_group):
    return UlyssesAllToAll.apply(input_tensor, sp_group)


def allreduce_non_expert_grads_across_sp(
    model,
    sp_group,
    sp_world_size: int,
    expert_local_param_suffixes: tuple[str, ...],
):
    if sp_world_size <= 1:
        return
    for name, param in model.named_parameters():
        grad = param.grad
        if grad is None:
            continue
        if name.endswith(expert_local_param_suffixes):
            continue
        dist.all_reduce(grad, op=dist.ReduceOp.SUM, group=sp_group)

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