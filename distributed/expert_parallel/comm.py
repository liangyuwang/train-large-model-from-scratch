import torch
import torch._dynamo
import torch.distributed as dist

class EPAllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden_states, input_splits, output_splits, ep_group):
        ctx.input_splits = input_splits
        ctx.output_splits = output_splits
        ctx.ep_group = ep_group
        
        out_tensor = torch.empty(
            (sum(output_splits), hidden_states.size(1)),
            dtype=hidden_states.dtype,
            device=hidden_states.device
        ).contiguous()
        
        dist.all_to_all_single(
            out_tensor, hidden_states.contiguous(),
            output_split_sizes=output_splits,
            input_split_sizes=input_splits,
            group=ep_group
        )
        return out_tensor

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = torch.empty(
            (sum(ctx.input_splits), grad_output.size(1)),
            dtype=grad_output.dtype,
            device=grad_output.device
        ).contiguous()
        
        dist.all_to_all_single(
            grad_input, grad_output.contiguous(),
            output_split_sizes=ctx.input_splits,
            input_split_sizes=ctx.output_splits,
            group=ctx.ep_group
        )
        return grad_input, None, None, None

@torch._dynamo.disable  # torch compile may remove .contiguous()
def ep_all_to_all(hidden_states, input_splits, output_splits, ep_group):
    return EPAllToAll.apply(hidden_states, input_splits, output_splits, ep_group)