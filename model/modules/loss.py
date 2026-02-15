import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from ...distributed import parallel_state


class CrossEntropyLoss(nn.Module):
    def __init__(self, ignore_index=-100):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        """
        logits: [B, T_local, Vocab_Size]
        targets: [B, T_local]
        """
        logits = logits.view(-1, logits.size(-1)) # [B * T_local, V]
        targets = targets.view(-1)                # [B * T_local]

        unreduced_loss = F.cross_entropy(
            logits, targets, 
            ignore_index=self.ignore_index, 
            reduction='none'
        )

        valid_mask = (targets != self.ignore_index).float()
        local_loss_sum = (unreduced_loss * valid_mask).sum()
        local_valid_count = valid_mask.sum()

        sp_group = parallel_state.get_sp_group()
        global_valid_count = local_valid_count.clone().detach()
        dist.all_reduce(global_valid_count, op=dist.ReduceOp.SUM, group=sp_group)
        global_valid_count = global_valid_count.clamp(min=1.0)
        loss = local_loss_sum / global_valid_count

        with torch.no_grad():
            global_loss_sum = local_loss_sum.clone().detach()
            dist.all_reduce(global_loss_sum, op=dist.ReduceOp.SUM, group=sp_group)
            global_loss = global_loss_sum / global_valid_count
            loss.logging_loss = global_loss # for logging

        return loss