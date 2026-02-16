import torch.distributed as dist

_SEP_GROUP = None
_DP_GROUP = None
_DP_SP_GROUP = None

_SEP_SIZE = None
_DP_SIZE = None
_DP_SP_SIZE = None

def initialize_model_parallel(sep_size: int):
    """
    Initialize the distributed communication grid
    Must be called after dist.init_process_group!
    """
    global _SEP_GROUP, _DP_GROUP, _DP_SP_GROUP, _SEP_SIZE, _DP_SIZE, _DP_SP_SIZE
    
    assert dist.is_initialized(), "Must initialize PyTorch distributed environment first!"
    
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    
    assert world_size % sep_size == 0, "World size must be divisible by sep_size"
    dp_size = world_size // sep_size
    
    _SEP_SIZE = sep_size
    _DP_SIZE = dp_size
    _DP_SP_SIZE = dp_size * sep_size

    # 1. Build SEP group (continuous Rank)
    for i in range(dp_size):
        sep_ranks = list(range(i * sep_size, (i + 1) * sep_size))
        group = dist.new_group(sep_ranks)
        if rank in sep_ranks:
            _SEP_GROUP = group

    # 2. Build DP group (same offset Rank across SEP groups)
    for j in range(sep_size):
        dp_ranks = [i * sep_size + j for i in range(dp_size)]
        group = dist.new_group(dp_ranks)
        if rank in dp_ranks:
            _DP_GROUP = group

    # 3. Build DP_SP group (same offset Rank across DP groups)
    dp_sp_ranks = list(range(world_size))
    _DP_SP_GROUP = dist.new_group(dp_sp_ranks)


def get_dp_group():
    assert _DP_GROUP is not None, "DP Group is not initialized"
    return _DP_GROUP

def get_dp_world_size():
    return _DP_SIZE

def get_dp_rank():
    return dist.get_rank(group=get_dp_group())

def get_dp_global_rank(rank: int = None):
    if rank is None:
        rank = get_dp_rank()
    return dist.get_global_rank(group=get_dp_group(), rank=rank)

def get_sep_group():
    assert _SEP_GROUP is not None, "SEP Group is not initialized"
    return _SEP_GROUP

def get_sep_world_size():
    return _SEP_SIZE

def get_sep_rank():
    return dist.get_rank(group=get_sep_group())

def get_sep_global_rank(rank: int = None):
    if rank is None:
        rank = get_sep_rank()
    return dist.get_global_rank(group=get_sep_group(), rank=rank)

def get_sp_group():
    return get_sep_group()

def get_sp_world_size():
    return get_sep_world_size()

def get_sp_rank():
    return get_sep_rank()

def get_sp_global_rank(rank: int = None):
    if rank is None:
        rank = get_sp_rank()
    return dist.get_global_rank(group=get_sp_group(), rank=rank)

def get_ep_group():
    return get_sep_group()

def get_ep_world_size():
    return get_sep_world_size()

def get_ep_rank():
    return get_sep_rank()

def get_ep_global_rank(rank: int = None):
    if rank is None:
        rank = get_ep_rank()
    return dist.get_global_rank(group=get_ep_group(), rank=rank)

def get_dp_sp_group():
    assert _DP_SP_GROUP is not None, "DP_SP Group is not initialized"
    return _DP_SP_GROUP

def get_dp_sp_world_size():
    return _DP_SP_SIZE

def get_dp_sp_rank():
    return dist.get_rank(group=get_dp_sp_group())

def get_dp_sp_global_rank(rank: int = None):
    if rank is None:
        rank = get_dp_sp_rank()
    return dist.get_global_rank(group=get_dp_sp_group(), rank=rank)