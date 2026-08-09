# unified server
import socket

from torch import distributed as dist
from torch.distributed._composable.fsdp import FSDPModule

from alloylm.engine.train_engine.utils import get_logger
from alloylm.utils import get_free_port

logger = get_logger()
get_free_port = get_free_port


class GeneConfig:
    def __init__(
        self,
        temperature=1.0,
        top_k=40,
        top_p=1.0,
        stop_token=[],
        total_max_length=4096,
        max_entropy=100,
        release_at_once=False,
    ):
        # generate config
        self.temperature = temperature
        self.top_k = 40 if top_k is None else top_k
        self.top_p = 1.0 if top_p is None else top_p
        if stop_token is None:
            stop_token = []
        self.stop_token = set(stop_token) if isinstance(stop_token, list) else stop_token
        self.total_max_length = total_max_length if total_max_length else 4096  # deal None in apiserver
        self.max_entropy = max_entropy
        self.release_at_once = release_at_once


class GatherContext:
    def __init__(self, model):
        self.model = model
        self.post_forward_mesh_info = {}
        self.fsdp_param_offload_state = {}

    def _offload_sharded_param(self, fsdp_param):
        sharded_param_data = fsdp_param._sharded_param_data
        if sharded_param_data.is_meta or sharded_param_data.device.type == "cpu":
            return

        self.fsdp_param_offload_state[fsdp_param] = {
            "offload_to_cpu": fsdp_param.offload_to_cpu,
            "pin_memory": fsdp_param.pin_memory,
            "device": sharded_param_data.device,
        }

        sharded_param_data = sharded_param_data.cpu()
        if fsdp_param.pin_memory and not sharded_param_data.is_pinned():
            sharded_param_data = sharded_param_data.pin_memory()
        fsdp_param._sharded_param_data = sharded_param_data

        shard_dim = fsdp_param.fsdp_placement.dim
        shard_size = fsdp_param.sharded_size
        shard_length = shard_size[shard_dim] if fsdp_param.sharded_param._local_tensor.numel() > 0 else 0
        local_tensor = sharded_param_data.view(fsdp_param.padded_sharded_param_size).narrow(
            dim=shard_dim,
            start=0,
            length=shard_length,
        )
        if not local_tensor.is_contiguous():
            raise AssertionError("Expected CPU sharded local tensor to be contiguous")
        fsdp_param.sharded_param._local_tensor = local_tensor
        fsdp_param.offload_to_cpu = True

    def _restore_sharded_param(self, fsdp_param):
        state = self.fsdp_param_offload_state.pop(fsdp_param, None)
        if state is None:
            return

        sharded_param_data = fsdp_param._sharded_param_data.to(state["device"], non_blocking=True)
        fsdp_param._sharded_param_data = sharded_param_data

        shard_dim = fsdp_param.fsdp_placement.dim
        shard_size = fsdp_param.sharded_size
        shard_length = shard_size[shard_dim] if fsdp_param.sharded_param._local_tensor.numel() > 0 else 0
        local_tensor = sharded_param_data.view(fsdp_param.padded_sharded_param_size).narrow(
            dim=shard_dim,
            start=0,
            length=shard_length,
        )
        if not local_tensor.is_contiguous():
            raise AssertionError("Expected restored sharded local tensor to be contiguous")
        fsdp_param.sharded_param._local_tensor = local_tensor
        fsdp_param.offload_to_cpu = state["offload_to_cpu"]
        fsdp_param.pin_memory = state["pin_memory"]

    def unshard(self):
        for module in self.model.modules():
            if isinstance(module, FSDPModule):
                module.unshard()
                state = module._get_fsdp_state()
                if fsdp_param_group := state._fsdp_param_group:
                    self.post_forward_mesh_info[fsdp_param_group] = fsdp_param_group.post_forward_mesh_info
                    fsdp_param_group.post_forward_mesh_info = None  # disable post forward mesh info during inference

                    for fsdp_param in fsdp_param_group.fsdp_params:
                        self._offload_sharded_param(fsdp_param)
                yield module

    def reshard(self):
        dist.barrier()
        for module in self.model.modules():
            if isinstance(module, FSDPModule):
                state = module._get_fsdp_state()
                if fsdp_param_group := state._fsdp_param_group:
                    fsdp_param_group.post_forward_mesh_info = self.post_forward_mesh_info.get(fsdp_param_group, None)
                module.reshard()
                if fsdp_param_group:
                    for fsdp_param in fsdp_param_group.fsdp_params:
                        self._restore_sharded_param(fsdp_param)
                yield module

    def __enter__(self):
        for _ in self.unshard():
            pass

    def __exit__(self, exc_type, exc_value, traceback):
        for _ in self.reshard():
            pass


def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    finally:
        s.close()
    return ip


def get_current_ip() -> str:
    """Return the current machine's primary local IP address."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
