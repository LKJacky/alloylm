import json
import os
import time
from contextlib import contextmanager
from typing import cast

import torch
from pydantic import BaseModel as PydanticModel
from pydantic import ConfigDict
from safetensors import safe_open
from torch import Tensor
from torch import distributed as dist
from torch.distributed._tensor import DTensor
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.nn.utils.clip_grad import _no_grad
from torch.utils._foreach_utils import (
    _device_has_foreach_support,
    _group_tensors_by_device_and_dtype,
    _has_foreach_support,
)

from alloylm.utils import get_logger

logger = get_logger()

# for trainer pipeline


@contextmanager
def profile_time_and_memory(desc):
    start_t = time.time()
    torch.cuda.reset_peak_memory_stats()

    yield

    max_memory = torch.cuda.max_memory_allocated()
    cost_time = time.time() - start_t

    get_logger().success(f"{desc} Elapsed time {cost_time:.2f} seconds, peak gpu memory {max_memory / 1024**3:.1f}G")


# for checkpoint


@torch.no_grad()
def lazy_init_fn(module, module2name, checkpoint_loader, enable_fp8=False, ep_mesh=None):
    device = torch.cuda.current_device()

    module_name = module2name[module]

    # if ".mlp.experts" in module_name:
    #     params = {}
    #     ep_rank = ep_mesh.get_local_rank()
    #     ep_size = ep_mesh.size()
    #     for name, _ in module.named_parameters(recurse=False):
    #         if "w1w3" in f"{module_name}.{name}" or "w2" in f"{module_name}.{name}":
    #             assert "weight" in f"{module_name}.{name}"
    #             key = f"{module_name}.{name}"
    #             key = key.replace(".weight", "")
    #             values = checkpoint_loader.load(key)
    #             values = values.cuda()
    #             assert values is not None, key
    #             values = values.view(module.num_routed_experts, -1, values.shape[-1])
    #             div_scale = values.shape[0] // ep_size
    #             values = values[ep_rank * div_scale : (ep_rank + 1) * div_scale]
    #             values = values.transpose(1, 2)
    #             values = values.reshape(-1, values.shape[-1])
    #         else:
    #             values = checkpoint_loader.load(f"{module_name}.{name}")
    #             values = values.cuda()
    #             div_scale = values.shape[0] // ep_size
    #             values = values[ep_rank * div_scale : (ep_rank + 1) * div_scale]

    #         params[name] = values
    # else:
    params = {}
    for name, _ in module.named_parameters(recurse=False):
        key = f"{module_name}.{name}"
        if "moe_pre_layer" in key:
            key = key.replace(".moe_pre_layer", "")
        params[name] = checkpoint_loader.load(key)

    buffers = {
        name: checkpoint_loader.load(f"{module_name}.{name}")
        for name, _ in module.named_buffers(recurse=False)
        if f"{module_name}.{name}" in checkpoint_loader.weight_map
    }

    module.to_empty(device=torch.cuda.current_device(), recurse=False)

    for name, param in module.named_parameters(recurse=False):
        if param.shape == params[name].shape:
            param.data.copy_(params[name])
        else:
            logger.warning(
                f"The shape of {module_name}.{name}({param.shape}) "
                f"is inconsistent with that in the checkpoint({params[name].shape}), "
                "it is initialized to 0 by default."
            )
            param.data.zero_()

    for name, buffer in module.named_buffers(recurse=False):
        if name in buffers:
            _buffer = buffers[name].to(device).to(buffer.dtype)

            if buffer.shape == _buffer.shape:
                buffer.data.copy_(_buffer)
            else:
                logger.warning(
                    f"The shape of {module_name}.{name}({buffer.shape}) "
                    f"is inconsistent with that in the checkpoint({_buffer.shape}), "
                    "it is initialized to 0 by default."
                )
                buffer.data.zero_()


def download_model_from_hub(
    model_name_or_path: str,
    from_hub="huggingface",
    cache_dir: str | None = None,
) -> str:
    """Automatically download model from the HUB.

    Note:
        If `model_name_or_path` is a local path, it will return the path
        directly without downloading it again.

    Args:
        model_name_or_path (str): The model name, model path or repo id.
        config (str | None): The config path. Default is None.
        from_hub (str): The model hosting hub, modelscope, or huggingface.
            Default is huggingface.
        cache_dir (str | None):
            The save path when downloading the model. If it is None, it
            will be stored in the default location of the HUB. For
            Huggingface, it's ~/.cache/huggingface/hub, for ModelScope,
            it's ~/.cache/modelscope/hub.
    Returns:
        str: The local path of the model.
    """
    if os.path.isdir(model_name_or_path):
        model_path = model_name_or_path
    elif from_hub == "huggingface":
        from huggingface_hub import snapshot_download

        model_path = snapshot_download(repo_id=model_name_or_path, cache_dir=cache_dir)
    else:
        # TODO support openxlab
        raise NotImplementedError(
            f"The model does not support downloading from {from_hub}, it only supports `huggingface` and `modelscope`."
        )

    return model_path


class HFCheckpointLoader:
    def __init__(self, model_path, cache_dir=None, from_hub="huggingface"):
        self.model_path = download_model_from_hub(model_path, from_hub, cache_dir)

        if "model.safetensors.index.json" in os.listdir(self.model_path):
            index_json = os.path.join(self.model_path, "model.safetensors.index.json")
            with open(index_json) as f:
                self.weight_map = json.load(f)["weight_map"]
            self.use_safetensors = True
        elif "model.bin.index.json" in os.listdir(self.model_path):
            index_json = os.path.join(self.model_path, "model.bin.index.json")
            with open(index_json) as f:
                self.weight_map = json.load(f)["weight_map"]
            self.use_safetensors = False
        elif "model.safetensors" in os.listdir(self.model_path):
            with safe_open(os.path.join(self.model_path, "model.safetensors"), framework="pt") as f:
                self.weight_map = {k: "model.safetensors" for k in f.keys()}  # noqa: SIM118
            self.use_safetensors = True
        else:
            raise FileNotFoundError

        self.current_file = None
        self.buffer = None

    def load(self, key):
        if key not in self.weight_map:
            raise KeyError(f"{key} not found in checkpoint.")
            logger.warning(f"{key} not in checkpoint.")
            return

        _file = self.weight_map[key]

        if self.use_safetensors:
            if self.current_file is None:
                self.buffer = safe_open(os.path.join(self.model_path, _file), framework="pt")
                self.current_file = _file

            if _file != self.current_file:
                self.buffer = safe_open(os.path.join(self.model_path, _file), framework="pt")
                self.current_file = _file
            weight = self.buffer.get_tensor(key)

        else:
            if self.current_file is None:
                self.buffer = torch.load(os.path.join(self.model_path, _file))
                self.current_file = _file

            if _file != self.current_file:
                self.buffer = torch.load(os.path.join(self.model_path, _file))

            weight = self.buffer[key]

        return weight


# others


def _group_tensors_by_mesh(
    tensors: list[Tensor],
) -> dict[DeviceMesh | None, list[Tensor]]:
    ret: dict[DeviceMesh | None, list[Tensor]] = {}
    for tensor in tensors:
        if isinstance(tensor, DTensor):
            device_mesh = cast(DTensor, tensor).device_mesh
        else:
            device_mesh = None
        if device_mesh in ret:
            ret[device_mesh].append(tensor)
        else:
            ret[device_mesh] = [tensor]
    return ret


class FSDPConfig(PydanticModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    # mesh
    train_mesh: DeviceMesh | dict | None = None  # support 1d mesh: 'fsdp' + "sp"
    infer_mesh: DeviceMesh | dict | None = None  # support 2d mesh: 'dp + tp'

    # for fsdp
    torch_compile: bool = False
    reshard_after_forward: bool = True
    recompute_ratio: float = 1.0

    # dtype for fsdp
    param_dtype: torch.dtype = torch.bfloat16
    reduce_dtype: torch.dtype = torch.bfloat16
    lm_head_dtype: torch.dtype = torch.float32
    shard_dtype: torch.dtype = torch.float32

    def init_device_mesh(self):
        assert self.train_mesh is not None and self.infer_mesh is not None, "train_mesh and infer_mesh must be set."
        if not isinstance(self.train_mesh, DeviceMesh):
            self.train_mesh = init_device_mesh(**self.train_mesh)
        if not isinstance(self.infer_mesh, DeviceMesh):
            self.infer_mesh = init_device_mesh(**self.infer_mesh)


DEFAULT_FSDP_CONFIG = FSDPConfig()

# for training


def pad_to_multiple_of(sequence, padding_value, multiple_of, dim=-1):
    length = sequence.shape[dim]
    if length % multiple_of == 0:
        return sequence

    pad_num = multiple_of - (length % multiple_of)
    pad_shape = (
        (*sequence.shape[:dim], pad_num, *sequence.shape[dim + 1 :]) if dim != -1 else (*sequence.shape[:dim], pad_num)
    )
    pad = torch.full(pad_shape, padding_value, dtype=sequence.dtype, device=sequence.device)
    sequence = torch.cat([sequence, pad], dim=dim)
    return sequence


def split_for_sequence_parallel(input, dim: int, sp_mesh):
    """Splits the input tensor along a given dimension for sequence parallel.

    Args:
        input: The input tensor to be split.
        dim: The dimension along which the tensor should be split.
        sp_group: The sequence parallel process group.

    Returns:
        The split tensor corresponding to the current rank's chunk.
    """
    sp_group = sp_mesh.get_group()
    sp_size = sp_mesh.size()
    if sp_size == 1:
        return input

    rank = dist.get_rank(sp_group)
    dim_size = input.size(dim)
    assert dim_size % sp_size == 0, (
        f"The dimension to split ({dim_size}) is not a multiple of sp size ({sp_size}), cannot split tensor evenly"
    )

    tensor_list = torch.split(input, dim_size // sp_size, dim=dim)
    output = tensor_list[rank].contiguous()

    return output


@_no_grad
def clip_grad_norm_(
    parameters,
    max_norm: float,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
    foreach=None,
) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    grads = [p.grad for p in parameters if p.grad is not None]
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if len(grads) == 0:
        return torch.tensor(0.0)
    first_device = grads[0].device

    grouped_grads: dict[tuple[torch.device, torch.dtype], tuple[list[list[Tensor]], list[int]]] = (
        _group_tensors_by_device_and_dtype([grads])
    )  # type: ignore[assignment]

    norms: list[Tensor] = []
    for (device, _), ([device_grads], _) in grouped_grads.items():  # type: ignore[assignment]
        if (foreach is None and _has_foreach_support(device_grads, device)) or (
            foreach and _device_has_foreach_support(device)
        ):
            # If model has applied different parallel strategies for its modules, e.g.
            # VLM apply pure FSDP for vision part and FSDP+TP for language part, grads
            # will have different device meshes. However, for_each operations doesn't
            # support multiple meshes as of torch 2.5.1. We group them manually.
            mesh_grouped_grads = _group_tensors_by_mesh(device_grads)
            for mesh_grads in mesh_grouped_grads.values():
                norms.extend(torch._foreach_norm(mesh_grads, norm_type))
        elif foreach:
            raise RuntimeError(f"foreach=True was passed, but can't use the foreach API on {device.type} tensors")
        else:
            norms.extend([torch.linalg.vector_norm(g, norm_type) for g in device_grads])

    # torch.stack doesn't support DTensors with different device meshes as of
    # torch 2.5.1. we manually group tensors by meshes, calculate mesh-wise norms
    # and then do reduction
    total_norms: list[Tensor] = []
    mesh_grouped_norms = _group_tensors_by_mesh(norms)
    for mesh_norms in mesh_grouped_norms.values():
        total_norm = torch.linalg.vector_norm(
            torch.stack([norm.to(first_device) for norm in mesh_norms]),
            norm_type,
        )
        if isinstance(total_norm, DTensor):
            total_norm = total_norm.full_tensor()
        total_norms.append(total_norm)
    if len(total_norms) == 1:
        total_norm = total_norms[0]
    else:
        total_norm = torch.linalg.vector_norm(torch.stack(total_norms), norm_type)

    if error_if_nonfinite and torch.logical_or(total_norm.isnan(), total_norm.isinf()):
        raise RuntimeError(
            f"The total norm of order {norm_type} for gradients from "
            "`parameters` is non-finite, so it cannot be clipped. To disable "
            "this error and scale the gradients by the non-finite norm anyway, "
            "set `error_if_nonfinite=False`"
        )
    clip_coef = max_norm / (total_norm + 1e-6)
    # Note: multiplying by the clamped coef is redundant when the coef is clamped to 1, but doing so
    # avoids a `if clip_coef < 1:` conditional which can require a CPU <=> device synchronization
    # when the gradients do not reside in CPU memory.
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
    for (device, _), ([device_grads], _) in grouped_grads.items():  # type: ignore[assignment]
        if (foreach is None and _has_foreach_support(device_grads, device)) or (
            foreach and _device_has_foreach_support(device)
        ):
            # If model has applied different parallel strategies for its modules, e.g.
            # VLM apply pure FSDP for vision part and FSDP+TP for language part, grads
            # will have different device meshes. However, for_each operations doesn't
            # support multiple meshes as of torch 2.5.1. We group them manually.
            mesh_grouped_grads = _group_tensors_by_mesh(device_grads)
            for mesh_grads in mesh_grouped_grads.values():
                torch._foreach_mul_(mesh_grads, clip_coef_clamped.to(device))
        elif foreach:
            raise RuntimeError(f"foreach=True was passed, but can't use the foreach API on {device.type} tensors")
        else:
            clip_coef_clamped_device = clip_coef_clamped.to(device)
            for g in device_grads:
                g.mul_(clip_coef_clamped_device.to(g.dtype))

    return total_norm
