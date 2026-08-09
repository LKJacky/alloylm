import unittest

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FSDPModule, fully_shard
from torch.distributed.tensor import DTensor, Shard, distribute_tensor

from alloylm.engine.infer_engine.utils import GatherContext
from alloylm.engine.spmd import SPMDActor, SPMDActorConfig
from alloylm.test_utils import CudaAsyncTestCase


class Actor:
    """Example SPMD actor.

    Each rank scales the input by ``(rank + 1)`` and then sum-all-reduces across
    the group, demonstrating a real SPMD collective. With ``world_size`` ranks
    every rank returns ``input * (1 + 2 + ... + world_size)``.
    """

    def __init__(self) -> None:
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

    def forward(self, input):
        x = torch.tensor(float(input)) * (self.rank + 1)
        x = x.cuda()
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        return x.item()

    def __del__(self) -> None:
        if dist.is_initialized():
            dist.destroy_process_group()


class LinearActor:
    def __init__(self, in_features: int = 8, out_features: int = 8) -> None:
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.in_features = in_features
        self.out_features = out_features
        self.device_mesh = init_device_mesh("cuda", (self.world_size,))

        self.linear = nn.Linear(self.in_features, self.out_features, bias=True)
        self.linear: FSDPModule = fully_shard(self.linear, mesh=self.device_mesh)
        self.gather_ctx = GatherContext(self.linear)

        self.layout = "fsdp"  # default layout

    def unshard_mode(self):
        """Context manager to temporarily unshard the linear module, allowing
        access to the full (replicated) weight and bias tensors."""
        if self.layout == "fsdp":
            self.gather_ctx.__enter__()
            self.layout = "unshard"
        elif self.layout == "row_tp" or self.layout == "col_tp":
            self.linear.weight.data = DTensor.from_local(
                self.linear.weight.data, self.device_mesh, placements=[Shard(1)]
            ).full_tensor()

            self.linear.bias = self.linear.bias.data * self.world_size
            self.layout = "unshard"
        else:
            assert self.layout == "unshard", f"unexpected layout {self.layout}"

    def row_tp(self):
        if self.layout in ["fsdp", "unshard", "col_tp"]:
            self.unshard_mode()
        elif self.layout == "row_tp":
            return
        self.linear.weight.data = distribute_tensor(
            self.linear.weight.data,
            self.device_mesh,
            placements=[Shard(1)],
        ).to_local()
        self.linear.bias.data /= self.world_size
        self.layout = "row_tp"

    def fsdp(self):
        if self.layout == "fsdp":
            return
        else:
            # self.unshard_mode()
            self.linear.reshard()
            self.layout = "fsdp"

    def forward(self, input):
        x = torch.as_tensor(input, dtype=torch.float32)

        if self.layout == "row_tp":
            x = torch.chunk(x, self.world_size, dim=-1)[self.rank]
            out = self.linear(x)
            dist.all_reduce(out, op=dist.ReduceOp.SUM)
        elif self.layout == "col_tp":
            pass
        elif self.layout == "fsdp" or self.layout == "unshard":
            out = self.linear(x)
        else:
            raise ValueError(f"unexpected layout {self.layout}")

        return out.tolist()

    def __del__(self) -> None:
        if dist.is_initialized():
            dist.destroy_process_group()


class MoeActor:
    """Wraps a single Qwen3 MoE layer.

    ``gate`` (router) and ``experts`` are FSDP-sharded *independently* (two
    separate FSDP units) rather than as a single combined unit, so the
    ``ep`` layout can keep ``experts`` in its native per-rank-sharded form
    while only the (tiny) router gets replicated.

    Also supports a *real* expert-parallel (``ep``) layout: each rank keeps
    (and ever only materializes) its own contiguous slice of
    ``num_experts // world_size`` experts, taken directly from FSDP's own
    per-rank shard (``DTensor.to_local()``, zero extra communication/memory)
    -- not a full replica of every expert masked down to a subset. Routing
    (top-k selection) uses the full/replicated router weights so every rank
    agrees on which experts each token is routed to; each rank then computes
    only its own experts' contribution to each token (via the group-gemm
    kernels) and the partial results are combined with an all-reduce sum --
    the same "compute-partial then all-reduce" trick ``LinearActor.row_tp``
    uses for tensor parallelism, applied along the expert dimension instead
    of the feature dimension.
    """

    def __init__(self, config) -> None:
        if not dist.is_initialized():
            dist.init_process_group("nccl")

        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.device_mesh = init_device_mesh("cuda", (self.world_size,))

        from alloylm.impl.engines.qwen.moe_layer import Qwen3MoeSparseMoeBlock

        self.moe = Qwen3MoeSparseMoeBlock(config).bfloat16().cuda()
        self.moe.gate: FSDPModule = fully_shard(self.moe.gate, mesh=self.device_mesh)
        self.moe.experts: FSDPModule = fully_shard(self.moe.experts, mesh=self.device_mesh)
        self.gather_ctx = GatherContext(self.moe)

        self.layout = "fsdp"  # default layout

    def unshard_mode(self):
        """Temporarily unshard the MoE layer, giving access to the full
        (replicated) expert and router weights."""
        if self.layout == "ep":
            self.moe.experts.disable_ep()
            self.moe.gate.reshard()
            self.layout = "fsdp"
        if self.layout == "fsdp":
            self.gather_ctx.__enter__()
            self.layout = "unshard"
        else:
            assert self.layout == "unshard", f"unexpected layout {self.layout}"

    def ep(self):
        """Switch to the real expert-parallel layout (see class docstring)."""
        if self.layout == "ep":
            return
        if self.layout == "unshard":
            # Drop the temporary full replica so ``experts`` is back in its
            # native FSDP-sharded (DTensor) form -- required by configure_ep()
            # to hand out a genuine, un-replicated per-rank expert shard.
            self.moe.gate.reshard()
            self.moe.experts.reshard()
            self.layout = "fsdp"
        assert self.layout == "fsdp", f"unexpected layout {self.layout}"

        self.moe.gate.unshard()  # router is tiny; replicate for consistent routing
        self.moe.experts.configure_ep(
            ep_rank=self.rank,
            ep_size=self.world_size,
            group=self.device_mesh.get_group(),
        )
        self.layout = "ep"

    def fsdp(self):
        if self.layout == "fsdp":
            return
        if self.layout == "ep":
            self.moe.experts.disable_ep()
            self.moe.gate.reshard()
        else:  # unshard
            self.moe.gate.reshard()
            self.moe.experts.reshard()
        self.layout = "fsdp"

    def forward(self, input):
        x = torch.as_tensor(input, dtype=torch.bfloat16, device="cuda")

        if self.layout == "ep":
            # Real data parallelism: each rank only processes its own slice
            # of the (global, replicated-on-the-driver-side) batch; results
            # are dispatched/combined across ranks by ``forward_ep`` itself.
            local_x = torch.chunk(x, self.world_size, dim=0)[self.rank]
            out = self.moe.forward_ep(local_x)
        else:
            out = self.moe(x)

        return out.float().tolist()

    def __del__(self) -> None:
        if dist.is_initialized():
            dist.destroy_process_group()


@unittest.skipIf(torch.cuda.device_count() < 2, "requires at least 2 GPUs")
class TestSPMD(CudaAsyncTestCase):
    async def test_spmd_actor(self):
        world_size = 4
        actor: Actor = SPMDActor.create_spmd_actor(
            Actor,
            spmd_config=SPMDActorConfig(world_size=world_size),
        )
        try:
            outputs = await actor.forward(1.0)
            expected = 1.0 * sum(r + 1 for r in range(world_size))  # 1+2+3+4 = 10
            self.assertTrue(all(abs(o - expected) < 1e-6 for o in outputs), outputs)
        finally:
            actor.shutdown()

    @unittest.skipIf(torch.cuda.device_count() < 2, "requires at least 2 GPUs")
    async def test_spmd_fsdp(self):
        world_size = min(2, torch.cuda.device_count())
        in_features, out_features = 64, 64
        actor: LinearActor = SPMDActor.create_spmd_actor(
            LinearActor,
            args=(in_features, out_features),
            spmd_config=SPMDActorConfig(world_size=world_size, num_gpus=1),
        )
        try:
            x = torch.randn(4, in_features)
            expected = torch.tensor((await actor.forward(x.tolist()))[0])

            for use_layout in ("unshard_mode", "row_tp"):
                await getattr(actor, use_layout)()
                outputs = await actor.forward(x.tolist())
                for out in outputs:
                    if not torch.allclose(torch.tensor(out), expected, atol=1e-5, rtol=1e-5):
                        self.assertTrue(
                            False, f"Layout {use_layout} failed: expected {expected}, got {torch.tensor(out)}"
                        )

        finally:
            actor.shutdown()

    @unittest.skip("ep is not ready yet")
    async def test_spmd_moe(self):
        from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig

        world_size = min(2, torch.cuda.device_count())
        config = Qwen3MoeConfig(
            hidden_size=256,
            moe_intermediate_size=128,
            num_experts=4,
            num_experts_per_tok=2,
            hidden_act="silu",
            norm_topk_prob=True,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
        )
        actor: MoeActor = SPMDActor.create_spmd_actor(
            MoeActor,
            args=(config,),
            spmd_config=SPMDActorConfig(world_size=world_size, num_gpus=1),
        )
        try:
            x = torch.randn(4, 8, config.hidden_size)
            expected = torch.tensor((await actor.forward(x.tolist()))[0])

            await actor.unshard_mode()
            outputs = await actor.forward(x.tolist())
            for out in outputs:
                if not torch.allclose(torch.tensor(out), expected, atol=1e-2, rtol=1e-2):
                    self.assertTrue(False, f"Unsharded layout failed: expected {expected}, got {torch.tensor(out)}")

            # "ep" is real data-parallel + expert-parallel: each rank only
            # returns its own slice of the batch, so concatenate them back
            # together (in rank order) before comparing to the full-batch
            # reference computed above.
            await actor.ep()
            outputs = await actor.forward(x.tolist())
            combined = torch.cat([torch.tensor(out) for out in outputs], dim=0)
            self.assertTrue(
                torch.allclose(combined, expected, atol=1e-2, rtol=1e-2),
                f"ep layout failed: expected {expected}, got {combined}",
            )
        finally:
            actor.shutdown()
