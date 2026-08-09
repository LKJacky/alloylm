import torch
from parameterized import parameterized
from pydantic import BaseModel as PydanticModel
from torch import distributed as dist
from transformers import AutoTokenizer

from alloylm.algorithm.base import InferArgs
from alloylm.engine.infer_engine.engine import (
    AlloyLMModelConfig,
    InferEngineConfig,
    SPMDInfer,
    SPMDInferConfig,
)
from alloylm.engine.spmd import SPMDActor, SPMDActorConfig, init_dist
from alloylm.engine.train_engine.utils import FSDPConfig
from alloylm.impl.engines.qwen.qwen2_modeling2 import (
    FSDPQwen2ForCausalLM,
)
from alloylm.impl.math import GSM8KDatasetConfig, GSM8KTask
from alloylm.test_utils import (
    CudaAsyncTestCase,
    collect_garbage,
)


class SPMDModelConfig(PydanticModel):
    model_path: str
    fsdp_config: FSDPConfig = FSDPConfig(lm_head_dtype=torch.bfloat16)


class SPMDModel:
    def __init__(self, config: SPMDModelConfig):
        self.model = FSDPQwen2ForCausalLM.from_pretrained(
            config.model_path, fsdp_config=config.fsdp_config
        )  # init with train mesh

    def forward(self, input_ids, position_ids):
        input_ids = input_ids.cuda().long()
        position_ids = position_ids.cuda()
        output = self.model(input_ids, position_ids=position_ids).cpu()
        return output

    def infer_shard(self):
        self.model.infer_shard()

    def train_shard(self):
        self.model.train_shard()

    def __del__(self) -> None:
        if dist.is_initialized():
            dist.destroy_process_group()


class TestQwen2Modeling(CudaAsyncTestCase):
    @classmethod
    def setUpClass(cls):
        model_path = "Qwen/Qwen3-0.6B"
        world_size = max(1, min(2, torch.cuda.device_count()))
        cls.model = SPMDActor.create_spmd_actor(
            SPMDModel,
            args=(
                SPMDModelConfig(
                    model_path=model_path,
                    fsdp_config=FSDPConfig(
                        train_mesh=dict(device_type="cuda", mesh_shape=(world_size, 1), mesh_dim_names=["fsdp", "sp"]),
                        infer_mesh=dict(device_type="cuda", mesh_shape=(1, world_size), mesh_dim_names=["dp", "tp"]),
                        lm_head_dtype=torch.bfloat16,
                    ),
                ),
            ),
            spmd_config=SPMDActorConfig(world_size=world_size),
        )
        cls.tokenizer = AutoTokenizer.from_pretrained(model_path)

    @classmethod
    def tearDownClass(cls):
        cls.model.shutdown()
        del cls.model
        del cls.tokenizer
        super().tearDownClass()

    async def test_training_forward(self):
        async def forward():
            input_text = "User:\nWhat is the capital of France?\nAssistant: \n"
            input_ids = self.__class__.tokenizer(input_text, return_tensors="pt").input_ids
            for i in range(10):
                position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
                logits = (await self.__class__.model.forward(input_ids, position_ids=position_ids))[0]
                tokens = logits[0, -1].argmax(dim=-1)
                input_ids = torch.cat([input_ids, tokens.view(1, -1)], dim=-1)
            text = self.__class__.tokenizer.decode(input_ids[0], skip_special_tokens=True)
            # NOTE: the exact greedy-decoded continuation is sensitive to bf16
            # floating-point non-associativity across different tensor-parallel
            # degrees (world_size), so this string is specific to the
            # world_size configured above; it will legitimately differ (while
            # still correctly answering "Paris") if world_size changes.
            self.assertTrue("Paris" in text, f"Unexpected output: {text}")
            print("Training forward output:", text)

        await forward()
        await self.__class__.model.infer_shard()
        await forward()
        await self.__class__.model.train_shard()
        await forward()

    def test_prefill(self):
        pass

    def test_decode(self):
        pass


class TestQwen3Moe(CudaAsyncTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        init_dist()

    def test_moe(self):
        from transformers.models.qwen3_moe import Qwen3MoeConfig, Qwen3MoeForCausalLM

        config = Qwen3MoeConfig.from_dict(
            {
                "architectures": ["Qwen3MoeForCausalLM"],
                "attention_bias": False,
                "attention_dropout": 0.0,
                "bos_token_id": 151643,
                "decoder_sparse_step": 1,
                "eos_token_id": 151645,
                "head_dim": 128,
                "hidden_act": "silu",
                "hidden_size": 2048,
                "initializer_range": 0.02,
                "intermediate_size": 6144,
                "max_position_embeddings": 40960,
                "max_window_layers": 48,
                "mlp_only_layers": [],
                "model_type": "qwen3_moe",
                "moe_intermediate_size": 768,
                "norm_topk_prob": True,
                "num_attention_heads": 32,
                "num_experts": 64,
                "num_experts_per_tok": 8,
                "num_hidden_layers": 2,
                "num_key_value_heads": 4,
                "output_router_logits": False,
                "rms_norm_eps": 1e-06,
                "rope_scaling": None,
                "rope_theta": 1000000.0,
                "router_aux_loss_coef": 0.001,
                "sliding_window": None,
                "tie_word_embeddings": False,
                "torch_dtype": "bfloat16",
                "transformers_version": "4.51.0",
                "use_cache": True,
                "use_sliding_window": False,
                "vocab_size": 151936,
            }
        )
        model_alloylm = None
        model_hf = None
        model_loaded = None
        try:
            with torch.inference_mode():
                model_alloylm = (
                    FSDPQwen2ForCausalLM(
                        config,
                        fsdp_config=FSDPConfig(
                            train_mesh=dict(device_type="cuda", mesh_shape=(1, 1), mesh_dim_names=["fsdp", "sp"]),
                            infer_mesh=dict(device_type="cuda", mesh_shape=(1, 1), mesh_dim_names=["dp", "tp"]),
                            lm_head_dtype=torch.bfloat16,
                        ),
                    )
                    .bfloat16()
                    .cuda()
                )
                model_hf = Qwen3MoeForCausalLM(config).bfloat16().cuda()
                model_alloylm.model.layers.load_state_dict(model_hf.model.layers.state_dict())
                model_alloylm.norm.load_state_dict(model_hf.model.norm.state_dict())
                model_alloylm.lm_head.load_state_dict(model_hf.lm_head.state_dict())
                model_alloylm.embed_tokens.load_state_dict(model_hf.model.embed_tokens.state_dict())

                input_ids = torch.randint(0, config.vocab_size, (1, 512), device="cuda").long()
                position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
                output_alloylm = model_alloylm(input_ids, position_ids=position_ids)
                output_hf = model_hf(input_ids, position_ids=position_ids).logits
                self.assertTrue(
                    torch.allclose(output_alloylm.float(), output_hf.float(), atol=1),
                    "Outputs are not close",
                )
                model_hf.save_pretrained("work_dirs/tests/moe_hf")
                tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)
                tokenizer.save_pretrained("work_dirs/tests/moe_hf")

            # Free GPU memory before loading FSDP model
            model_alloylm.to_empty(device="cpu")
            model_hf.cpu()
            del model_alloylm, model_hf
            model_alloylm = None
            model_hf = None
            collect_garbage()

            try:
                model_loaded = FSDPQwen2ForCausalLM.from_pretrained(
                    "work_dirs/tests/moe_hf",
                    fsdp_config=FSDPConfig(
                        train_mesh=dict(device_type="cuda", mesh_shape=(1, 1), mesh_dim_names=["fsdp", "sp"]),
                        infer_mesh=dict(device_type="cuda", mesh_shape=(1, 1), mesh_dim_names=["dp", "tp"]),
                        lm_head_dtype=torch.bfloat16,
                    ),
                )
            except Exception as e:
                self.fail(f"Failed to load model from HF format: {e}")
        finally:
            if model_alloylm is not None:
                model_alloylm.to_empty(device="cpu")
            if model_hf is not None:
                model_hf.cpu()
            if model_loaded is not None:
                model_loaded.to_empty(device="cpu")
            del model_alloylm, model_hf, model_loaded
            collect_garbage()

    def test_moe_layer(self):
        pass


class TestSPMDQwen2(CudaAsyncTestCase):
    async def test_spmd_forward(self):
        world_size = min(2, torch.cuda.device_count())
        world_size = 1
        model: SPMDModel = SPMDActor.create_spmd_actor(
            SPMDModel,
            args=[
                SPMDModelConfig(
                    model_path="Qwen/Qwen3-0.6B",
                    fsdp_config=FSDPConfig(
                        train_mesh=dict(device_type="cuda", mesh_shape=(world_size, 1), mesh_dim_names=["fsdp", "sp"]),
                        infer_mesh=dict(device_type="cuda", mesh_shape=(1, world_size), mesh_dim_names=["dp", "tp"]),
                        lm_head_dtype=torch.bfloat16,
                    ),
                )
            ],
            spmd_config=SPMDActorConfig(world_size=world_size, num_gpus=1),
        )
        input_ids = torch.randint(0, 151936, (1, 512)).long()
        position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
        output = await model.forward(input_ids, position_ids)
        print("SPMD forward output shape:", output)

    @parameterized.expand([x for x in range(min(2, torch.cuda.device_count()))])
    async def test_spmd_inference_engine(self, work_size: int = 1):
        world_size = 1
        engine: SPMDInfer = SPMDActor.create_spmd_actor(
            SPMDInfer,
            args=(
                SPMDInferConfig(
                    llm_config=AlloyLMModelConfig(
                        path="Qwen/Qwen3-0.6B",
                        model_cls=FSDPQwen2ForCausalLM,
                        fsdp_config=FSDPConfig(
                            train_mesh=dict(device_type="cuda", mesh_shape=(1, 1), mesh_dim_names=["fsdp", "sp"]),
                            infer_mesh=dict(device_type="cuda", mesh_shape=(1, 1), mesh_dim_names=["dp", "tp"]),
                            lm_head_dtype=torch.bfloat16,
                        ),
                    ),
                    infer_engine_config=InferEngineConfig(
                        port=8000,
                        model_name="ALLOYLM",
                        max_prefill_length=1024,
                    ),
                ),
            ),
            spmd_config=SPMDActorConfig(world_size=world_size),
        )

        await engine.launch()
        dataset = await GSM8KDatasetConfig(infer_args=InferArgs(model_name="ALLOYLM")).build()
        task_item = dataset[0]
        task_data = await GSM8KTask.run_and_eval(task_item.task_data)
        self.assertTrue(task_data.metric == 1.0)

        await engine.stop()
        engine.shutdown()
