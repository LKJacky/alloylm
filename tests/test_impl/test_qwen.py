import torch
from transformers import AutoTokenizer

from alloylm.engine.spmd import init_dist
from alloylm.impl.engines.qwen.qwen2_modeling2 import (
    FSDPConfig,
    FSDPQwen2ForCausalLM,
)
from alloylm.impl.engines.qwen.swa_cache import SwaCacheManager
from alloylm.test_utils import CudaAsyncTestCase, collect_garbage


class TestQwenModel(CudaAsyncTestCase):
    model_path = "Qwen/Qwen3-0.6B"

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        init_dist()
        world_size = 1

        cls.model = FSDPQwen2ForCausalLM.from_pretrained(
            cls.model_path,
            fsdp_config=FSDPConfig(
                train_mesh={"device_type": "cuda", "mesh_shape": (world_size, 1), "mesh_dim_names": ["fsdp", "sp"]},
                infer_mesh={"device_type": "cuda", "mesh_shape": (1, world_size), "mesh_dim_names": ["dp", "tp"]},
                lm_head_dtype=torch.bfloat16,
            ),
        )
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_path, trust_remote_code=True)

    @classmethod
    def tearDownClass(cls):
        cls.model.to_empty(device="cpu")
        del cls.model
        collect_garbage()
        super().tearDownClass()

    async def test_forward(self):
        def forward():
            input_ids = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": "What is the capital of France?"}],
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
                return_tensors="pt",
                return_dict=True,
            ).input_ids.cuda()
            with torch.no_grad():
                for _ in range(16):
                    position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
                    logits = self.model(input_ids, position_ids=position_ids)
                    token = logits[:, -1].argmax(dim=-1, keepdim=True)
                    input_ids = torch.cat([input_ids, token], dim=-1)
                    if token.item() == self.tokenizer.eos_token_id:
                        break
            text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
            self.assertTrue("Paris" in text, f"Unexpected output: {text}")
            return text

        self.model.eval()
        train_output = forward()
        self.model.infer_shard()
        try:
            infer_output = forward()
        finally:
            self.model.train_shard()
        restored_train_output = forward()

        print("Training forward output:", train_output)
        print("Inference forward output:", infer_output)
        print("Restored training forward output:", restored_train_output)

    async def test_prefill(self):
        input_ids = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": "What is the capital of France?"}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_tensors="pt",
            return_dict=True,
        ).input_ids.cuda()
        position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)

        self.model.eval()
        with torch.no_grad():
            expected_logits = self.model(input_ids, position_ids=position_ids)[:, -1]

        self.model.apply_swa_config()
        cache = SwaCacheManager(
            num_layers=self.model.config.num_hidden_layers,
            num_head=self.model.config.num_key_value_heads,
            head_dim=self.model.config.head_dim,
            window_size=self.model.config.ws,
            block_size=256,
            memory_usage=0.01,
            device="cuda",
            use_cuda_graph=False,
        )
        session = None
        self.model.infer_shard(max_prefill_length=input_ids.shape[1])
        try:
            session = cache.create_device_session(0)
            session.append_input_tokens(input_ids)
            self.assertTrue(cache.allocate_cache(session))

            prefill_logits = self.model.prefill([session], cache)

            self.assertEqual(prefill_logits.shape, expected_logits.shape)
            self.assertTrue(torch.isfinite(prefill_logits).all())
            self.assertEqual(
                prefill_logits.argmax(dim=-1).item(),
                expected_logits.argmax(dim=-1).item(),
            )
            self.assertTrue(
                torch.allclose(prefill_logits.float(), expected_logits.float(), atol=0.5, rtol=0.02),
                "Prefill logits differ from the normal forward pass",
            )
            self.assertEqual(session.forwarded_tokens, input_ids.flatten().tolist())
            self.assertEqual(session.tokens, [])
        finally:
            if session is not None:
                cache.reset([session])
            cache.close()
            del cache
            self.model.train_shard()
            collect_garbage()

    async def test_decoding(self):
        input_ids = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": "What is the capital of France?"}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_tensors="pt",
            return_dict=True,
        ).input_ids.cuda()

        self.model.eval()
        self.model.apply_swa_config()
        cache = SwaCacheManager(
            num_layers=self.model.config.num_hidden_layers,
            num_head=self.model.config.num_key_value_heads,
            head_dim=self.model.config.head_dim,
            window_size=self.model.config.ws,
            block_size=256,
            memory_usage=0.01,
            device="cuda",
            use_cuda_graph=False,
        )
        session = None
        self.model.infer_shard(max_prefill_length=input_ids.shape[1])
        try:
            session = cache.create_device_session(0)
            session.append_input_tokens(input_ids)
            self.assertTrue(cache.allocate_cache(session))
            cached_logits = self.model.prefill([session], cache)

            full_input_ids = input_ids
            for _ in range(10):
                next_token = cached_logits.argmax(dim=-1, keepdim=True)
                full_input_ids = torch.cat([full_input_ids, next_token], dim=-1)

                session.append_input_tokens(next_token)
                self.assertTrue(cache.allocate_cache(session))
                decode_logits = self.model.decode([session], cache)

                position_ids = torch.arange(full_input_ids.shape[1], device="cuda").unsqueeze(0)
                with torch.no_grad():
                    expected_logits = self.model(full_input_ids, position_ids=position_ids)[:, -1]

                self.assertEqual(decode_logits.shape, expected_logits.shape)
                self.assertTrue(torch.isfinite(decode_logits).all())
                self.assertEqual(
                    decode_logits.argmax(dim=-1).item(),
                    expected_logits.argmax(dim=-1).item(),
                )
                max_logit_diff = (decode_logits.float() - expected_logits.float()).abs().max().item()
                self.assertTrue(
                    torch.allclose(decode_logits.float(), expected_logits.float(), atol=1.0, rtol=0.02),
                    f"Decode logits differ from the normal forward pass; max difference: {max_logit_diff:.4f}",
                )
                cached_logits = decode_logits

            self.assertEqual(session.forwarded_tokens, full_input_ids.flatten().tolist())
            self.assertEqual(session.tokens, [])
            output_text = self.tokenizer.decode(full_input_ids[0], skip_special_tokens=True)
            self.assertTrue("Paris" in output_text, f"Unexpected output: {output_text}")
            print("Decoding output:", output_text)
        finally:
            if session is not None:
                cache.reset([session])
            cache.close()
            del cache
            self.model.train_shard()
            collect_garbage()
