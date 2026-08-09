import sys
import unittest

from alloylm.test_utils import CudaAsyncTestCase


@unittest.skip("Skip for fast development")
class TestTools(CudaAsyncTestCase):
    async def test_compare_train_infer(self):
        from tools.compare_train_infer import main

        sys.argv = [
            "compare_train_infer.py",
            "--model",
            "Qwen/Qwen2.5-0.5B-Instruct",
            "--file",
            "tests/resource/msgs.jsonl",
            "--max-samples",
            "1",
        ]
        main()
