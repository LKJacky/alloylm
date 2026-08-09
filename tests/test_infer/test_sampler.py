import torch

from alloylm.engine.infer_engine.scheduler import BatchSampler
from alloylm.test_utils import CudaAsyncTestCase


class TestBatchSampler(CudaAsyncTestCase):
    @torch.no_grad()
    def test_all_top_k_one_batch_is_argmax(self):
        logits = torch.tensor([[0.1, 3.0, 2.0], [5.0, 1.0, 4.0]], dtype=torch.float32)
        tokens = BatchSampler._batch_sample(
            logits,
            temperature=torch.tensor([1.0, 1.0]),
            top_k=torch.tensor([1, 1], dtype=torch.int32),
            top_p=torch.tensor([1.0, 1.0]),
            generator=torch.Generator().manual_seed(0),
        )
        self.assertTrue(torch.equal(tokens, torch.tensor([1, 0])))

    @torch.no_grad()
    def test_top_k_one_is_greedy(self):
        logits = torch.tensor(
            [
                [0.1, 3.0, 2.0],
                [0.3, 0.2, 0.5],
            ],
            dtype=torch.float32,
        )
        tokens = BatchSampler._batch_sample(
            logits,
            temperature=torch.tensor([1.0, 1.0]),
            top_k=torch.tensor([1, 1], dtype=torch.int32),
            top_p=torch.tensor([1.0, 1.0]),
            generator=torch.Generator().manual_seed(0),
        )
        self.assertEqual(tuple(tokens.flatten().tolist()), (1, 2))

    @torch.no_grad()
    def test_top_k_larger_than_vocab_is_handled(self):
        logits = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32).cuda()
        tokens = BatchSampler._batch_sample(
            logits,
            temperature=torch.tensor([1.0]).cuda(),
            top_k=torch.tensor([100], dtype=torch.int32).cuda(),
            top_p=torch.tensor([1.0]).cuda(),
            generator=torch.Generator().manual_seed(0),
        )
        self.assertEqual(tokens.shape, torch.Size([1]))
