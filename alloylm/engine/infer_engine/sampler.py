import flashinfer
import torch
from torch import Tensor
from torch import distributed as dist


class BatchSampler:
    def __init__(self, real_vocab_size=-1):
        self.real_vocab_size = real_vocab_size

        self.random_generator = torch.Generator(device="cuda")
        if dist.is_initialized():
            self.random_generator.manual_seed(dist.get_rank() * 100)
        else:
            self.random_generator.manual_seed(0)

    @torch.inference_mode()
    def batch_sample(self, batch_logits: Tensor, temperature: Tensor, top_k: Tensor, top_p: Tensor):
        sample_batch_size = 512
        tokens = []
        log_probs = []
        entropy = []
        top_k[top_k <= 0] = batch_logits.shape[-1]
        for i in range(0, batch_logits.shape[0], sample_batch_size):
            batch_tokens = self._batch_sample(
                batch_logits[i : i + sample_batch_size],
                temperature[i : i + sample_batch_size],
                top_k[i : i + sample_batch_size],
                top_p[i : i + sample_batch_size],
                generator=self.random_generator,
            ).long()
            batch_log_softmax = batch_logits[i : i + sample_batch_size].log_softmax(dim=-1)
            batch_log_probs = batch_log_softmax.gather(1, batch_tokens.unsqueeze(-1)).flatten()

            batch_log_softmax.mul_(batch_log_softmax.exp())  # to triton kernel
            lop_softmax_mul_softmax = batch_log_softmax
            batch_entropy = -lop_softmax_mul_softmax.sum(dim=-1).flatten()

            tokens.append(batch_tokens)
            log_probs.append(batch_log_probs)
            entropy.append(batch_entropy)
        tokens = torch.cat(tokens, dim=0)
        log_probs = torch.cat(log_probs, dim=0)
        entropy = torch.cat(entropy, dim=0)
        return tokens, log_probs, entropy

    @classmethod
    @torch.inference_mode()
    def _batch_sample(cls, logits: Tensor, temperature: Tensor, top_k: Tensor, top_p: Tensor, generator=None):
        """
        logits: [B,D]
        temperature: [B]
        top_k: [B]
        top_p: [B]

        return: sampled: [B]
        """

        B, D = logits.shape

        # argmax sample
        if top_k.eq(1).all():
            return torch.argmax(logits, dim=-1).flatten()

        # temperature
        logits = logits / temperature[:, None]

        # topk & topp
        return flashinfer.sampling.top_k_top_p_sampling_from_logits(logits, top_k, top_p)
