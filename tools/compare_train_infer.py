"""Compare hidden states between training and inference forward passes.

Measures the numerical gap between:
  - Training path: flash_attention_2 (packed sequences, one forward)
  - Inference path: flashinfer batch-prefill + batch-decode (teacher forcing)

Captures outputs after every submodule inside each decoder layer
(input_layernorm, self_attn, post_attention_layernorm, mlp, and the full
layer output), as well as the final RMSNorm and lm_head.

Input is a JSONL file where each line has a ``messages`` list (chat format).
The last message must have role ``assistant`` and is treated as the response
that gets decoded; everything before it is prefilled as the prompt.

Usage:
    python tools/compare_train_infer.py --model Qwen/Qwen2.5-0.5B-Instruct
    python tools/compare_train_infer.py --file path/to/msgs.jsonl --max-samples 4
"""

import argparse
import json
import os
import random

import torch
from matplotlib import pyplot as plt
from mmengine.dist import init_dist
from torch import distributed as dist
from torch import nn
from transformers import AutoTokenizer

from alloylm.impl.engines.qwen.qwen2_modeling2 import (
    FSDPQwen2ForCausalLM,
)

# Submodules hooked inside each Qwen2DecoderLayer.
LAYER_SUBMODULES = [
    ("input_layernorm", "input_ln"),
    ("self_attn", "attn"),
    ("post_attention_layernorm", "post_ln"),
    ("mlp", "ffn"),
]
GLOBAL_MODULES = ["final_norm", "lm_head"]
PER_LAYER_LABELS = [label for _, label in LAYER_SUBMODULES] + ["output"]


def init_distributed():
    if not dist.is_initialized():
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_RANK", "0")
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", str(random.randint(20000, 30000)))
        init_dist("pytorch")


def load_samples(path, max_samples=None):
    """Load messages from a JSONL file.

    Each line must have a ``messages`` field.
    """

    def compute_len_prompt(labels):
        for i, x in enumerate(labels):
            if x != -100:
                return i
        assert False, "All -100 in labels?"

    samples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if "rl_data" in item.get("others", {}):
                input_ids = item["others"]["rl_data"]["input_ids"]
                labels = item["others"]["rl_data"]["labels"]
            else:
                input_ids = item["input_ids"]
                labels = item["labels"]
            assert input_ids is not None and labels is not None
            prompt_len = compute_len_prompt(labels)

            prompt_ids = torch.tensor(input_ids[:prompt_len]).reshape(1, -1).cuda()
            response_ids = torch.tensor(input_ids[prompt_len:]).reshape(1, -1).cuda()

            samples.append((prompt_ids, response_ids))
            print(
                f"Loaded sample with prompt length {prompt_ids.shape[1]} and response length {response_ids.shape[1]}"
            )
            if max_samples and len(samples) >= max_samples:
                break
    return samples


def tokenize_prompt_response(tokenizer, messages):
    """Split messages into prompt and response token ids.

    Returns prompt_ids [1, P] and response_ids [1, R] on CUDA so that concat(prompt_ids, response_ids) == full_ids.
    """
    assert messages[-1]["role"] == "assistant", "Last message must be assistant"
    prompt_msgs = messages[:-1]

    prompt_ids = (
        torch.tensor(
            tokenizer.apply_chat_template(prompt_msgs, add_generation_prompt=True),
            dtype=torch.long,
        )
        .unsqueeze(0)
        .cuda()
    )
    full_ids = (
        torch.tensor(
            tokenizer.apply_chat_template(messages),
            dtype=torch.long,
        )
        .unsqueeze(0)
        .cuda()
    )

    prompt_len = prompt_ids.shape[1]
    assert torch.equal(full_ids[0, :prompt_len], prompt_ids[0]), (
        "Prompt tokens are not a prefix of full tokens — chat template inconsistency"
    )
    response_ids = full_ids[:, prompt_len:]
    return prompt_ids, response_ids, full_ids


# ---------------------------------------------------------------------------
# Hook helpers
# ---------------------------------------------------------------------------


def register_hooks(model):
    """Register forward hooks on every submodule of interest.

    Hooks *accumulate* tensors so that multiple forward calls (prefill + decode steps) can be concatenated afterwards.
    """
    captured: dict[str, list[torch.Tensor]] = {}
    hooks: list[torch.utils.hooks.RemovableHook] = []

    def _make_hook(key):
        def hook(_module, _input, output):
            out = output[0] if isinstance(output, tuple) else output
            captured[key].append(out.detach().clone())

        return hook

    for i, layer in enumerate(model.model.layers):
        for attr, label in LAYER_SUBMODULES:
            key = f"layer.{i}.{label}"
            captured[key] = []
            hooks.append(getattr(layer, attr).register_forward_hook(_make_hook(key)))
        key = f"layer.{i}.output"
        captured[key] = []
        hooks.append(layer.register_forward_hook(_make_hook(key)))

    captured["final_norm"] = []
    hooks.append(model.norm.register_forward_hook(_make_hook("final_norm")))
    captured["lm_head"] = []
    hooks.append(model.lm_head.register_forward_hook(_make_hook("lm_head")))

    return captured, hooks


def remove_hooks(hooks):
    for h in hooks:
        h.remove()


# ---------------------------------------------------------------------------
# Batched training forward
# ---------------------------------------------------------------------------


def batched_training_forward(model, all_full_ids):
    """Pack all samples and run one training forward pass with cu_seq_lens.

    Returns per-sample (logits, states) list.
    """
    n = len(all_full_ids)
    seq_lens = [ids.shape[1] for ids in all_full_ids]
    packed = torch.cat([ids.squeeze(0) for ids in all_full_ids]).unsqueeze(0)  # [1, total]
    pos = torch.cat([torch.arange(slen, device=packed.device) for slen in seq_lens]).unsqueeze(0)  # [1, total]

    cu = torch.zeros(n + 1, dtype=torch.int32, device=packed.device)
    for i, slen in enumerate(seq_lens):
        cu[i + 1] = cu[i] + slen
    max_len = max(seq_lens)

    captured, hooks = register_hooks(model)
    with torch.no_grad():
        logits = model(packed, pos, cu_seq_lens_q=cu, cu_seq_lens_k=cu, max_length_q=max_len, max_length_k=max_len)
    remove_hooks(hooks)

    # Split per sample — training is one forward, so captured[key] has exactly 1 tensor
    per_sample = []
    for i in range(n):
        s, e = cu[i].item(), cu[i + 1].item()
        sample_logits = logits[:, s:e]
        sample_states = {key: tensors[0][:, s:e] for key, tensors in captured.items()}
        per_sample.append((sample_logits, sample_states))
    return per_sample


# ---------------------------------------------------------------------------
# Batched inference forward
# ---------------------------------------------------------------------------


def batched_inference_forward(model, cache, all_prompt_ids, all_response_ids):
    """Batch-prefill all prompts, then batch-decode all responses step by step.

    Returns per-sample (last_logits, states) list.
    """
    n = len(all_prompt_ids)
    resp_lens = [r.shape[1] for r in all_response_ids]
    max_resp_len = max(resp_lens) if resp_lens else 0

    # Disable CUDA-graph decoding so hooks fire on every decode step.
    _orig_cuda_graph_decoding = model.cuda_graph_decoding

    def _decode_no_graph(input_ids, position_ids, attention_args, cache=None):
        return model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_args=attention_args,
            prefilling=False,
            attention_mask={"full_attention": None, "sliding_attention": None},
        )

    model.cuda_graph_decoding = _decode_no_graph

    captured, hooks = register_hooks(model)
    cache.prepare()
    model.infer_shard(max(ids.shape[1] for ids in all_prompt_ids))

    # Create all sessions and record prefill layout
    sessions = []
    prefill_ranges = []
    offset = 0
    for i in range(n):
        sess = cache.create_device_session(session_id=i)
        sess.append_input_tokens(all_prompt_ids[i])
        assert cache.allocate_cache(sess), f"OOM allocating cache for sample {i}"
        sessions.append(sess)
        plen = all_prompt_ids[i].shape[1]
        prefill_ranges.append((offset, offset + plen))
        offset += plen

    # ---- batch prefill ----
    model.prefill(sessions, cache)
    # captured[key][0] = [1, total_packed_prefill_tokens, D]

    # ---- batch decode step by step ----
    # At each step only include sessions that still have response tokens.
    decode_batch_map: list[dict[int, int]] = []  # step -> {sample_idx: pos_in_batch}
    last_logits: list[torch.Tensor | None] = [None] * n

    for t in range(max_resp_len):
        active_sessions = []
        idx_to_pos: dict[int, int] = {}
        for i in range(n):
            if t < resp_lens[i]:
                sessions[i].append_input_tokens(all_response_ids[i][:, t : t + 1])
                cache.allocate_cache(sessions[i])
                idx_to_pos[i] = len(active_sessions)
                active_sessions.append(sessions[i])
        logits = model.decode(active_sessions, cache)  # [K, vocab]
        decode_batch_map.append(idx_to_pos)
        # Record each sample's last logits
        for si, pos in idx_to_pos.items():
            if t == resp_lens[si] - 1:
                last_logits[si] = logits[pos : pos + 1]

    remove_hooks(hooks)
    cache.reset(sessions)
    cache.close()
    model.train_shard()
    model.cuda_graph_decoding = _orig_cuda_graph_decoding

    # ---- split hook outputs per sample ----
    per_sample = []
    for i in range(n):
        sample_states: dict[str, torch.Tensor] = {}
        for key, tensors in captured.items():
            parts = []

            # Prefill outputs are packed as [1, total_tokens, D]; decode outputs
            # are [K, D] (one row per active session). Normalize both to
            # [1, tokens, D] so they can be sliced and concatenated together.
            def _as_3d(t):
                return t.unsqueeze(0) if t.dim() == 2 else t

            # Prefill portion
            s, e = prefill_ranges[i]
            parts.append(_as_3d(tensors[0])[:, s:e])
            # Decode portions
            for t in range(resp_lens[i]):
                pos = decode_batch_map[t][i]
                parts.append(_as_3d(tensors[1 + t])[:, pos : pos + 1])
            sample_states[key] = torch.cat(parts, dim=1)
        per_sample.append((last_logits[i], sample_states))

    # Free raw captured tensors
    captured.clear()
    return per_sample


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def _tensor_stats(t, f):
    """Compute comparison statistics between two tensors of the same shape."""
    diff = (t - f).abs()
    return {
        "max_diff": diff.max().item(),
        "mean_diff": diff.mean().item(),
        "cosine_sim": nn.functional.cosine_similarity(t.reshape(1, -1), f.reshape(1, -1)).item(),
        "shape": list(t.shape),
    }


def compare_one(train_states, infer_states, num_layers, train_logits, infer_logits, prompt_ids, response_ids):
    """Per-module comparison for one sample."""
    module_stats: dict[str, dict] = {}
    for key in train_states:
        t = train_states[key].float().squeeze(0)
        f = infer_states[key].float().squeeze(0)
        if key == "lm_head":
            # Inference runs lm_head only on the last prefill token plus each
            # decode step, so it covers positions [prompt_len-1 : total]. Slice
            # the training lm_head to the same range before comparing.
            t = t[prompt_ids.numel() - 1 :]
        assert t.shape == f.shape, f"Shape mismatch at {key}: {t.shape} vs {f.shape}"
        module_stats[key] = _tensor_stats(t, f)

    train_last = train_logits[0, -1].float()
    infer_last = infer_logits[0].float()
    logit_diff = (train_last - infer_last).abs()

    # Per-token logprobs for response tokens (shifted by 1: logit at pos t predicts token t+1)
    train_lm = train_states["lm_head"][:, prompt_ids.numel() - 1 : -1].float()
    # Inference lm_head already starts at position prompt_len-1, so drop only its
    # trailing row (which would predict a token past the end of the response).
    infer_lm = infer_states["lm_head"][:, :-1].float()

    train_logprob = train_lm.log_softmax(dim=-1).gather(2, response_ids.unsqueeze(-1)).squeeze(-1)
    infer_logprob = infer_lm.log_softmax(dim=-1).gather(2, response_ids.unsqueeze(-1)).squeeze(-1)
    logprob_diff = (train_logprob - infer_logprob).abs()
    max_logprob_index = logprob_diff.argmax().item() + prompt_ids.numel() - 1

    # Per-token entropy from training logits
    train_log_softmax = train_lm.log_softmax(dim=-1)
    train_probs = train_log_softmax.exp()
    per_token_entropy = -(train_probs * train_log_softmax).sum(dim=-1).squeeze(0)  # [R]

    logit_stats = {
        "max_diff": logit_diff.max().item(),
        "mean_diff": logit_diff.mean().item(),
        "cosine_sim": nn.functional.cosine_similarity(train_last.unsqueeze(0), infer_last.unsqueeze(0)).item(),
        "train_token": train_last.argmax().item(),
        "infer_token": infer_last.argmax().item(),
        "max_logprob_diff": logprob_diff.max().item(),
        "mean_logprob_diff": logprob_diff.mean().item(),
        "max_logprob_diff_token": max_logprob_index,
    }
    token_stats = {
        "entropy": per_token_entropy.cpu(),
        "logprob_diff": logprob_diff.squeeze(0).cpu(),
    }
    return module_stats, logit_stats, token_stats


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

W = 96


def print_report(all_module_stats, all_logit_stats, num_layers, tokenizer):
    """Print per-sample tables and an aggregate summary."""
    n = len(all_module_stats)

    header = f"{'Module':<12}{'Shape':<25}{'Max Diff':<15}{'Mean Diff':<15}{'Cosine Sim':<15}"

    for idx in range(n):
        ms = all_module_stats[idx]
        lo = all_logit_stats[idx]
        print(f"\n{'=' * W}")
        print(f"Sample {idx}")
        print(f"{'=' * W}")

        for i in range(num_layers):
            print(f"Layer {i}:")
            print(f"  {header}")
            print(f"  {'-' * (W - 2)}")
            for label in PER_LAYER_LABELS:
                key = f"layer.{i}.{label}"
                s = ms[key]
                print(
                    f"  {label:<12}{s['shape']!s:<25}"
                    f"{s['max_diff']:<15.6e}{s['mean_diff']:<15.6e}{s['cosine_sim']:<15.10f}"
                )

        print("\nGlobal:")
        print(f"  {header}")
        print(f"  {'-' * (W - 2)}")
        for key in GLOBAL_MODULES:
            s = ms[key]
            print(
                f"  {key:<12}{s['shape']!s:<25}{s['max_diff']:<15.6e}{s['mean_diff']:<15.6e}{s['cosine_sim']:<15.10f}"
            )

        print(f"\n  Last-token logit max diff:  {lo['max_diff']:.6e}")
        print(f"  Last-token logit mean diff: {lo['mean_diff']:.6e}")
        print(f"  Last-token logit cosine:    {lo['cosine_sim']:.10f}")
        print(f"  logprob max diff:  {lo['max_logprob_diff']:.6e}")
        print(f"  logprob mean diff: {lo['mean_logprob_diff']:.6e}")
        t_tok, i_tok = lo["train_token"], lo["infer_token"]
        print(f"  Train token: {tokenizer.decode([t_tok])!r} ({t_tok})")
        print(f"  Infer token: {tokenizer.decode([i_tok])!r} ({i_tok})")
        print(f"  Match: {t_tok == i_tok}")
        print(
            f"  Max logprob diff token: {tokenizer.decode([lo['max_logprob_diff_token']])!r}"
            f" ({lo['max_logprob_diff_token']})"
        )

    # ---- Aggregate ----
    print(f"\n{'=' * W}")
    print(f"Aggregate over {n} sample(s)")
    print(f"{'=' * W}")

    print(f"\n{'Module':<12}{'Avg Max Diff':<18}{'Worst Max Diff':<18}{'Avg Cosine Sim':<18}")
    print("-" * 66)

    worst_key, worst_val = "", 0.0
    for label in PER_LAYER_LABELS:
        all_diffs, all_coses = [], []
        for s_idx in range(n):
            for i in range(num_layers):
                key = f"layer.{i}.{label}"
                all_diffs.append(all_module_stats[s_idx][key]["max_diff"])
                all_coses.append(all_module_stats[s_idx][key]["cosine_sim"])
        avg_d = sum(all_diffs) / len(all_diffs)
        w_d = max(all_diffs)
        avg_c = sum(all_coses) / len(all_coses)
        if w_d > worst_val:
            worst_val = w_d
            worst_key = label
        print(f"{label:<12}{avg_d:<18.6e}{w_d:<18.6e}{avg_c:<18.10f}")

    for key in GLOBAL_MODULES:
        diffs = [all_module_stats[s][key]["max_diff"] for s in range(n)]
        coses = [all_module_stats[s][key]["cosine_sim"] for s in range(n)]
        avg_d, w_d, avg_c = sum(diffs) / n, max(diffs), sum(coses) / n
        if w_d > worst_val:
            worst_val, worst_key = w_d, key
        print(f"{key:<12}{avg_d:<18.6e}{w_d:<18.6e}{avg_c:<18.10f}")

    logit_max = [s["max_diff"] for s in all_logit_stats]
    matches = sum(1 for s in all_logit_stats if s["train_token"] == s["infer_token"])
    print(f"\n  Worst max diff overall: {worst_val:.6e}  (module: {worst_key})")
    print(f"  Last-token logit worst max diff: {max(logit_max):.6e}")
    print(f"  Token match rate: {matches}/{n}")
    print(f"{'=' * W}\n")


def plot_entropy_vs_diff(all_token_stats, output_path):
    """Scatter plot of per-token entropy vs train-infer logprob diff."""
    all_entropy = torch.cat([ts["entropy"] for ts in all_token_stats]).numpy()
    all_diff = torch.cat([ts["logprob_diff"] for ts in all_token_stats]).numpy()

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(all_entropy, all_diff, alpha=0.15, s=4, color="steelblue")
    ax.set_xlabel("Entropy (training)")
    ax.set_ylabel("|logprob_train - logprob_infer|")
    ax.set_title(f"Entropy vs Train-Infer Logprob Diff (n={len(all_entropy)})")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved entropy-vs-diff plot to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Compare training vs inference hidden states")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", help="HuggingFace model or path")
    parser.add_argument("--file", type=str, default="tests/resource/msgs.jsonl", help="JSONL file with messages")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples to compare")
    parser.add_argument("--memory-usage", type=float, default=0.5, help="Fraction of GPU memory for KV cache")
    args = parser.parse_args()

    init_distributed()

    print(f"Loading model: {args.model}")
    model = FSDPQwen2ForCausalLM.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    samples = load_samples(args.file, args.max_samples)
    print(f"Loaded {len(samples)} sample(s) from {args.file}")

    # ---- tokenize all samples ----
    all_prompt_ids, all_response_ids, all_full_ids = [], [], []
    for idx, (prompt_ids, response_ids) in enumerate(samples):
        full_ids = torch.cat([prompt_ids, response_ids], dim=1)
        all_prompt_ids.append(prompt_ids)
        all_response_ids.append(response_ids)
        all_full_ids.append(full_ids)
        print(
            f"  [Sample {idx}]"
            f"(prompt {prompt_ids.shape[1]} + response {response_ids.shape[1]} = {full_ids.shape[1]} tokens)"
        )

    all_module_stats: list[dict[str, dict]] = []
    all_logit_stats: list[dict] = []
    all_token_stats: list[dict] = []

    model.eval()
    cache = model.create_cache(memory_usage=args.memory_usage, use_cuda_graph=False)

    # ---- batched training forward (one packed forward pass) ----
    print("\nRunning batched training forward …")
    train_results = batched_training_forward(model, all_full_ids)

    # ---- batched inference forward (batch prefill + batch decode) ----
    print("Running batched inference forward …")
    infer_results = batched_inference_forward(model, cache, all_prompt_ids, all_response_ids)

    # ---- compare per sample ----
    for idx in range(len(samples)):
        train_logits, train_states = train_results[idx]
        infer_logits, infer_states = infer_results[idx]
        module_stats, logit_stats, token_stats = compare_one(
            train_states,
            infer_states,
            model.config.num_hidden_layers,
            train_logits,
            infer_logits,
            all_prompt_ids[idx],
            all_response_ids[idx],
        )
        all_module_stats.append(module_stats)
        all_logit_stats.append(logit_stats)
        all_token_stats.append(token_stats)

    print_report(all_module_stats, all_logit_stats, model.config.num_hidden_layers, tokenizer)

    os.makedirs("work_dirs", exist_ok=True)
    basename = os.path.basename(args.file).replace(".jsonl", "")
    plot_entropy_vs_diff(all_token_stats, f"work_dirs/{basename}_entropy_vs_diff.png")

    if dist.is_initialized():
        dist.destroy_process_group()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
