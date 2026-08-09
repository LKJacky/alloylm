# AlloyLM

`AlloyLM` is a self-contained framework for building and testing LLM training,
inference, evaluation, and reinforcement-learning pipelines. It includes:

- An OpenAI-compatible inference API with continuous batching and session
  support.
- Distributed training and checkpointing on the same model used for inference.
- Dataset and task abstractions for concurrent evaluation.
- Asynchronous rollout and distributed RL training.
- FSDP/SPMD model execution and Qwen/Qwen-MoE implementations.

> [!WARNING]
> This project is under active development. Its APIs and configuration formats
> may change, and the training and inference paths require NVIDIA GPUs.

## Requirements

- Linux
- Python 3.12
- NVIDIA GPU(s) with a CUDA-compatible PyTorch installation
- CUDA build tools for GPU extensions such as FlashAttention

The development container is the recommended environment. It is based on a CUDA-enabled PyTorch image and installs the project dependencies.

## Installation

Install PyTorch for the CUDA version available on your machine first, then run:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m pip install flash-attn --no-build-isolation
```

## Design

```Mermaid
---
title: AlloyLM Design
---
classDiagram
    class Algo{
        - TrainInferEngine engine
    }
    class TrainInferEngine{
        - Model model
        - TrainEngine train_engine
        - InferEngine infer_engine
        + rl_step(RLInput)
    }
    class Model{
        + train_forward(TrainInput)
        + prefill(DeviceSessions,Cache)
        + decode(DeviceSessions,Cache)
        + train_shard()
        + infer_shard()
    }

    Algo --> TrainInferEngine
    TrainInferEngine --> Model

    note for Algo "Single Controller Algorithm, e.g. RL, eval"
    note for TrainInferEngine "run in SPMD mode, supports both training and inference"
    note for Model "run in FSDP mode"
```

## Evaluation

Evaluation runs against an OpenAI-compatible API. A configuration module must
export an `EvalConfig` named `config`:

```python
# eval_config.py
from alloylm.algorithm.base import InferArgs
from alloylm.algorithm.eval.base import EvalConfig
from alloylm.impl.math import GSM8KDatasetConfig

config = EvalConfig(
    datasets=[
        GSM8KDatasetConfig(
            name="gsm8k",
            split="test",
            infer_args=InferArgs(
                model_name="ALLOYLM",
                sample_args={"temperature": 0.0, "max_tokens": 2048},
            ),
        )
    ]
)
```

Run the evaluation against a server on `localhost:8000`:

```bash
python -m alloylm.algorithm.eval.run eval_config \
  --url http://127.0.0.1:8000/v1 \
  --work_dir work_dirs/eval \
  --concurrency 128
```

Use `--resume` to continue from JSONL results already present in the work
directory. Each dataset writes its samples to `<dataset-name>.jsonl`, and the
combined metrics are written to `result.json`.

## RL training

An RL configuration module must provide a `get_trainer()` function. The
trainer is normally created by passing a `UnifiedConfig` to `create_trainer()`.
The configuration defines the model and FSDP topology, training and evaluation
datasets, rollout behavior, checkpoint schedule, and worker concurrency.

Start training by passing the importable module name:

```bash
python -m alloylm.algorithm.rl.run my_training_config
```

`tests/resource/gsm8k.py` contains a complete multi-GPU GSM8K configuration.
Adjust its model, worker count, sequence lengths, and batch sizes for your
hardware before running it.

## Inference API

The inference stack exposes:

| Endpoint                    | Purpose                             |
| --------------------------- | ----------------------------------- |
| `POST /v1/chat/completions` | OpenAI-compatible chat completions  |
| `POST /v1/chat/interactive` | Stateful interactive chat sessions  |
| `POST /generate`            | Token-ID or prompt-based generation |
| `POST /abort_request`       | Request cancellation                |
| `GET /v1/models`            | Available models                    |
| `GET /health`               | Health check                        |

The proxy server can register multiple inference workers and route each request
to the least-busy worker. See `alloylm/engine/infer_engine/` and
`tests/test_infer/test_whole_system.py` for the server lifecycle and client
examples.

## Project layout

```text
alloylm/
|-- algorithm/
|   |-- base.py           # Task and dataset interfaces
|   |-- eval/             # Concurrent evaluation runner
|   `-- rl/               # RL algorithms, configuration, and trainer
|-- engine/
|   |-- infer_engine/      # API server, scheduler, sampler, and proxy
|   `-- train_engine/      # Distributed training integration
|-- impl/                  # Datasets, RL helpers, and model implementations
`-- server/                # Async inference client
tests/                     # Unit, integration, GPU, and system tests
tools/                     # Training/inference comparison utilities
```

## Tests

Run the default test discovery:

```bash
python -m unittest
```

## License

This project is derived from [XTuner](https://github.com/InternLM/xtuner) and contains substantial modifications for its training, inference, and RL architecture.

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE).
