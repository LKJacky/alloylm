from __future__ import annotations

import json
import logging
import os
import socket
import sys
import time
from contextvars import ContextVar
from pathlib import Path
from threading import Lock

import orjson
import ray


def dispatch_triton():
    """Monkey-patch triton.Config to cap num_warps=4 on SM90 (H100) GPUs for
    FLA kernels."""
    import torch
    from triton import Config

    config_origin_init = Config.__init__

    def dispatch_init__(self: Config, *args, **kwargs):
        config_origin_init(self, *args, **kwargs)
        self.num_warps = min(self.num_warps, 4)

    if (
        torch.cuda.is_available()
        and not hasattr(dispatch_triton, "TRITON_CONFIG_DISPATCHED")
        and torch.cuda.get_device_properties(0).major == 9
    ):
        print("dispatch triton config")
        Config.__init__ = dispatch_init__
        dispatch_triton.TRITON_CONFIG_DISPATCHED = True


_logger_lock = Lock()


def get_logger(
    name: str = "default",
    path: str | os.PathLike[str] | None = None,
    output_to_stdout: bool = True,
    log_level: int | str = logging.INFO,
    force_recreate: bool = False,
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    with _logger_lock:
        if force_recreate or not logger.handlers:
            for handler in logger.handlers[:]:
                logger.removeHandler(handler)
                handler.close()
            formatter = logging.Formatter(
                f"[AlloyLM][{name}][%(asctime)s][%(levelname)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            if path:
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                file_handler = logging.FileHandler(path)
                file_handler.setLevel(logging.DEBUG)
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)

            if output_to_stdout:
                stdout_handler = logging.StreamHandler(sys.stdout)
                stdout_handler.setLevel(log_level)
                stdout_handler.setFormatter(formatter)
                logger.addHandler(stdout_handler)
    return logger


def write_jsonl(file_path, data):
    if "/" in file_path:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as f:
        f.writelines(orjson.dumps(item).decode("utf-8") + "\n" for item in data)


def load_jsonl(file_path):
    data = []

    with open(file_path) as f:
        for line in f:
            try:
                data.append(orjson.loads(line))
            except json.JSONDecodeError:
                data.append(json.loads(line))
    return data


# for server


def get_free_port(forbid_port=()):
    """Let the OS choose an available port, avoiding forbidden ports."""
    while True:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("", 0))  # 0 means let OS pick
                port = s.getsockname()[1]  # Get the port number assigned
                if port not in forbid_port:
                    return port
        except OSError:
            continue  # If the port is not available, try again


# for tokenizer


def get_chat_template_from_tokenizer(tokenizer):

    class ChatTemplate:
        """Callable wrapping ``tokenizer.apply_chat_template`` to the signature the
        SFT data path expects: ``(messages, add_generation_prompt=False) -> str``.

        The same instance is used both while packing (``sft_tokenize`` /
        ``analyze_jsonl_file`` on the driver) and by the engine's ``SFTDataset``
        (via ``set_sft_data`` on the workers), so the ``num_tokens`` computed while
        packing match what the engine recomputes and its
        ``assert num_token == len(input_id)`` holds. A plain instance (not a bound
        function) is used so it survives being pickled to the Ray actors.
        """

        def __init__(self, tokenizer):
            self.tokenizer = tokenizer

        def render(self, messages, add_generation_prompt=False, **kwargs):
            return self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=add_generation_prompt, tokenize=False
            )

    return ChatTemplate(tokenizer)


class MeasureTime:
    """Context manager timing a section and its nested sections as a tree.

    Nested ``with MeasureTime(...)`` blocks attach to their enclosing block,
    so ``summary()`` returns a flat ``{path.to.label: seconds}`` dict and
    ``format_summary()`` the indented tree. The current node lives in a
    ``ContextVar``, so concurrent asyncio tasks each build their own tree and
    the stack is restored exactly on exit.
    """

    _cur_node: ContextVar[MeasureTime | None] = ContextVar("measure_time_cur_node", default=None)

    def __init__(self, label):
        self.label = label
        self.parent: MeasureTime | None = None
        self.children: list[MeasureTime] = []
        self.start_time = None
        self.interval = -1.0
        self._token = None

    def __enter__(self):
        self.parent = self._cur_node.get()
        if self.parent is not None:
            self.parent.children.append(self)
        self._token = self._cur_node.set(self)
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.interval = time.perf_counter() - self.start_time
        # Restore the enclosing node before logging so the stack stays
        # consistent even if the logger raises.
        self._cur_node.reset(self._token)
        return False

    def summary(self):
        """Return a flat ``{label.path: seconds}`` dict of this tree."""
        summary_dict = {}

        def _summary(node, parent_prefix):
            key = parent_prefix + node.label
            summary_dict[key] = node.interval
            children_sum = 0
            for child in node.children:
                _summary(child, key + ".")
                children_sum += child.interval
            if node.children:
                summary_dict[key + ".others"] = node.interval - children_sum

        _summary(self, "")
        return summary_dict

    def format_summary(self):
        """Return the indented tree string for human logs."""

        def _format(node, parent_prefix):
            indent = parent_prefix.count(".")
            key = parent_prefix + node.label
            s = "  " * indent + f"{node.label}: {node.interval:.4f}s"
            children_sum = 0
            for child in node.children:
                s += "\n" + _format(child, key + ".")
                children_sum += child.interval
            if node.children:
                others = node.interval - children_sum
                s += "\n" + "  " * (indent + 1) + f"others: {others:.4f}s"
            return s

        return _format(self, "")


# ray


def init_ray():
    if not ray.is_initialized():
        try:
            ray.init(address="auto")
        except BaseException:  # noqa
            ray.init(
                ignore_reinit_error=True,
                include_dashboard=False,
                _system_config={
                    "prestart_worker_first_driver": False,
                    "enable_worker_prestart": False,
                },
            )
