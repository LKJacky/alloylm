import json
import os
import socket
import sys

import orjson


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


_logger_instance = None


def log_format(rank, debug=False):
    formatter = f"[AlloyLM][RANK {rank}]"
    formatter += "[{time:YYYY-MM-DD HH:mm:ss}][<level>{level}</level>]"

    if debug:
        formatter += "[<cyan>{name}</cyan>:"
        formatter += "<cyan>{function}</cyan>:"
        formatter += "<cyan>{line}</cyan>]"

    formatter += " <level>{message}</level>"
    return formatter


def get_logger(level="INFO"):
    from loguru import logger

    global _logger_instance
    if _logger_instance is None:
        # Remove the original logger in Python to prevent duplicate printing.
        logger.remove()
        logger.add(sys.stderr, level=level, format=log_format(0, debug=level == "DEBUG"))
        _logger_instance = logger
    return _logger_instance


def init_logger(log_file):
    global _logger_instance  # noqa: PLW0602
    _logger_instance.remove()
    log_file = os.path.join(log_file)
    _logger_instance.add(sys.stderr, level="INFO", format=log_format(0, False))
    _logger_instance.add(log_file, level="DEBUG", format=log_format(0, True), backtrace=True, catch=True)


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


def get_free_port(forbid_port=[]):
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
