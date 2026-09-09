from __future__ import annotations

import json

from torch.utils.tensorboard import SummaryWriter

from ...utils import get_logger


class DummySummaryWriter(SummaryWriter):
    default = None

    def __init__(self, *args, **kwargs):
        pass

    @classmethod
    def init_writer(cls, *args, **kwargs):
        cls.default = SummaryWriter(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.__class__.default, name)


def get_tb_writer():
    return DummySummaryWriter()


def load_jsonl(path):
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line))
    return data


_collected_error = set()


def report_error_once(message: str):
    global _collected_error  # noqa
    if message not in _collected_error:
        logger = get_logger()
        logger.critical(message)
        _collected_error.add(message)
