from __future__ import annotations

import json
import time
from contextvars import ContextVar

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


_collected_error = set()


def report_error_once(message: str):
    global _collected_error  # noqa
    if message not in _collected_error:
        logger = get_logger()
        logger.critical(message)
        _collected_error.add(message)
