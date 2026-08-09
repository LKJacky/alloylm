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


class MeasureTime:
    saved_time = {}

    def __init__(self, label):
        self.label = label

    def __enter__(self):
        import time

        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        import time

        self.end_time = time.time()
        self.interval = self.end_time - self.start_time
        MeasureTime.saved_time[self.label] = self.interval

    @classmethod
    def clear(cls):
        cls.saved_time = {}


_collected_error = set()


def report_error_once(message: str):
    global _collected_error
    if message not in _collected_error:
        logger = get_logger()
        logger.critical(message)
        _collected_error.add(message)
