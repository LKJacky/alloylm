import time
from abc import ABC, abstractmethod

from pydantic import BaseModel as PydanticBaseModel
from pydantic import ConfigDict
from torch import Tensor, nn

from alloylm.engine.train_engine.utils import DEFAULT_FSDP_CONFIG, FSDPConfig


def tolist(tensor: Tensor):
    if tensor is None:
        return None
    elif isinstance(tensor, Tensor):
        return tensor.flatten().tolist()
    else:
        assert isinstance(tensor, list), f"Expected tensor or list, but got {type(tensor)}"
        return tensor


# data for training


class TrainInput(PydanticBaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    input_ids: Tensor  # 1 seq_lens.sum()
    position_ids: Tensor  # 1 seq_lens.sum()
    seq_lens: Tensor  # B


# data for inference


class DeviceSession:
    def __init__(self, session_id: int):
        self.session_id = session_id
        self.tokens: list = []  # wait to be forwarded
        self.forwarded_tokens: list = []

        self.log_probs: list = []  # for reward calculation
        self.entropy: list = []  # for reward calculation

        self.forwarded_log_probs: list = []  # for reward calculation
        self.forwarded_entropy: list = []  # for reward calculation
        self.create_time = time.time()

    @property
    def resource_id(self):
        return self.session_id

    def step(self):
        self.forwarded_tokens.extend(self.tokens)
        self.tokens = []

        self.forwarded_log_probs.extend(self.log_probs)
        self.log_probs = []

        self.forwarded_entropy.extend(self.entropy)
        self.entropy = []

    def append_input_tokens(self, tokens: Tensor, log_probs: Tensor = None, entropy: Tensor = None):
        # Convert tensor to list and extend
        tokens, log_probs, entropy = tolist(tokens), tolist(log_probs), tolist(entropy)

        self.tokens.extend(tokens)

        if log_probs is None:
            log_probs = [0.0] * len(tokens)
        if entropy is None:
            entropy = [0.0] * len(tokens)

        self.log_probs.extend(log_probs)
        self.entropy.extend(entropy)

    def truncate_tokens(self, max_length):
        assert max_length < len(self.tokens)
        removed_tokens = self.tokens[max_length:]
        removed_log_probs = self.log_probs[max_length:]
        removed_entropy = self.entropy[max_length:]

        self.tokens = self.tokens[:max_length]
        self.log_probs = self.log_probs[:max_length]
        self.entropy = self.entropy[:max_length]

        return removed_tokens, removed_log_probs, removed_entropy

    def total_num_tokens(self):
        return len(self.forwarded_tokens) + len(self.tokens)

    def release_forwarded(self):
        self.tokens = self.forwarded_tokens + self.tokens
        self.log_probs = self.forwarded_log_probs + self.log_probs
        self.entropy = self.forwarded_entropy + self.entropy

        self.forwarded_tokens = []
        self.forwarded_log_probs = []
        self.forwarded_entropy = []

    def on_device(self):
        raise NotImplementedError("This method should be implemented in subclasses")


class Cache:
    @abstractmethod
    def prepare(self):
        pass

    @abstractmethod
    def close(self):
        pass

    @abstractmethod
    def allocate_cache(self, session: DeviceSession) -> bool:
        raise NotImplementedError("This method should be implemented in subclasses")

    @abstractmethod
    def reset(self, sessions: list[DeviceSession]):
        raise NotImplementedError("This method should be implemented in subclasses")

    @abstractmethod
    def create_device_session(self, session_id: int) -> DeviceSession:
        return DeviceSession(session_id=session_id)

    @property
    @abstractmethod
    def max_infer_length(self):
        raise NotImplementedError()

    @property
    @abstractmethod
    def max_infer_batch_size(self):
        raise NotImplementedError()

    @abstractmethod
    def cache_usage(self, device_sessions: tuple[DeviceSession, ...] = ()) -> float:
        raise NotImplementedError()


class AlloyLMModelConfig(PydanticBaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    path: str
    model_cls: object
    tokenizer_path: str = ""
    fsdp_config: FSDPConfig = FSDPConfig()

    def model_post_init(self, context):
        if self.tokenizer_path is None:
            self.tokenizer_path = self.path
        return super().model_post_init(context)

    def build(self) -> "AlloyLMModel":
        assert self.path, "Model path must be specified in the configuration."
        return self.model_cls.from_pretrained(self.path, fsdp_config=self.fsdp_config)


class AlloyLMModel(ABC):
    lm_head: nn.Module

    def __init__(self, fsdp_config: FSDPConfig):
        self.fsdp_config = fsdp_config
        self.fsdp_config.init_device_mesh()
        self.shard_mode = "train"

    # for hf
    @classmethod
    @abstractmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | None = None,
        *model_args,
        fsdp_config: FSDPConfig = DEFAULT_FSDP_CONFIG,
        **kwargs,
    ) -> "AlloyLMModel":
        raise NotImplementedError("from_pretrained method is not implemented yet.")

    @abstractmethod
    def save_pretrained(self, save_directory, **kwargs):
        raise NotImplementedError("save_pretrained method is not implemented yet.")

    # sharding methods

    @abstractmethod
    def train_shard(self):
        raise NotImplementedError("train_shard method is not implemented yet.")

    @abstractmethod
    def infer_shard(self, max_prefill_length: int = 0):
        raise NotImplementedError("infer_shard method is not implemented yet.")

    # for inference engine

    @abstractmethod
    def get_real_vocab_size(self, tokenizer):
        return -1

    @abstractmethod
    def prefill(self, device_sessions: list[DeviceSession], cache: Cache) -> Tensor:
        """
        device_sessions: list of AttentionDeviceSession
        """

    @abstractmethod
    def decode(self, device_sessions: list[DeviceSession], cache: Cache) -> Tensor:
        """
        device_sessions: list of AttentionDeviceSession
        """

    @abstractmethod
    def create_cache(self, memory_usage=0.8, use_cuda_graph=True) -> Cache:
        raise NotImplementedError("This method should be implemented in subclasses")

    # for training engine
    @abstractmethod
    def train_forward(self, input: TrainInput) -> Tensor:
        pass
