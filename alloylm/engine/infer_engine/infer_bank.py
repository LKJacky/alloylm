import hashlib
import json
from collections import OrderedDict

import ray


class InferBank:
    #  save inference information (input_ids, labels, etc.) for each inference request
    def __init__(self, max_num_items=128 * 8 * 50):
        self.bank = OrderedDict()
        self.max_num_items = max_num_items

    def update(self, infer_info: dict):
        # update the bank with new inference information
        for key, value in infer_info.items():
            self.bank[key] = value
            if len(self.bank) > self.max_num_items:
                obj = self.bank.popitem(last=False)
                ray.internal.free(obj[1])  # release the memory of the object

    def retrieve_infer_info(self, messages):
        key = self.hash_messages(messages)
        return self.bank.get(key)

    @staticmethod
    def hash_messages(messages):
        # make sure the last message is from the assistant, otherwise we will not be able to get the correct inference information
        for i in range(len(messages), 1, -1):
            if messages[i - 1]["role"] == "assistant":
                break
        assert i > 0, "No assistant message found in messages"
        messages = messages[:i]
        serialized = json.dumps(messages, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode()).hexdigest()
