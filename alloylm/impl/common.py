import copy
import os

import orjson
import requests

from alloylm.algorithm.base import Dataset, DatasetConfig, Task, TaskData, TaskItem
from alloylm.server.client import HighConcurrentClient as AsyncClient


def load_jsonl(file_path: str):
    data = []
    with open(file_path) as f:
        for line in f:
            data.append(orjson.loads(line))
    return data


class LazyJsonlData:
    def __init__(self, path):
        self.line_offsets = []
        with open(path) as f:
            offset = 0
            for line in f:
                self.line_offsets.append(offset)
                offset += len(line.encode("utf-8"))
        self.file_path = path

    def __getitem__(self, index):
        with open(self.file_path) as f:
            f.seek(self.line_offsets[index])
            line = f.readline()
            return orjson.loads(line)


class JsonlDataset(Dataset):
    def __init__(self, config: "JsonlDatasetConfig"):
        super().__init__(config)
        self.data = load_jsonl(config.file_path)

    def __len__(self):
        return len(self.data) * self.config.repeat


class JsonlDatasetConfig(DatasetConfig):
    file_path: str
    system_prompt: str = ""
    repeat: int = 1


class OneStepTask(Task):
    @classmethod
    async def infer(cls, task_data: TaskData) -> TaskData:
        client = AsyncClient(api_key="EMPTY", base_url=task_data.infer_args.model_url)
        try:
            response = await client.chat.completions.create(
                model=task_data.infer_args.model_name,
                messages=task_data.messages,
                **task_data.infer_args.sample_args,
            )
            task_data.messages.append({"role": "assistant", "content": response.choices[0].message.content})

            task_data.finish_reason = response.choices[0].finish_reason
            task_data.input_tokens = response.usage.prompt_tokens
            task_data.output_tokens = response.usage.completion_tokens
            task_data.total_tokens = response.usage.total_tokens
        except TimeoutError as e:
            raise RuntimeError(
                f"Inference timeout {task_data.infer_args.sample_args.get('timeout', 'unknow')} seconds"
            ) from e
        finally:
            await client.close()
        return task_data


class OneStepJsonlDataset(Dataset):
    def __init__(self, config: JsonlDatasetConfig):
        super().__init__(config)

    def __getitem__(self, index):
        messages = copy.deepcopy(self.data[index]["message"])
        if messages[0]["role"] == "system":
            messages[0] = {"role": "system", "content": self.config.system_prompt}
        else:
            messages.insert(0, {"role": "system", "content": self.config.system_prompt})

        return TaskItem(
            task_data=TaskData(
                id=f"{self.config.name}_{index}",
                messages=messages,
                infer_args=self.config.infer_args,
                others={"origin_data": self.data[index]},
            ),
            task_cls=self.config.task_cls,
        )


class OneStepJsonlDatasetConfig(JsonlDatasetConfig):
    system_prompt: str = "Answer below question and response your final answer in \\boxed"


def extract_boxed(content: str):
    if "\\boxed{" not in content:
        return ""
    else:
        content = content.split("\\boxed{")[-1]
        end = 1
        branket_count = 1
        while branket_count > 0 and end < len(content):
            if content[end] == "{":
                branket_count += 1
            elif content[end] == "}":
                branket_count -= 1
            end += 1
        return content[: end - 1]


class LLMVerifier:
    LLM_JUDGER_NAME = None

    @classmethod
    def judger_name(cls):
        if cls.LLM_JUDGER_NAME is None:
            try:
                cls.LLM_JUDGER_NAME = requests.get(
                    f"{os.environ.get('LLM_JUDGER_URL', 'http://127.0.0.1:8000/v1')}/models",
                    headers={"Authorization": "Bearer "},
                ).json()["data"][0]["id"]
            except Exception:
                cls.LLM_JUDGER_NAME = "no llm judger"
        if cls.LLM_JUDGER_NAME == "no llm judger":
            return None
        else:
            return cls.LLM_JUDGER_NAME

    @classmethod
    async def verify(cls, question, prediction, answer):
        verification_prompt = "Please as a grading expert, judge whether the final answers given by the candidates below are consistent with the standard answers, that is, whether the candidates answered correctly. \n    \n    Here are some evaluation criteria:\n    1. Please refer to the given standard answer. You don't need to re-generate the answer to the question because the standard answer has been given. You only need to judge whether the candidate's answer is consistent with the standard answer according to the form of the question. Don't try to answer the original question. You can assume that the standard answer is definitely correct.\n    2. Because the candidate's answer may be different from the standard answer in the form of expression, before making a judgment, please understand the question and the standard answer first, and then judge whether the candidate's answer is correct, but be careful not to try to answer the original question.\n    3. Some answers may contain multiple items, such as multiple-choice questions, multiple-select questions, fill-in-the-blank questions, etc. As long as the answer is the same as the standard answer, it is enough. For multiple-select questions and multiple-blank fill-in-the-blank questions, the candidate needs to answer all the corresponding options or blanks correctly to be considered correct.\n    4. Some answers may be expressed in different ways, such as some answers may be a mathematical expression, some answers may be a textual description, as long as the meaning expressed is the same. And some formulas are expressed in different ways, but they are equivalent and correct.\n    5. If the prediction is given with \\boxed{{}}, please ignore the \\boxed{{}} and only judge whether the candidate's answer is consistent with the standard answer.\n\n    Please judge whether the following answers are consistent with the standard answer based on the above criteria. Grade the predicted answer of this new question as one of:\n    A: CORRECT \n    B: INCORRECT\n    Just return the letters \"A\" or \"B\", with no text around it.\n\n    Here is your task. Simply reply with either CORRECT, INCORRECT. Don't apologize or correct yourself if there was a mistake; we are just trying to grade the answer.\n\n\n    <Original Question Begin>: \n{question}\n<Original Question End>\n\n\n    <Gold Target Begin>: \n{answer}\n<Gold Target End>\n\n\n    <Predicted Answer Begin>: \n{prediction}\n<Predicted End>\n\n\n    \n    Judging the correctness of candidates' answers:"
        judger_name = cls.judger_name()
        if judger_name:
            client = AsyncClient(base_url=f"{os.environ.get('LLM_JUDGER_URL', 'http://127.0.0.1:8000/v1')}")
            try:
                verification_prompt = verification_prompt.format(
                    question=question,
                    prediction=prediction,
                    answer=answer,
                )

                verification_result = await client.chat.completions.create(
                    model=judger_name,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a helpful assistant who evaluates the correctness and quality of models' outputs.",
                        },
                        {"role": "user", "content": verification_prompt},
                    ],
                    max_completion_tokens=1,
                )
            finally:
                await client.close()
            response = verification_result.choices[0].message.content.upper().strip()
            result = 1.0 if response == "A" else 0
            return result
        else:
            print("No LLM judger available, skip verification.")
            return None
