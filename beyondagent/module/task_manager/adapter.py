import json
import tempfile
from typing import Iterator, Optional, Sequence
import uuid


from beyondagent.schema.task import TaskObjective
from verl.utils.dataset.rl_dataset import RLHFDataset
from torch.utils.data import IterableDataset
import pandas as pd
from omegaconf import DictConfig, ListConfig
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.processing_utils import ProcessorMixin


def to_rl_dataset(
    tasks: Sequence[TaskObjective],
    tokenizer: PreTrainedTokenizer,
    config: DictConfig,
    processor: Optional[ProcessorMixin] = None,
) -> RLHFDataset:
    processed_records = []

    for task_obj in tasks:
        task = task_obj.task

        # 处理 query 字段
        query = task.query
        if isinstance(query, list):
            query_str = "\n".join(str(x) for x in query)
        else:
            query_str = str(query)

        prompt = [{"content": query_str, "role": "user"}]

        # 构建 reward_model
        ground_truth = [task_obj.ground_truth] if task_obj.ground_truth else []

        # 生成 uuid
        record_uuid = str(uuid.uuid4())

        # 构建单条记录
        record = {
            "data_source": task.env_type,
            "prompt": prompt,
            "reward_model": {"ground_truth": ground_truth, "style": "rule"},
            "uuid": record_uuid,
            "extras": {"task_id": task.task_id},
        }

        processed_records.append(record)

    df = pd.DataFrame(processed_records)
    with tempfile.NamedTemporaryFile(delete=False) as f:
        df.to_parquet(f.name)

    # 转换为 Dataset
    return RLHFDataset([f.name], tokenizer, config, processor)


class OnflyRlDataset(IterableDataset):
    def __init__(self, release_used_dataset: bool = True):
        super().__init__()
        self._do_release_used_dataset = release_used_dataset

        self._datasets: list[RLHFDataset] = []
        self._passed_datasets_cnt = 0
        self._cur_dataset = 0
        self._cur = 0

    def __len__(self):
        pass

    def __iter__(self) -> Iterator:
        return self

    def __next__(self):
        if len(self._datasets) <= self._cur_dataset:
            raise StopIteration

        this_cur = self._cur - self._passed_datasets_cnt
        if this_cur >= len(self._datasets[self._cur_dataset]):
            self._passed_datasets_cnt += len(self._datasets[self._cur_dataset])
            self._cur_dataset += 1
            this_cur = 0

        if len(self._datasets) <= self._cur_dataset:
            raise StopIteration

        # release used datasets
        if self._do_release_used_dataset:
            self._release_used_dataset()

        self._cur += 1
        return self._datasets[self._cur_dataset][this_cur]

    @property
    def num_rest_data(self) -> int:
        return sum([len(d) for d in self._datasets[self._cur_dataset :]]) - (
            self._cur - self._passed_datasets_cnt
        )

    def append_dataset(self, dataset: RLHFDataset):
        self._datasets.append(dataset)

    def _release_used_dataset(self):
        self._datasets = self._datasets[self._cur_dataset :]
        self._cur_dataset = 0
