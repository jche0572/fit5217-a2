"""Dataset and batching utilities for Task 1 sequence-to-sequence models."""

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from src.preprocessing import EOS_ID, PAD_ID


class RecipeDataset(Dataset):
    """Load preprocessed recipe ID pairs from ``outputs/processed/*.pkl``.

    Parameters
    ----------
    path : str or Path
        Pickle file produced by the Task 1.1 preprocessing pipeline.
    max_tgt_len : int, default 80
        Maximum target length after truncation. The default follows the data
        exploration result where the recipe-token p95 is around 80.
    max_items : int, optional
        If provided, keep only the first ``max_items`` examples. This is used
        by local sample runs and does not create a new processed file.
    """

    def __init__(
        self,
        path: Union[str, Path],
        max_tgt_len: int = 80,
        max_items: Optional[int] = None,
    ) -> None:
        self.path = Path(path)
        self.max_tgt_len = max_tgt_len

        with open(self.path, "rb") as f:
            examples: List[Dict[str, List[int]]] = pickle.load(f)

        if max_items is not None:
            examples = examples[:max_items]

        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        row = self.examples[idx]
        src_ids = row["ingredients_ids"]
        tgt_ids = row["recipe_ids"]

        # decoder 训练至少需要 <SOS> + 一个目标 token
        if len(tgt_ids) > self.max_tgt_len:
            tgt_ids = tgt_ids[: self.max_tgt_len]
            # 截断后强制保留 EOS, 否则 generate 学不到稳定停止信号
            tgt_ids[-1] = EOS_ID

        return {"src_ids": src_ids, "tgt_ids": tgt_ids}


def collate_fn(batch: Sequence[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
    """Pad a batch of variable-length source and target ID sequences."""
    src_tensors = [torch.tensor(item["src_ids"], dtype=torch.long) for item in batch]
    tgt_tensors = [torch.tensor(item["tgt_ids"], dtype=torch.long) for item in batch]

    # pack_padded_sequence 需要真实长度; clamp_min 防止极端空输入让 RNN 崩掉
    src_lengths = torch.tensor([max(len(x), 1) for x in src_tensors], dtype=torch.long)
    tgt_lengths = torch.tensor([max(len(x), 1) for x in tgt_tensors], dtype=torch.long)

    src_padded = pad_sequence(src_tensors, batch_first=True, padding_value=PAD_ID)
    tgt_padded = pad_sequence(tgt_tensors, batch_first=True, padding_value=PAD_ID)

    return {
        "src": src_padded,
        "src_lengths": src_lengths,
        "tgt": tgt_padded,
        "tgt_lengths": tgt_lengths,
    }


def load_vocab_size(path: Union[str, Path] = "outputs/vocab.pkl") -> int:
    """Return vocabulary size from the persisted Task 1.1 vocabulary state."""
    with open(path, "rb") as f:
        state: Dict[str, Any] = pickle.load(f)
    return len(state["word2idx"])
