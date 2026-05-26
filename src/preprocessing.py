"""Tokenization and vocabulary utilities for the FIT5217 A2 recipe-generation task.

Builds the integer-ID pipeline for the Task 1 RNN encoder-decoder.

Components
----------
* :func:`tokenize` — a lowercase regex tokenizer that isolates punctuation as
  its own tokens and keeps simple numeric expressions (including fractions
  such as ``1/2`` and decimals like ``1.5``) as a single token.
* :class:`Vocabulary` — ``word2idx`` / ``idx2word`` mapping with four mandatory
  special tokens (``<PAD>=0``, ``<UNK>=1``, ``<SOS>=2``, ``<EOS>=3``). Built
  **only** from the training split as required by the assignment.
* :func:`ingredients_to_ids` / :func:`recipe_to_ids` — convenience encoders
  that join a JSON-list row value into a single string, tokenize, and encode.
  ``recipe_to_ids`` wraps the result with ``<SOS>`` and ``<EOS>`` for the
  teacher-forcing decoder.

Design choices
--------------
(A) **Tokenization**: all text is lowercased first; numbers and simple
    fractions (``1/2``, ``1.5``) remain as single tokens; every punctuation
    character is split into its own token, *including the trailing periods of
    abbreviations* (e.g. ``tbsp.`` → ``["tbsp", "."]``). Keeping the period
    attached to each abbreviation would otherwise double the type count for
    every short unit word ("c" / "c." / "oz" / "oz." …) without adding
    information the decoder cannot learn from context. Letter classes use the
    Unicode-aware pattern ``[^\\W\\d_]`` so foreign characters (``é``, ``û``)
    are preserved rather than silently dropped.

(B) **Shared vocabulary**: a single vocabulary is built from both the
    ingredients and recipe sides of the training split. Lexical overlap is
    very large ("butter", "sugar", "stir", numbers, units), and a shared
    embedding matrix is the standard NMT setup that keeps the parameter count
    down and lets the same token use the same representation on both sides.

(C) **List → sequence joining**: list elements (both ingredients and recipe
    steps) are joined with a single space. The existing periods at the end of
    most recipe steps and the commas inside ingredient phrases ("1 small
    onion, chopped") already provide natural separators after tokenization, so
    no additional separator token is required.

The ``min_freq`` threshold (decision D in the spec) is empirically chosen via
``scripts/compare_min_freq.py``.
"""

import pickle
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Union

# 四个特殊 token 的 ID 严格固定为 0~3, 后续模型构建 padding mask / loss ignore_index
# 都依赖 PAD_ID=0 这一约定, 不可随意更改
PAD_ID: int = 0
UNK_ID: int = 1
SOS_ID: int = 2
EOS_ID: int = 3
SPECIAL_TOKENS: tuple = ("<PAD>", "<UNK>", "<SOS>", "<EOS>")
assert tuple(range(4)) == (PAD_ID, UNK_ID, SOS_ID, EOS_ID), "special token ids must be 0..3"


# 分词正则的三段交替, 顺序很重要 (前面的优先匹配):
#   1. 数字 / 分数 / 小数  例: "350", "1/2", "1.5"
#   2. 字母序列 (Unicode 字母, 不含数字下划线)  例: "butter", "crème"
#   3. 其他非空白单字符 (标点)  例: ".", ",", "(", ")"
_TOKEN_RE = re.compile(r"\d+(?:[./]\d+)?|[^\W\d_]+|[^\w\s]")


def tokenize(text: str) -> List[str]:
    """Lowercase ``text`` and split it into a list of tokens.

    See the module docstring (decision A) for the tokenization rationale.

    Parameters
    ----------
    text : str
        Raw text to be tokenized.

    Returns
    -------
    list[str]
        Token list. Empty string yields an empty list.
    """
    # 先 lower 再 findall, 字母类既包含 ASCII 也包含 Unicode 字母
    return _TOKEN_RE.findall(text.lower())


class Vocabulary:
    """Token-to-id mapping with mandatory ``<PAD>/<UNK>/<SOS>/<EOS>`` specials.

    The four special tokens are inserted before any real token so their ids
    are deterministically 0/1/2/3 regardless of which corpus the vocabulary is
    built from. Construction is done via :meth:`build_from_texts`; ``__init__``
    only seeds the specials.
    """

    # 这些类属性便于在外部直接通过 Vocabulary.PAD_ID 引用
    PAD_ID: int = PAD_ID
    UNK_ID: int = UNK_ID
    SOS_ID: int = SOS_ID
    EOS_ID: int = EOS_ID
    SPECIAL_TOKENS: tuple = SPECIAL_TOKENS

    def __init__(self) -> None:
        # 初始时只放四个特殊 token, build_from_texts 才会填充其余词
        self.word2idx: dict = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
        self.idx2word: dict = {i: tok for tok, i in self.word2idx.items()}
        # 记录构建时的频率阈值与原始词频, 便于复盘和保存到磁盘
        self.min_freq: Optional[int] = None
        self.token_counts: dict = {}

    # ----- 基本查询 -----
    def __len__(self) -> int:
        return len(self.word2idx)

    def __contains__(self, token: str) -> bool:
        return token in self.word2idx

    # ----- 构建 -----
    @classmethod
    def build_from_texts(
        cls,
        tokenized_texts: Iterable[Sequence[str]],
        min_freq: int = 3,
    ) -> "Vocabulary":
        """Build a vocabulary from an iterable of already-tokenized texts.

        Only tokens with frequency ``>= min_freq`` enter the vocabulary; rarer
        tokens will map to ``<UNK>`` at encode time. Iteration order of the
        added tokens is fixed (frequency desc, then lexicographic asc) so that
        running the builder twice on the same corpus yields identical ids.

        Parameters
        ----------
        tokenized_texts : Iterable[Sequence[str]]
            An iterable of token lists. The caller is responsible for tokenizing
            and for ensuring only the training split is used.
        min_freq : int, default 3
            Inclusive lower bound on token frequency.

        Returns
        -------
        Vocabulary
            The constructed vocabulary.
        """
        vocab = cls()
        vocab.min_freq = min_freq

        # 一次性遍历统计词频
        counter: Counter = Counter()
        for tokens in tokenized_texts:
            counter.update(tokens)
        vocab.token_counts = dict(counter)

        # 频率高的先注册, 同频按字典序, 保证可复现
        for token, count in sorted(counter.items(), key=lambda x: (-x[1], x[0])):
            if count < min_freq:
                continue
            # 极少数情况下真实 token 可能与特殊 token 字面冲突, 跳过避免覆盖
            if token in vocab.word2idx:
                continue
            idx = len(vocab.word2idx)
            vocab.word2idx[token] = idx
            vocab.idx2word[idx] = token
        return vocab

    # ----- 编码 / 解码 -----
    def encode(self, tokens: Sequence[str]) -> List[int]:
        """Map a list of string tokens to a list of integer ids.

        Tokens not present in the vocabulary are mapped to ``<UNK>``.
        """
        # dict.get 提供 O(1) 查询和 OOV 默认值, 比 if-in-else 更紧凑
        return [self.word2idx.get(tok, UNK_ID) for tok in tokens]

    def decode(self, ids: Sequence[int], skip_special: bool = True) -> str:
        """Map a list of ids back to a space-joined string.

        Boundary specials (``<PAD>``, ``<SOS>``, ``<EOS>``) are skipped by
        default so the result reads as natural text. ``<UNK>`` is **kept**
        so the reader can see what was lost in encoding.
        """
        skip_ids = {PAD_ID, SOS_ID, EOS_ID} if skip_special else set()
        # idx2word.get(i, "<UNK>") 兼容传入了越界 id 的情况 (训练早期模型可能出现)
        return " ".join(
            self.idx2word.get(i, "<UNK>") for i in ids if i not in skip_ids
        )

    # ----- 持久化 -----
    def save(self, path: Union[str, Path]) -> None:
        """Pickle this vocabulary's state to ``path``."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # 仅持久化最小必要状态, 避免日后类结构变化导致 load 失败
        state = {
            "word2idx": self.word2idx,
            "min_freq": self.min_freq,
            "token_counts": self.token_counts,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "Vocabulary":
        """Restore a vocabulary previously persisted via :meth:`save`."""
        with open(path, "rb") as f:
            state = pickle.load(f)
        vocab = cls()
        vocab.word2idx = state["word2idx"]
        # idx2word 直接由 word2idx 反向重建, 不必序列化
        vocab.idx2word = {i: w for w, i in vocab.word2idx.items()}
        vocab.min_freq = state["min_freq"]
        vocab.token_counts = state["token_counts"]
        return vocab


# --------------------------------------------------------------------------- #
# 行级 helpers: 直接处理 train.csv 中一行的 Ingredients / Recipe JSON 列表
# --------------------------------------------------------------------------- #


def ingredients_to_ids(
    ingredients: Sequence[str], vocab: Vocabulary
) -> List[int]:
    """Encode a JSON list of ingredient phrases into a flat ID sequence.

    Parameters
    ----------
    ingredients : Sequence[str]
        The ``Ingredients`` field of a single row, already parsed from JSON.
    vocab : Vocabulary
        The (shared) vocabulary built from the training split.

    Returns
    -------
    list[int]
        Token ids. No ``<SOS>`` / ``<EOS>`` is added — the encoder does not
        need them.
    """
    # 直接空格拼成一段话, 让 tokenize 自然产出 token 序列
    text = " ".join(ingredients)
    return vocab.encode(tokenize(text))


def recipe_to_ids(recipe: Sequence[str], vocab: Vocabulary) -> List[int]:
    """Encode a JSON list of recipe steps into ``<SOS> tokens <EOS>``.

    Parameters
    ----------
    recipe : Sequence[str]
        The ``Recipe`` field of a single row, already parsed from JSON.
    vocab : Vocabulary
        The (shared) vocabulary built from the training split.

    Returns
    -------
    list[int]
        Token ids with ``<SOS>`` prepended and ``<EOS>`` appended, as required
        by the teacher-forcing decoder.
    """
    text = " ".join(recipe)
    body = vocab.encode(tokenize(text))
    return [SOS_ID] + body + [EOS_ID]


# --------------------------------------------------------------------------- #
# 模块自检 —— 运行 `python src/preprocessing.py` 时执行
# --------------------------------------------------------------------------- #


def _self_test() -> None:
    """Internal sanity checks executed when the module is run directly."""
    # 1) 四个特殊 token 必须同时出现在 word2idx 和 idx2word, 且 id 严格 0..3
    v_empty = Vocabulary()
    for tok, expected_id in zip(SPECIAL_TOKENS, range(4)):
        assert tok in v_empty.word2idx, f"{tok} missing from word2idx"
        assert expected_id in v_empty.idx2word, f"id {expected_id} missing from idx2word"
        assert v_empty.word2idx[tok] == expected_id
        assert v_empty.idx2word[expected_id] == tok

    # 2) 用最小语料构造词表, 验证 encode -> decode 对非 OOV token 能精确还原
    corpus = [
        tokenize("Combine ingredients in saucepan."),
        tokenize("Boil 5 minutes, stirring constantly."),
        tokenize("Cool and serve."),
    ]
    vocab = Vocabulary.build_from_texts(corpus, min_freq=1)
    tokens = tokenize("Boil 5 minutes.")
    ids = vocab.encode(tokens)
    assert all(i != UNK_ID for i in ids), "min_freq=1 should keep every token"
    decoded = vocab.decode(ids)
    assert decoded == " ".join(tokens), f"round-trip failed: {decoded!r}"

    # 3) recipe_to_ids 必须以 <SOS> 开头并以 <EOS> 结尾
    rec_ids = recipe_to_ids(["Combine ingredients.", "Cool and serve."], vocab)
    assert rec_ids[0] == SOS_ID, "recipe_to_ids must start with <SOS>"
    assert rec_ids[-1] == EOS_ID, "recipe_to_ids must end with <EOS>"

    # 4) save -> load 后状态保持一致
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        vocab.save(tmp_path)
        reloaded = Vocabulary.load(tmp_path)
        assert reloaded.word2idx == vocab.word2idx, "word2idx changed after reload"
        assert reloaded.idx2word == vocab.idx2word, "idx2word changed after reload"
        assert reloaded.min_freq == vocab.min_freq
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    print("[preprocessing] self-test passed.")


if __name__ == "__main__":
    _self_test()
