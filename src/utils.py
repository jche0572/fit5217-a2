"""Common utilities: reproducibility, device detection, logging, environment checks."""

import logging
import os
import random
import sys
from pathlib import Path
from typing import Optional, Union

import numpy as np


def set_seed(seed: int = 42) -> None:
    """Seed Python ``random``, NumPy, and PyTorch (CPU + CUDA) for reproducibility.

    Also forces cuDNN into deterministic mode so convolution-style ops produce
    the same results across runs at the cost of some throughput.

    Parameters
    ----------
    seed : int, default 42
        Global random seed. The assignment specifies 42 across all submissions.
    """
    # PYTHONHASHSEED 影响 dict / set 的哈希顺序,要在解释器层面固定
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    # torch 是可选依赖(Task 3 的纯 RAG 环境不一定要装),用 try/except 兜底
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            # 多 GPU 时也一起设
            torch.cuda.manual_seed_all(seed)
        # cudnn 默认会按硬件挑最快算法,会引入非确定性,这里强制关掉
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def get_device():
    """Return the best available torch device.

    Preference order: CUDA → Apple Silicon MPS → CPU. Importing torch lazily so
    this module can be loaded in non-torch environments (e.g. a Task 3 RAG-only
    Colab runtime).

    Returns
    -------
    torch.device
        Device to move models and tensors onto.
    """
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    # Apple Silicon 的 MPS 后端,本地 macOS 调试时有用
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def setup_logging(
    log_file: Optional[Union[str, Path]] = None,
    level: int = logging.INFO,
    name: str = "fit5217",
) -> logging.Logger:
    """Configure a logger that writes to stdout and optionally a file.

    Re-invoking this function with the same ``name`` will not duplicate handlers,
    so it is safe to call from notebook cells that are run repeatedly.

    Parameters
    ----------
    log_file : str or Path, optional
        If provided, log records are also appended to this path. Parent
        directories are created automatically.
    level : int, default ``logging.INFO``
        Logging threshold.
    name : str, default ``"fit5217"``
        Logger name. Pass distinct names if you want separate loggers per task.

    Returns
    -------
    logging.Logger
        Configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    # 已经配置过的话直接返回,避免 notebook 反复执行时挂上重复 handler 导致重复输出
    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 控制台 handler 始终挂上
    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    # 文件 handler 可选,要确保父目录存在再创建
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    # 阻止日志冒泡到 root logger,否则 Jupyter 会再打印一遍
    logger.propagate = False
    return logger


def is_colab() -> bool:
    """Return True if running inside a Google Colab runtime.

    Used to switch checkpoint/index paths, pip install behaviour, and Drive
    mounting on/off depending on the host.
    """
    # Colab 注入了独有的 google.colab 模块,本地 / Jupyter 都没有这个名字
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False
