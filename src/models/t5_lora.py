"""T5 + LoRA utilities for Task 2.1 recipe generation."""

import re
from typing import Any, Dict, List, Sequence, Tuple, Union

import torch


def format_t5_input(ingredients_list: list) -> str:
    """Format a list of ingredients as the T5 source prompt."""
    # 使用稳定、短小的任务前缀, 让 T5 明确这是 ingredients -> recipe 的生成任务
    ingredients = " ; ".join(str(item).strip() for item in ingredients_list if str(item).strip())
    return f"generate recipe from ingredients: {ingredients}"


def format_t5_output(recipe_list: list) -> str:
    """Format a list of recipe steps as a numbered target string."""
    # 编号步骤对人和正则都友好, 后续 parse_t5_output 可以稳定反解析
    steps = [str(step).strip() for step in recipe_list if str(step).strip()]
    return " ".join(f"{i}. {step}" for i, step in enumerate(steps, start=1))


def parse_t5_output(text: str) -> list[str]:
    """Parse a generated T5 string back into a list of recipe steps."""
    text = " ".join(str(text).strip().split())
    if not text:
        return []

    # 优先解析 "1. ... 2. ..." 编号格式; 如果模型漏编号, 再按句号粗略切分
    matches = list(re.finditer(r"(?:^|\s)(\d+)[.)]\s+", text))
    if matches:
        steps = []
        for i, match in enumerate(matches):
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            step = text[start:end].strip(" ;")
            if step:
                steps.append(step)
        return steps

    chunks = re.split(r"(?<=[.!?])\s+", text)
    return [chunk.strip() for chunk in chunks if chunk.strip()]


def _normalise_lora_config(lora_config: Union[Dict[str, Any], Any]) -> Dict[str, Any]:
    """Convert a dict or PEFT config object into keyword arguments."""
    if isinstance(lora_config, dict):
        return dict(lora_config)
    return {
        "r": lora_config.r,
        "lora_alpha": lora_config.lora_alpha,
        "lora_dropout": lora_config.lora_dropout,
        "target_modules": lora_config.target_modules,
    }


def build_t5_lora_model(model_name, lora_config) -> Tuple[Any, Any]:
    """Build a T5 sequence-to-sequence model with LoRA adapters attached.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier, e.g. ``"google-t5/t5-small"``.
    lora_config : dict or peft.LoraConfig
        LoRA hyperparameters. Dicts may use either ``r`` or ``lora_r``.

    Returns
    -------
    tuple
        ``(model, tokenizer)`` where ``model`` is a PEFT-wrapped T5 model.
    """
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    config_dict = _normalise_lora_config(lora_config)
    r = config_dict.pop("lora_r", config_dict.pop("r", 8))
    alpha = config_dict.pop("lora_alpha", 16)
    dropout = config_dict.pop("lora_dropout", 0.05)
    target_modules = config_dict.pop("target_modules", ["q", "v"])

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        **config_dict,
    )
    model = get_peft_model(base_model, peft_config)
    return model, tokenizer


@torch.no_grad()
def generate_recipes(model, tokenizer, ingredients_list, **gen_kwargs) -> list[str]:
    """Generate recipe strings for one or more ingredient lists."""
    model.eval()
    if not ingredients_list:
        return []

    # 兼容单条 list[str] 和批量 list[list[str]]
    if all(isinstance(item, str) for item in ingredients_list):
        batch_ingredients: List[Sequence[str]] = [ingredients_list]
    else:
        batch_ingredients = ingredients_list

    prompts = [format_t5_input(list(ingredients)) for ingredients in batch_ingredients]
    device = next(model.parameters()).device
    max_input_length = gen_kwargs.pop("max_input_length", 256)
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_input_length,
    ).to(device)

    defaults = {
        "max_new_tokens": 128,
        "num_beams": 4,
        "early_stopping": True,
        "no_repeat_ngram_size": 3,
    }
    defaults.update(gen_kwargs)
    output_ids = model.generate(**inputs, **defaults)
    return tokenizer.batch_decode(output_ids, skip_special_tokens=True)


def count_trainable_parameters(model) -> tuple[int, int, float]:
    """Return trainable count, total count, and trainable percentage."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    ratio = 100.0 * trainable / total if total else 0.0
    return trainable, total, ratio
