"""GPT-2 + LoRA utilities for Task 2.2 recipe generation."""

from typing import Any, Dict, List, Sequence, Tuple, Union

import torch


MODEL_NAME = "openai-community/gpt2"


def _format_ingredients(ingredients: Sequence[str]) -> str:
    """Format ingredient strings as a compact bullet list."""
    return "\n".join(f"- {str(item).strip()}" for item in ingredients if str(item).strip())


def _format_recipe(recipe: Sequence[str]) -> str:
    """Format recipe steps as numbered lines."""
    return "\n".join(f"{i}. {str(step).strip()}" for i, step in enumerate(recipe, start=1) if str(step).strip())


def format_gpt2_inference_prompt(ingredients) -> str:
    """Format ingredients as the GPT-2 inference prompt."""
    return f"Ingredients:\n{_format_ingredients(ingredients)}\n\nRecipe:\n"


def format_gpt2_training_example(ingredients, recipe, tokenizer, max_length: int = 256) -> Dict[str, List[int]]:
    """Create a masked decoder-only training example for GPT-2.

    The prompt tokens cover the ingredient section and the ``Recipe:`` marker.
    Their labels are set to ``-100`` so cross-entropy is computed only on the
    recipe continuation.
    """
    prompt = format_gpt2_inference_prompt(ingredients)
    target = _format_recipe(recipe) + tokenizer.eos_token

    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    target_ids = tokenizer(target, add_special_tokens=False)["input_ids"]

    if len(prompt_ids) + len(target_ids) > max_length:
        # 单序列太长时必须保留一段 recipe 监督信号, 否则整条 labels 都是 -100 会得到 nan loss
        target_keep = min(len(target_ids), max_length // 2)
        prompt_keep = max_length - target_keep
        if prompt_keep > len(prompt_ids):
            prompt_keep = len(prompt_ids)
            target_keep = max_length - prompt_keep
        prompt_ids = prompt_ids[-prompt_keep:]
        target_ids = target_ids[:target_keep]

    input_ids = prompt_ids + target_ids
    labels = [-100] * len(prompt_ids) + target_ids
    attention_mask = [1] * len(input_ids)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def build_gpt2_lora_model(lora_config=None) -> Tuple[Any, Any]:
    """Build GPT-2 with LoRA adapters attached."""
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    config = dict(lora_config or {})
    model_name = config.pop("model_name", MODEL_NAME)
    r = config.pop("r", config.pop("lora_r", 8))
    alpha = config.pop("lora_alpha", 16)
    dropout = config.pop("lora_dropout", 0.05)
    target_modules = config.pop("target_modules", ["c_attn"])

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base_model = AutoModelForCausalLM.from_pretrained(model_name)
    base_model.config.pad_token_id = tokenizer.pad_token_id

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        fan_in_fan_out=True,
        **config,
    )
    model = get_peft_model(base_model, peft_config)
    return model, tokenizer


@torch.no_grad()
def generate_recipes(model, tokenizer, ingredients_list, decoding_strategy: str = "greedy", **kwargs) -> list[str]:
    """Generate recipe continuations with greedy, beam, or sampling decoding."""
    model.eval()
    if not ingredients_list:
        return []

    # 兼容单条 list[str] 和批量 list[list[str]]
    if all(isinstance(item, str) for item in ingredients_list):
        batch_ingredients = [ingredients_list]
    else:
        batch_ingredients = ingredients_list

    prompts = [format_gpt2_inference_prompt(ingredients) for ingredients in batch_ingredients]
    device = next(model.parameters()).device
    max_prompt_length = kwargs.pop("max_prompt_length", 192)
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_prompt_length,
    ).to(device)

    strategy = decoding_strategy.lower()
    generation_args = {
        "max_new_tokens": kwargs.pop("max_new_tokens", 128),
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if strategy == "greedy":
        generation_args.update({"do_sample": False, "num_beams": 1})
    elif strategy == "beam":
        generation_args.update({"do_sample": False, "num_beams": kwargs.pop("num_beams", 4), "early_stopping": True})
    elif strategy in {"sample", "sampling"}:
        generation_args.update(
            {
                "do_sample": True,
                "top_p": kwargs.pop("top_p", 0.9),
                "temperature": kwargs.pop("temperature", 0.8),
                "num_beams": 1,
            }
        )
    else:
        raise ValueError(f"Unknown decoding_strategy: {decoding_strategy}")

    generation_args.update(kwargs)
    output_ids = model.generate(**inputs, **generation_args)
    prompt_width = inputs["input_ids"].shape[1]
    continuations = output_ids[:, prompt_width:]
    return tokenizer.batch_decode(continuations, skip_special_tokens=True)


def count_trainable_parameters(model) -> tuple[int, int, float]:
    """Return trainable count, total count, and trainable percentage."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    ratio = 100.0 * trainable / total if total else 0.0
    return trainable, total, ratio
