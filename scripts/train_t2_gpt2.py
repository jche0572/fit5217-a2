"""Train GPT-2 with LoRA adapters for Task 2.2."""

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import torch
import yaml
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.gpt2_lora import (  # noqa: E402
    build_gpt2_lora_model,
    count_trainable_parameters,
    format_gpt2_inference_prompt,
    format_gpt2_training_example,
)
from src.utils import get_device, set_seed  # noqa: E402


class GPT2RecipeDataset(Dataset):
    """Load raw recipe rows and expose GPT-2 decoder-only examples."""

    def __init__(self, csv_path: Path, tokenizer, max_length: int = 256, max_items: Optional[int] = None) -> None:
        df = pd.read_csv(csv_path)
        if max_items is not None:
            df = df.head(max_items)
        self.rows = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.rows.iloc[idx]
        ingredients = json.loads(row["Ingredients"])
        recipe = json.loads(row["Recipe"])
        example = format_gpt2_training_example(ingredients, recipe, self.tokenizer, max_length=self.max_length)
        example["prompt"] = format_gpt2_inference_prompt(ingredients)
        return example


class GPT2RecipeCollator:
    """Pad GPT-2 decoder-only examples and preserve label masking."""

    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        input_ids = [torch.tensor(ex["input_ids"], dtype=torch.long) for ex in examples]
        attention_mask = [torch.tensor(ex["attention_mask"], dtype=torch.long) for ex in examples]
        labels = [torch.tensor(ex["labels"], dtype=torch.long) for ex in examples]

        # GPT-2 用 left padding 做 generation 更自然; labels 的 padding 必须继续是 -100
        input_ids = [torch.flip(x, dims=[0]) for x in input_ids]
        attention_mask = [torch.flip(x, dims=[0]) for x in attention_mask]
        labels = [torch.flip(x, dims=[0]) for x in labels]
        batch_input_ids = torch.flip(
            pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id),
            dims=[1],
        )
        batch_attention = torch.flip(
            pad_sequence(attention_mask, batch_first=True, padding_value=0),
            dims=[1],
        )
        batch_labels = torch.flip(
            pad_sequence(labels, batch_first=True, padding_value=-100),
            dims=[1],
        )
        return {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention,
            "labels": batch_labels,
            "prompt": [ex["prompt"] for ex in examples],
        }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/t2_gpt2.yaml"))
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_dev_samples", type=int, default=None)
    parser.add_argument("--print_batch", action="store_true")
    return parser.parse_args()


def load_config(path: Path) -> Dict[str, Any]:
    """Load YAML config and fill defaults."""
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config.setdefault("seed", 42)
    config.setdefault("model_name", "openai-community/gpt2")
    config.setdefault("train_path", "data/raw/train.csv")
    config.setdefault("dev_path", "data/raw/dev.csv")
    config.setdefault("output_dir", "checkpoints/t2_gpt2")
    config.setdefault("num_train_epochs", 3)
    config.setdefault("batch_size", 16)
    config.setdefault("eval_batch_size", 16)
    config.setdefault("learning_rate", 2e-4)
    config.setdefault("weight_decay", 0.0)
    config.setdefault("grad_clip", 1.0)
    config.setdefault("max_length", 256)
    config.setdefault("use_amp", True)
    config.setdefault("max_steps", None)
    return config


def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, torch.Tensor]:
    """Move tensor fields in a collated batch onto ``device``."""
    return {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}


def print_masked_batch(batch: Dict[str, Any], tokenizer) -> None:
    """Print one smoke-test batch with visible label masking."""
    print("\n=== GPT-2 masked batch example ===")
    ids = batch["input_ids"][0].tolist()
    attention = batch["attention_mask"][0].tolist()
    labels = batch["labels"][0].tolist()
    print("Prompt:")
    print(batch["prompt"][0])
    first_real = next((i for i, value in enumerate(attention) if value == 1), 0)
    first_target = next((i for i, label in enumerate(labels) if label != -100), None)
    prompt_window_end = min(first_real + 40, len(ids))
    target_window_end = min((first_target or 0) + 40, len(ids))
    print(f"first real token index: {first_real}")
    print("first non-masked label index:", first_target)
    print("prompt input_ids window:", ids[first_real:prompt_window_end])
    print("prompt labels window:   ", labels[first_real:prompt_window_end])
    if first_target is not None:
        print("target input_ids window:", ids[first_target:target_window_end])
        print("target labels window:   ", labels[first_target:target_window_end])
        print("decoded supervised prefix:", tokenizer.decode(ids[first_target:first_target + 40]))


@torch.no_grad()
def evaluate(model, dataloader, device: torch.device, use_amp: bool = False, max_batches: Optional[int] = None) -> float:
    """Return mean dev loss."""
    model.eval()
    losses = []
    for step, batch in enumerate(tqdm(dataloader, desc="eval", leave=False), start=1):
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            outputs = model(**move_batch_to_device(batch, device))
        losses.append(outputs.loss.item())
        if max_batches is not None and step >= max_batches:
            break
    model.train()
    return float(sum(losses) / max(len(losses), 1))


def save_training_state(path: Path, optimizer, epoch: int, global_step: int, history: List[Dict[str, float]]) -> None:
    """Save optimizer state and scalar history next to adapter weights."""
    path.mkdir(parents=True, exist_ok=True)
    torch.save(optimizer.state_dict(), path / "optimizer.pt")
    with open(path / "training_state.json", "w", encoding="utf-8") as f:
        json.dump({"epoch": epoch, "global_step": global_step, "history": history}, f, indent=2)


def load_training_state(path: Path, optimizer) -> tuple[int, int, List[Dict[str, float]]]:
    """Load optimizer state and scalar history if present."""
    state_path = path / "training_state.json"
    optimizer_path = path / "optimizer.pt"
    if not state_path.exists():
        return 0, 0, []
    with open(state_path, "r", encoding="utf-8") as f:
        state = json.load(f)
    if optimizer_path.exists():
        optimizer.load_state_dict(torch.load(optimizer_path, map_location="cpu"))
    return int(state.get("epoch", 0)), int(state.get("global_step", 0)), list(state.get("history", []))


def main() -> None:
    """Run GPT-2 LoRA fine-tuning."""
    args = parse_args()
    config = load_config(args.config)
    if args.max_steps is not None:
        config["max_steps"] = args.max_steps
    if args.max_train_samples is not None:
        config["max_train_samples"] = args.max_train_samples
    if args.max_dev_samples is not None:
        config["max_dev_samples"] = args.max_dev_samples

    set_seed(config["seed"])
    device = get_device()
    output_dir = Path(config["output_dir"])
    last_dir = output_dir / "last"
    best_dir = output_dir / "best"
    output_dir.mkdir(parents=True, exist_ok=True)

    lora_config = dict(config["lora"])
    lora_config["model_name"] = config["model_name"]
    model, tokenizer = build_gpt2_lora_model(lora_config)
    model.to(device)
    trainable, total, ratio = count_trainable_parameters(model)
    print(f"Model: {config['model_name']}")
    print(f"LoRA config: {config['lora']}")
    print(f"Trainable params: {trainable:,} / {total:,} ({ratio:.3f}%)")

    train_dataset = GPT2RecipeDataset(
        Path(config["train_path"]),
        tokenizer,
        max_length=config["max_length"],
        max_items=config.get("max_train_samples"),
    )
    dev_dataset = GPT2RecipeDataset(
        Path(config["dev_path"]),
        tokenizer,
        max_length=config["max_length"],
        max_items=config.get("max_dev_samples"),
    )
    collator = GPT2RecipeCollator(tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=collator,
        num_workers=config.get("num_workers", 0),
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=config.get("eval_batch_size", config["batch_size"]),
        shuffle=False,
        collate_fn=collator,
        num_workers=config.get("num_workers", 0),
    )

    if args.print_batch:
        print_masked_batch(next(iter(train_loader)), tokenizer)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    start_epoch, global_step, history = load_training_state(last_dir, optimizer)
    if start_epoch > 0:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM

        base_model = AutoModelForCausalLM.from_pretrained(config["model_name"])
        base_model.config.pad_token_id = tokenizer.pad_token_id
        model = PeftModel.from_pretrained(base_model, last_dir, is_trainable=True).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
        load_training_state(last_dir, optimizer)
        print(f"Resumed from {last_dir} at epoch={start_epoch}, global_step={global_step}")

    use_amp = bool(config.get("use_amp", True) and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    max_steps = config.get("max_steps")
    best_dev_loss = min((row["dev_loss"] for row in history if "dev_loss" in row), default=math.inf)
    train_start = time.time()

    model.train()
    for epoch in range(start_epoch + 1, int(config["num_train_epochs"]) + 1):
        losses = []
        progress = tqdm(train_loader, desc=f"epoch {epoch}", leave=True)
        for batch in progress:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                outputs = model(**move_batch_to_device(batch, device))
                loss = outputs.loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
            scaler.step(optimizer)
            scaler.update()

            global_step += 1
            losses.append(loss.item())
            progress.set_postfix(loss=f"{loss.item():.4f}", step=global_step)

            if max_steps is not None and global_step >= int(max_steps):
                dev_loss = evaluate(model, dev_loader, device, use_amp=use_amp, max_batches=config.get("max_eval_batches"))
                train_loss = float(sum(losses) / max(len(losses), 1))
                history.append({"epoch": epoch, "global_step": global_step, "train_loss": train_loss, "dev_loss": dev_loss})
                print(f"step={global_step} train_loss={train_loss:.4f} dev_loss={dev_loss:.4f}")
                model.save_pretrained(last_dir)
                tokenizer.save_pretrained(last_dir)
                save_training_state(last_dir, optimizer, epoch, global_step, history)
                if dev_loss < best_dev_loss:
                    model.save_pretrained(best_dir)
                    tokenizer.save_pretrained(best_dir)
                    save_training_state(best_dir, optimizer, epoch, global_step, history)
                print(f"Stopped at max_steps={max_steps}. Elapsed: {(time.time() - train_start)/60:.1f} min")
                return

        dev_loss = evaluate(model, dev_loader, device, use_amp=use_amp, max_batches=config.get("max_eval_batches"))
        train_loss = float(sum(losses) / max(len(losses), 1))
        history.append({"epoch": epoch, "global_step": global_step, "train_loss": train_loss, "dev_loss": dev_loss})
        print(f"epoch={epoch} train_loss={train_loss:.4f} dev_loss={dev_loss:.4f}")

        model.save_pretrained(last_dir)
        tokenizer.save_pretrained(last_dir)
        save_training_state(last_dir, optimizer, epoch, global_step, history)
        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)
            save_training_state(best_dir, optimizer, epoch, global_step, history)
            print(f"Saved best checkpoint to {best_dir}")

    print(f"Training complete. Elapsed: {(time.time() - train_start)/60:.1f} min")


if __name__ == "__main__":
    main()
