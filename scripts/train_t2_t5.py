"""Train T5-small with LoRA adapters for Task 2.1."""

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.t5_lora import (  # noqa: E402
    build_t5_lora_model,
    count_trainable_parameters,
    format_t5_input,
    format_t5_output,
)
from src.utils import get_device, set_seed  # noqa: E402


class RecipeTextDataset(Dataset):
    """Load raw CSV rows and expose formatted T5 source/target strings."""

    def __init__(
        self,
        csv_path: Path,
        max_items: Optional[int] = None,
        seed: int = 42,
        shuffle: bool = False,
    ) -> None:
        df = pd.read_csv(csv_path)
        if max_items is not None and max_items < len(df):
            if shuffle:
                df = df.sample(n=max_items, random_state=seed)
            else:
                df = df.head(max_items)
        self.rows = df.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.rows.iloc[idx]
        ingredients = json.loads(row["Ingredients"])
        recipe = json.loads(row["Recipe"])
        return {
            "title": row.get("Title", ""),
            "ingredients": ingredients,
            "recipe": recipe,
            "source_text": format_t5_input(ingredients),
            "target_text": format_t5_output(recipe),
        }


class T5RecipeCollator:
    """Tokenize a batch of formatted T5 recipe examples."""

    def __init__(self, tokenizer, max_source_length: int = 256, max_target_length: int = 160) -> None:
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        sources = [ex["source_text"] for ex in examples]
        targets = [ex["target_text"] for ex in examples]

        model_inputs = self.tokenizer(
            sources,
            max_length=self.max_source_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        labels = self.tokenizer(
            text_target=targets,
            max_length=self.max_target_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )["input_ids"]
        labels[labels == self.tokenizer.pad_token_id] = -100
        model_inputs["labels"] = labels
        model_inputs["source_text"] = sources
        model_inputs["target_text"] = targets
        return model_inputs


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Path to a YAML training config.")
    parser.add_argument("--max_steps", type=int, default=None, help="Optional debug cap on optimizer steps.")
    parser.add_argument("--max_train_samples", type=int, default=None, help="Override train sample count.")
    parser.add_argument("--max_dev_samples", type=int, default=None, help="Override dev sample count.")
    parser.add_argument("--print_batch", action="store_true", help="Print one formatted batch before training.")
    return parser.parse_args()


def load_config(path: Path) -> Dict[str, Any]:
    """Load a YAML config and fill derived defaults."""
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config["config_name"] = path.stem
    config.setdefault("seed", 42)
    config.setdefault("model_name", "google-t5/t5-small")
    config.setdefault("train_path", "data/raw/train.csv")
    config.setdefault("dev_path", "data/raw/dev.csv")
    config.setdefault("output_dir", f"checkpoints/t2_t5_{path.stem}")
    config.setdefault("max_source_length", 256)
    config.setdefault("max_target_length", 160)
    config.setdefault("num_train_epochs", 5)
    config.setdefault("batch_size", 8)
    config.setdefault("learning_rate", 2e-4)
    config.setdefault("weight_decay", 0.0)
    config.setdefault("grad_clip", 1.0)
    config.setdefault("eval_every_steps", None)
    config.setdefault("max_steps", None)
    config.setdefault("use_amp", True)
    return config


def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, torch.Tensor]:
    """Move tensor fields in a collated batch onto ``device``."""
    return {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}


def print_formatted_batch(batch: Dict[str, Any]) -> None:
    """Print one source/target pair for smoke-test inspection."""
    print("\n=== Formatted T5 batch example ===")
    print("SOURCE:", batch["source_text"][0])
    print("TARGET:", batch["target_text"][0])
    print("input_ids shape:", tuple(batch["input_ids"].shape))
    print("labels shape:", tuple(batch["labels"].shape))


@torch.no_grad()
def evaluate(
    model,
    dataloader,
    device: torch.device,
    max_batches: Optional[int] = None,
    use_amp: bool = False,
) -> float:
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
    """Save optimizer state and scalar history next to a PEFT checkpoint."""
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
    """Run LoRA fine-tuning."""
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

    model, tokenizer = build_t5_lora_model(config["model_name"], config["lora"])
    model.to(device)
    trainable, total, ratio = count_trainable_parameters(model)
    print(f"Model: {config['model_name']}")
    print(f"LoRA config: {config['lora']}")
    print(f"Trainable params: {trainable:,} / {total:,} ({ratio:.3f}%)")

    train_dataset = RecipeTextDataset(
        Path(config["train_path"]),
        max_items=config.get("max_train_samples"),
        seed=config["seed"],
        shuffle=config.get("shuffle_train_subset", False),
    )
    dev_dataset = RecipeTextDataset(
        Path(config["dev_path"]),
        max_items=config.get("max_dev_samples"),
        seed=config["seed"],
        shuffle=False,
    )
    collator = T5RecipeCollator(
        tokenizer,
        max_source_length=config["max_source_length"],
        max_target_length=config["max_target_length"],
    )
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
        first_batch = next(iter(train_loader))
        print_formatted_batch(first_batch)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    start_epoch, global_step, history = load_training_state(last_dir, optimizer)
    if start_epoch > 0:
        from peft import PeftModel
        from transformers import AutoModelForSeq2SeqLM

        base_model = AutoModelForSeq2SeqLM.from_pretrained(config["model_name"])
        model = PeftModel.from_pretrained(base_model, last_dir, is_trainable=True).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"],
        )
        load_training_state(last_dir, optimizer)
        print(f"Resumed from {last_dir} at epoch={start_epoch}, global_step={global_step}")

    best_dev_loss = min((row["dev_loss"] for row in history if "dev_loss" in row), default=math.inf)
    max_steps = config.get("max_steps")
    use_amp = bool(config.get("use_amp", True) and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    train_start = time.time()

    model.train()
    for epoch in range(start_epoch + 1, int(config["num_train_epochs"]) + 1):
        epoch_losses = []
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
            epoch_losses.append(loss.item())
            progress.set_postfix(loss=f"{loss.item():.4f}", step=global_step)

            eval_every = config.get("eval_every_steps")
            should_eval_mid_epoch = eval_every and global_step % int(eval_every) == 0
            should_stop = max_steps is not None and global_step >= int(max_steps)
            if should_eval_mid_epoch or should_stop:
                dev_loss = evaluate(
                    model,
                    dev_loader,
                    device,
                    max_batches=config.get("max_eval_batches"),
                    use_amp=use_amp,
                )
                train_loss = float(sum(epoch_losses) / max(len(epoch_losses), 1))
                history.append({"epoch": epoch, "global_step": global_step, "train_loss": train_loss, "dev_loss": dev_loss})
                print(f"step={global_step} train_loss={train_loss:.4f} dev_loss={dev_loss:.4f}")
                model.save_pretrained(last_dir)
                tokenizer.save_pretrained(last_dir)
                save_training_state(last_dir, optimizer, epoch, global_step, history)
                if dev_loss < best_dev_loss:
                    best_dev_loss = dev_loss
                    model.save_pretrained(best_dir)
                    tokenizer.save_pretrained(best_dir)
                    save_training_state(best_dir, optimizer, epoch, global_step, history)
                if should_stop:
                    elapsed = time.time() - train_start
                    print(f"Stopped at max_steps={max_steps}. Elapsed: {elapsed/60:.1f} min")
                    return

        dev_loss = evaluate(
            model,
            dev_loader,
            device,
            max_batches=config.get("max_eval_batches"),
            use_amp=use_amp,
        )
        train_loss = float(sum(epoch_losses) / max(len(epoch_losses), 1))
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

    elapsed = time.time() - train_start
    print(f"Training complete. Elapsed: {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
