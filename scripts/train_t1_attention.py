"""Train the Task 1.3 GRU encoder-decoder with Bahdanau attention."""

import argparse
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import RecipeDataset, collate_fn, load_vocab_size
from src.models.rnn_attention import Seq2SeqAttention
from src.training import EarlyStopping, eval_epoch, make_criterion, save_checkpoint, train_epoch
from src.utils import get_device, set_seed, setup_logging


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", action="store_true", help="Use a small local slice for smoke testing.")
    parser.add_argument("--epochs", type=int, default=10, help="Maximum number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=128, help="Mini-batch size.")
    parser.add_argument("--max_tgt_len", type=int, default=80, help="Target truncation length.")
    parser.add_argument("--embed_size", type=int, default=128, help="Embedding dimension.")
    parser.add_argument("--hidden_size", type=int, default=256, help="GRU hidden dimension.")
    parser.add_argument("--num_layers", type=int, default=1, help="Number of GRU layers.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Embedding dropout.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--clip", type=float, default=1.0, help="Gradient clipping max norm.")
    parser.add_argument("--patience", type=int, default=3, help="Early-stopping patience on dev loss.")
    parser.add_argument("--train_path", type=Path, default=Path("outputs/processed/train_ids.pkl"))
    parser.add_argument("--dev_path", type=Path, default=Path("outputs/processed/dev_ids.pkl"))
    parser.add_argument("--vocab_path", type=Path, default=Path("outputs/vocab.pkl"))
    parser.add_argument("--checkpoint_dir", type=Path, default=Path("checkpoints/t1_attention"))
    parser.add_argument("--sample_train_items", type=int, default=1024)
    parser.add_argument("--sample_dev_items", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    """Run attention-model training."""
    args = parse_args()
    set_seed(42)
    logger = setup_logging(args.checkpoint_dir / "train.log", name="t1_attention")
    device = get_device()

    train_limit = args.sample_train_items if args.sample else None
    dev_limit = args.sample_dev_items if args.sample else None
    train_dataset = RecipeDataset(args.train_path, max_tgt_len=args.max_tgt_len, max_items=train_limit)
    dev_dataset = RecipeDataset(args.dev_path, max_tgt_len=args.max_tgt_len, max_items=dev_limit)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    vocab_size = load_vocab_size(args.vocab_path)
    model = Seq2SeqAttention(
        vocab_size=vocab_size,
        embed_size=args.embed_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = make_criterion()
    stopper = EarlyStopping(patience=args.patience)

    config = vars(args).copy()
    config.update(
        {
            "vocab_size": vocab_size,
            "device": str(device),
            "seed": 42,
            "cell": "GRU",
            "bidirectional": False,
            "attention": "bahdanau_manual",
        }
    )
    logger.info("config=%s", config)
    logger.info("train_examples=%d dev_examples=%d", len(train_dataset), len(dev_dataset))

    train_losses = []
    dev_losses = []
    best_path = args.checkpoint_dir / "best.pt"
    last_path = args.checkpoint_dir / "last.pt"

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, clip=args.clip)
        dev_loss = eval_epoch(model, dev_loader, criterion, device)
        elapsed = time.time() - start
        train_losses.append(train_loss)
        dev_losses.append(dev_loss)

        logger.info(
            "epoch=%d train_loss=%.4f dev_loss=%.4f elapsed=%.1fs",
            epoch,
            train_loss,
            dev_loss,
            elapsed,
        )

        save_checkpoint(last_path, model, optimizer, epoch, train_losses, dev_losses, config)
        if dev_loss <= min(dev_losses):
            save_checkpoint(best_path, model, optimizer, epoch, train_losses, dev_losses, config)
            logger.info("saved best checkpoint to %s", best_path)

        if stopper.step(dev_loss):
            logger.info("early stopping at epoch=%d best_dev_loss=%.4f", epoch, stopper.best_loss)
            break

    logger.info("done. last checkpoint: %s", last_path)


if __name__ == "__main__":
    main()
