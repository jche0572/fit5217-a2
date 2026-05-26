"""Generate and save Bahdanau attention heatmaps for two test examples."""

import argparse
import os
import pickle
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs" / "matplotlib_cache"))

import matplotlib.pyplot as plt

from src.dataset import RecipeDataset, collate_fn, load_vocab_size
from src.models.rnn_attention import Seq2SeqAttention
from src.preprocessing import EOS_ID, PAD_ID, SOS_ID
from src.training import load_checkpoint
from src.utils import get_device, set_seed


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/t1_attention/best.pt"))
    parser.add_argument("--test_path", type=Path, default=Path("outputs/processed/test_ids.pkl"))
    parser.add_argument("--vocab_path", type=Path, default=Path("outputs/vocab.pkl"))
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/attention_heatmaps"))
    parser.add_argument("--num_samples", type=int, default=2)
    parser.add_argument("--max_len", type=int, default=80)
    return parser.parse_args()


def load_idx2word(path: Path) -> dict[int, str]:
    """Load the inverse vocabulary mapping from the saved vocabulary state."""
    with open(path, "rb") as f:
        state = pickle.load(f)
    return {idx: token for token, idx in state["word2idx"].items()}


def ids_to_tokens(ids: torch.Tensor, idx2word: dict[int, str], stop_at_eos: bool = False) -> list[str]:
    """Convert token IDs to display tokens."""
    tokens = []
    for token_id in ids.detach().cpu().tolist():
        if token_id == PAD_ID:
            continue
        if token_id == SOS_ID:
            continue
        if token_id == EOS_ID:
            if stop_at_eos:
                break
            continue
        tokens.append(idx2word.get(token_id, "<UNK>"))
    return tokens


def build_model(args: argparse.Namespace, device: torch.device) -> Seq2SeqAttention:
    """Instantiate the attention model and load checkpoint weights."""
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = state.get("config", {})
    model = Seq2SeqAttention(
        vocab_size=config.get("vocab_size", load_vocab_size(args.vocab_path)),
        embed_size=config.get("embed_size", 128),
        hidden_size=config.get("hidden_size", 256),
        num_layers=config.get("num_layers", 1),
        dropout=config.get("dropout", 0.1),
    ).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    model.eval()
    return model


def save_heatmap(
    attention: torch.Tensor,
    src_tokens: list[str],
    generated_tokens: list[str],
    path: Path,
) -> None:
    """Save one attention heatmap image."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = max(len(generated_tokens), 1)
    cols = max(len(src_tokens), 1)
    matrix = attention[:rows, :cols].detach().cpu().numpy()

    fig_width = max(8, min(18, cols * 0.35))
    fig_height = max(4, min(16, rows * 0.28))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_xticks(range(cols))
    ax.set_xticklabels(src_tokens or ["<empty>"], rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(rows))
    ax.set_yticklabels(generated_tokens or ["<empty>"], fontsize=8)
    ax.set_xlabel("Ingredient tokens")
    ax.set_ylabel("Generated recipe tokens")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    """Generate attention heatmaps from the selected checkpoint."""
    args = parse_args()
    set_seed(42)
    device = get_device()
    idx2word = load_idx2word(args.vocab_path)
    model = build_model(args, device)

    dataset = RecipeDataset(args.test_path, max_tgt_len=args.max_len, max_items=args.num_samples)
    loader = DataLoader(dataset, batch_size=args.num_samples, shuffle=False, collate_fn=collate_fn)
    batch = next(iter(loader))
    src = batch["src"].to(device)
    src_lengths = batch["src_lengths"].to(device)

    generated, attention = model.generate(
        src,
        src_lengths,
        max_len=args.max_len,
        return_attention=True,
    )

    for i in range(src.size(0)):
        src_tokens = ids_to_tokens(batch["src"][i], idx2word)
        generated_tokens = ids_to_tokens(generated[i], idx2word, stop_at_eos=True)
        save_heatmap(
            attention[i],
            src_tokens,
            generated_tokens,
            args.output_dir / f"attention_sample_{i + 1}.png",
        )
        print(f"saved {args.output_dir / f'attention_sample_{i + 1}.png'}")


if __name__ == "__main__":
    main()
