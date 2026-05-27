"""Generate the assignment submission prediction workbook."""

import argparse
import json
from pathlib import Path
from typing import Iterable, List

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs" / "generated_35423757.xlsx"
EXPECTED_TEST_ROWS = 1081


PREDICTION_CANDIDATES = {
    "Recipe_T1_Baseline": [
        "t1_baseline_test.json",
    ],
    "Recipe_T1_Attention": [
        "t1_attention_test.json",
    ],
    "Recipe_T2_T5": [
        "t2_t5_config2_test.json",
        "t2_t5_t2_t5_config2_test.json",
    ],
    "Recipe_T2_GPT2": [
        "t2_gpt2_beam_test.json",
    ],
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test_csv", type=Path, default=ROOT / "data" / "raw" / "test.csv")
    parser.add_argument("--pred_dir", type=Path, default=ROOT / "outputs" / "predictions")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def resolve_prediction_file(pred_dir: Path, candidates: Iterable[str]) -> Path:
    """Return the first existing candidate prediction file."""
    for name in candidates:
        path = pred_dir / name
        if path.exists():
            return path
    names = ", ".join(candidates)
    raise FileNotFoundError(f"None of these prediction files exist in {pred_dir}: {names}")


def load_prediction_strings(path: Path, expected_len: int) -> List[str]:
    """Load prediction strings from a JSON prediction file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise TypeError(f"{path} must contain a JSON list, got {type(data).__name__}")
    if len(data) != expected_len:
        raise ValueError(f"{path} length mismatch: {len(data)} != {expected_len}")

    predictions: List[str] = []
    for i, item in enumerate(data):
        if isinstance(item, dict):
            value = item.get("prediction")
        else:
            value = item
        if value is None:
            raise ValueError(f"{path} row {i} has prediction=None")
        predictions.append(str(value))
    return predictions


def validate_no_empty(df: pd.DataFrame) -> None:
    """Raise if any required output cell is null or an empty string."""
    null_counts = df.isna().sum()
    empty_counts = (df.astype(str).apply(lambda col: col.str.len() == 0)).sum()
    bad = {
        col: {"null": int(null_counts[col]), "empty": int(empty_counts[col])}
        for col in df.columns
        if int(null_counts[col]) or int(empty_counts[col])
    }
    if bad:
        raise ValueError(f"Found null/empty cells: {bad}")


def main() -> None:
    """Create ``outputs/generated_35423757.xlsx`` from raw test rows and predictions."""
    args = parse_args()
    test_df = pd.read_csv(args.test_csv, dtype=str, keep_default_na=False)
    if len(test_df) != EXPECTED_TEST_ROWS:
        raise ValueError(f"test.csv row count mismatch: {len(test_df)} != {EXPECTED_TEST_ROWS}")

    output_df = pd.DataFrame(
        {
            "Title": test_df["Title"],
            "Ingredients": test_df["Ingredients"],
            "Recipe_Gold": test_df["Recipe"],
        }
    )

    actual_files = {}
    for column, candidates in PREDICTION_CANDIDATES.items():
        path = resolve_prediction_file(args.pred_dir, candidates)
        actual_files[column] = path
        output_df[column] = load_prediction_strings(path, expected_len=len(test_df))

    expected_columns = [
        "Title",
        "Ingredients",
        "Recipe_Gold",
        "Recipe_T1_Baseline",
        "Recipe_T1_Attention",
        "Recipe_T2_T5",
        "Recipe_T2_GPT2",
    ]
    output_df = output_df[expected_columns]
    validate_no_empty(output_df)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_excel(args.output, index=False, engine="openpyxl")

    print("=== Prediction files used ===")
    for column, path in actual_files.items():
        print(f"{column}: {path}")
    print(f"\nSaved: {args.output}")
    print(f"Rows: {len(output_df)}")
    print("\n=== Preview first 2 rows ===")
    print(output_df.head(2).to_string(index=False))
    print("\n=== Non-empty check ===")
    for column in output_df.columns:
        n_empty = int((output_df[column].astype(str).str.len() == 0).sum())
        print(f"{column}: empty={n_empty}, null={int(output_df[column].isna().sum())}")


if __name__ == "__main__":
    main()
