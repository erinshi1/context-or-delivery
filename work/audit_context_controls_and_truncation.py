from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer


def count_turns(text: object) -> int:
    text = "" if pd.isna(text) else str(text).strip()
    if not text:
        return 0
    return len(text.split(" [CTX] "))


def donor_availability(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group_col, label in [("speaker_x", "speaker-matched"), ("show_x", "show-matched")]:
        available = []
        for _, row in table.iterrows():
            pool = table[(table[group_col].eq(row[group_col])) & (~table["sample_id"].eq(row["sample_id"]))]
            available.append(not pool.empty)
        n_available = int(sum(available))
        rows.append(
            {
                "control": label,
                "available": n_available,
                "fallback": int(len(table) - n_available),
                "available_pct": n_available / len(table),
                "fallback_pct": (len(table) - n_available) / len(table),
            }
        )
    return pd.DataFrame(rows)


def context_turn_distribution(table: pd.DataFrame) -> pd.DataFrame:
    n = len(table)
    rows = [{"measure": "mean_context_turns", "value": float(table["n_context"].mean())}]
    rows.append({"measure": "median_context_turns", "value": float(table["n_context"].median())})
    for threshold in [0, 1, 2, 3, 4, 5, 6]:
        rows.append(
            {
                "measure": f"samples_with_more_than_{threshold}_turns",
                "value": int((table["n_context"] > threshold).sum()),
                "pct": float((table["n_context"] > threshold).mean()),
            }
        )
    rows.append({"measure": "samples_where_all_exceeds_c3", "value": int((table["n_context"] > 3).sum()), "pct": float((table["n_context"] > 3).mean())})
    return pd.DataFrame(rows)


def token_stats(table: pd.DataFrame, tokenizer_name: str, max_length: int) -> pd.DataFrame:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, local_files_only=True)
    rows = []
    for condition in ["target", "context_1_target", "context_2_target", "context_3_target", "context_all_target"]:
        lengths = []
        target_complete = []
        for text, target in zip(table[condition].fillna("").astype(str), table["target"].fillna("").astype(str)):
            ids = tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"]
            target_ids = tokenizer(target, add_special_tokens=False)["input_ids"]
            truncated_ids = tokenizer(text, add_special_tokens=True, truncation=True, max_length=max_length)["input_ids"]
            lengths.append(len(ids))
            if not target_ids:
                target_complete.append(True)
            else:
                # Natural-order inputs place the target at the end. Check whether all target tokens survive.
                target_complete.append(len(truncated_ids) >= len(target_ids) + 1 and truncated_ids[-(len(target_ids) + 1) : -1] == target_ids)
        s = pd.Series(lengths)
        rows.append(
            {
                "condition": condition,
                "mean_tokens": float(s.mean()),
                "median_tokens": float(s.median()),
                "p95_tokens": float(s.quantile(0.95)),
                "max_tokens": int(s.max()),
                "truncated": int((s > max_length).sum()),
                "truncated_pct": float((s > max_length).mean()),
                "target_complete_after_truncation": int(sum(target_complete)),
                "target_complete_pct": float(sum(target_complete) / len(target_complete)),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--context-table", type=Path, default=Path("outputs/context_bert_embeddings/context_table.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/context_bert_paper_ready_plus_v2"))
    parser.add_argument("--tokenizer", default="bert-base-uncased")
    parser.add_argument("--max-length", type=int, default=256)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(args.context_table)

    donor = donor_availability(table)
    turns = context_turn_distribution(table)
    tokens = token_stats(table, args.tokenizer, args.max_length)

    donor.to_csv(args.output_dir / "context_control_donor_availability.csv", index=False)
    turns.to_csv(args.output_dir / "context_turn_distribution.csv", index=False)
    tokens.to_csv(args.output_dir / "bert_context_token_stats.csv", index=False)

    print("Donor availability")
    print(donor.to_string(index=False))
    print("\nContext turns")
    print(turns.to_string(index=False))
    print("\nToken stats")
    print(tokens.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
