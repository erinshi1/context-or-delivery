from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluate import binary_metrics
from src.sarcasm_data import load_sarcasm_organized


TEXT_CONDITIONS = [
    "target",
    "context_1",
    "context_2",
    "context_3",
    "context_all",
    "target_context_1",
    "target_context_2",
    "target_context_3",
    "target_context_all",
]

MAIN_CONDITIONS = [
    ("target_text", "target", ()),
    ("context_1_text", "context_1", ()),
    ("context_2_text", "context_2", ()),
    ("context_3_text", "context_3", ()),
    ("context_all_text", "context_all", ()),
    ("target_context_1_text", "target_context_1", ()),
    ("target_context_2_text", "target_context_2", ()),
    ("target_context_3_text", "target_context_3", ()),
    ("target_context_all_text", "target_context_all", ()),
    ("target_text_av", "target", ("audio", "video")),
    ("target_context_1_text_av", "target_context_1", ("audio", "video")),
    ("target_context_2_text_av", "target_context_2", ("audio", "video")),
    ("target_context_3_text_av", "target_context_3", ("audio", "video")),
    ("target_context_all_text_av", "target_context_all", ("audio", "video")),
    ("av_only", None, ("audio", "video")),
]

SHUFFLE_CONDITIONS = [
    ("target_context_3_text", "target_context_3", ()),
    ("target_random_context_3_text", "target_random_context_3", ()),
    ("target_same_speaker_context_3_text", "target_same_speaker_context_3", ()),
    ("target_same_show_context_3_text", "target_same_show_context_3", ()),
]


def clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return " ".join(str(value).split())


def build_context_table(data_dir: Path, sample_ids: list[str], seed: int) -> pd.DataFrame:
    meta_path = data_dir / "core" / "final_context_videos" / "SE-MUStARD-Final.csv"
    all_rows = pd.read_csv(meta_path)
    rng = np.random.default_rng(seed)
    records = []
    for sample_id in sample_ids:
        key_rows = all_rows[all_rows["KEY"].astype(str) == str(sample_id)].copy()
        labeled_positions = key_rows.index[key_rows["SARCASM"].notna()].tolist()
        if not labeled_positions:
            continue
        target_index = labeled_positions[-1]
        target_row = all_rows.loc[target_index]
        context_rows = key_rows.loc[key_rows.index < target_index]
        context = [clean_text(x) for x in context_rows["SENTENCE"].tolist()]
        target = clean_text(target_row["SENTENCE"])

        def last_n(n: int) -> str:
            return " [CTX] ".join(context[-n:])

        records.append(
            {
                "sample_id": str(sample_id),
                "target": target,
                "context_1": last_n(1),
                "context_2": last_n(2),
                "context_3": last_n(3),
                "context_all": " [CTX] ".join(context),
                "n_context": len(context),
            }
        )

    table = pd.DataFrame(records)
    for n in [1, 2, 3]:
        table[f"target_context_{n}"] = (
            table["target"].fillna("") + " [SEP] " + table[f"context_{n}"].fillna("")
        ).str.strip()
    table["target_context_all"] = (
        table["target"].fillna("") + " [SEP] " + table["context_all"].fillna("")
    ).str.strip()

    base_contexts = table["context_3"].to_numpy()
    table["target_random_context_3"] = (
        table["target"].to_numpy() + " [SEP] " + rng.permutation(base_contexts)
    )
    return table


def add_group_random_contexts(table: pd.DataFrame, metadata: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    merged = table.merge(metadata[["sample_id", "speaker", "show"]], on="sample_id", how="left")
    for group_col, out_col in [
        ("speaker", "target_same_speaker_context_3"),
        ("show", "target_same_show_context_3"),
    ]:
        sampled = []
        for _, row in merged.iterrows():
            pool = merged[(merged[group_col] == row[group_col]) & (merged["sample_id"] != row["sample_id"])]
            if pool.empty:
                pool = merged[merged["sample_id"] != row["sample_id"]]
            sampled_context = pool.iloc[int(rng.integers(0, len(pool)))]["context_3"]
            sampled.append(f"{row['target']} [SEP] {sampled_context}".strip())
        merged[out_col] = sampled
    return merged


def make_text_pipeline(seed: int) -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.95,
                    sublinear_tf=True,
                    max_features=12000,
                ),
            ),
        ]
    )


def prepare_features(
    text_by_condition: dict[str, np.ndarray],
    modalities: dict[str, np.ndarray],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    text_condition: str | None,
    dense_modalities: tuple[str, ...],
    seed: int,
) -> tuple[sparse.csr_matrix | np.ndarray, sparse.csr_matrix | np.ndarray]:
    parts_train = []
    parts_test = []
    if text_condition is not None:
        pipe = make_text_pipeline(seed)
        train_text = text_by_condition[text_condition][train_idx]
        test_text = text_by_condition[text_condition][test_idx]
        X_train_text = pipe.fit_transform(train_text)
        X_test_text = pipe.transform(test_text)
        parts_train.append(X_train_text)
        parts_test.append(X_test_text)

    if dense_modalities:
        dense_train = np.hstack([modalities[name][train_idx] for name in dense_modalities]).astype(np.float32)
        dense_test = np.hstack([modalities[name][test_idx] for name in dense_modalities]).astype(np.float32)
        scaler = StandardScaler()
        dense_train = scaler.fit_transform(dense_train)
        dense_test = scaler.transform(dense_test)
        parts_train.append(sparse.csr_matrix(dense_train))
        parts_test.append(sparse.csr_matrix(dense_test))

    if len(parts_train) == 1:
        return parts_train[0], parts_test[0]
    return sparse.hstack(parts_train, format="csr"), sparse.hstack(parts_test, format="csr")


def fit_predict(X_train, y_train, X_test, seed: int) -> tuple[np.ndarray, np.ndarray]:
    clf = LogisticRegression(
        class_weight="balanced",
        max_iter=3000,
        solver="liblinear",
        random_state=seed,
    )
    clf.fit(X_train, y_train)
    prob = clf.predict_proba(X_test)[:, 1]
    pred = (prob >= 0.5).astype(int)
    return pred, prob


def evaluate_groupkfold(
    conditions: list[tuple[str, str | None, tuple[str, ...]]],
    text_by_condition: dict[str, np.ndarray],
    modalities: dict[str, np.ndarray],
    y: np.ndarray,
    groups: np.ndarray,
    sample_ids: list[str],
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    pred_rows = []
    splitter = GroupKFold(n_splits=5)
    for condition_name, text_condition, dense_modalities in conditions:
        oof_pred = np.zeros_like(y)
        oof_prob = np.zeros(len(y), dtype=float)
        fold_scores = []
        for fold, (train_idx, test_idx) in enumerate(splitter.split(np.zeros(len(y)), y, groups=groups), start=1):
            X_train, X_test = prepare_features(
                text_by_condition,
                modalities,
                train_idx,
                test_idx,
                text_condition,
                dense_modalities,
                seed + fold,
            )
            pred, prob = fit_predict(X_train, y[train_idx], X_test, seed + fold)
            oof_pred[test_idx] = pred
            oof_prob[test_idx] = prob
            fold_metrics = binary_metrics(y[test_idx], pred, prob)
            fold_scores.append(fold_metrics["macro_f1"])
            rows.append(
                {
                    "protocol": "speaker_groupkfold",
                    "condition": condition_name,
                    "fold": fold,
                    "n_test": int(len(test_idx)),
                    **{k: v for k, v in fold_metrics.items() if k != "confusion_matrix"},
                }
            )
        pooled = binary_metrics(y, oof_pred, oof_prob)
        rows.append(
            {
                "protocol": "speaker_groupkfold_pooled",
                "condition": condition_name,
                "fold": 0,
                "n_test": int(len(y)),
                "fold_mean_macro_f1": float(np.mean(fold_scores)),
                "fold_sd_macro_f1": float(np.std(fold_scores, ddof=1)),
                **{k: v for k, v in pooled.items() if k != "confusion_matrix"},
            }
        )
        for idx, sid in enumerate(sample_ids):
            pred_rows.append(
                {
                    "condition": condition_name,
                    "sample_id": sid,
                    "label": int(y[idx]),
                    "pred": int(oof_pred[idx]),
                    "prob": float(oof_prob[idx]),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(pred_rows)


def evaluate_repeated_splits(
    conditions: list[tuple[str, str | None, tuple[str, ...]]],
    text_by_condition: dict[str, np.ndarray],
    modalities: dict[str, np.ndarray],
    y: np.ndarray,
    groups: np.ndarray,
    repeats: int,
    seed: int,
) -> pd.DataFrame:
    rows = []
    for repeat in range(repeats):
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed + repeat)
        train_idx, test_idx = next(splitter.split(np.zeros(len(y)), y, groups=groups))
        for condition_name, text_condition, dense_modalities in conditions:
            X_train, X_test = prepare_features(
                text_by_condition,
                modalities,
                train_idx,
                test_idx,
                text_condition,
                dense_modalities,
                seed + repeat,
            )
            pred, prob = fit_predict(X_train, y[train_idx], X_test, seed + repeat)
            metrics = binary_metrics(y[test_idx], pred, prob)
            rows.append(
                {
                    "repeat": repeat,
                    "condition": condition_name,
                    "n_train": int(len(train_idx)),
                    "n_test": int(len(test_idx)),
                    **{k: v for k, v in metrics.items() if k != "confusion_matrix"},
                }
            )
    return pd.DataFrame(rows)


def summarize_repeated(repeated: pd.DataFrame) -> pd.DataFrame:
    return (
        repeated.groupby("condition", as_index=False)
        .agg(
            repeats=("repeat", "nunique"),
            mean_macro_f1=("macro_f1", "mean"),
            sd_macro_f1=("macro_f1", "std"),
            mean_accuracy=("accuracy", "mean"),
            mean_roc_auc=("roc_auc", "mean"),
        )
        .sort_values("mean_macro_f1", ascending=False)
    )


def compute_deltas(repeated: pd.DataFrame) -> pd.DataFrame:
    wide = repeated.pivot(index="repeat", columns="condition", values="macro_f1")
    comparisons = [
        ("context_1_minus_target", "target_context_1_text", "target_text"),
        ("context_2_minus_target", "target_context_2_text", "target_text"),
        ("context_3_minus_target", "target_context_3_text", "target_text"),
        ("context_all_minus_target", "target_context_all_text", "target_text"),
        ("av_minus_target", "target_text_av", "target_text"),
        ("context3_av_minus_context3", "target_context_3_text_av", "target_context_3_text"),
        ("context3_av_minus_target_av", "target_context_3_text_av", "target_text_av"),
        ("true_context3_minus_random", "target_context_3_text", "target_random_context_3_text"),
        ("true_context3_minus_same_speaker", "target_context_3_text", "target_same_speaker_context_3_text"),
        ("true_context3_minus_same_show", "target_context_3_text", "target_same_show_context_3_text"),
    ]
    rows = []
    for name, a, b in comparisons:
        if a not in wide.columns or b not in wide.columns:
            continue
        delta = wide[a] - wide[b]
        rows.append(
            {
                "comparison": name,
                "mean_delta": float(delta.mean()),
                "sd_delta": float(delta.std(ddof=1)),
                "positive_rate": float((delta > 0).mean()),
                "negative_rate": float((delta < 0).mean()),
            }
        )
    return pd.DataFrame(rows)


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "(empty)"
    display = frame.copy()
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(lambda x: "" if pd.isna(x) else f"{x:.3f}")
    headers = [str(c) for c in display.columns]
    rows = [[str(x) for x in row] for row in display.to_numpy()]
    widths = [
        max(len(headers[i]), *(len(row[i]) for row in rows))
        for i in range(len(headers))
    ]
    lines = [
        "| " + " | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))) + " |",
        "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(headers))) + " |")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--text-embedding-npz", type=Path, default=Path("outputs/bert_regenerated/bert_cls_last4_by_sample_id.npz"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/context_av_pilot"))
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    modalities, y, sample_ids, metadata, excluded, notes = load_sarcasm_organized(args.data_dir, args.text_embedding_npz)
    context_table = build_context_table(args.data_dir, sample_ids, args.seed)
    context_table = add_group_random_contexts(context_table, metadata, args.seed)
    context_table = metadata[["sample_id", "speaker", "show", "label"]].merge(context_table, on="sample_id", how="left")

    condition_text_names = set(TEXT_CONDITIONS + [name for _, name, _ in SHUFFLE_CONDITIONS])
    text_by_condition = {
        name: context_table[name].fillna("").astype(str).to_numpy()
        for name in sorted(condition_text_names)
        if name in context_table.columns
    }

    groups = metadata["speaker"].astype(str).to_numpy()
    gkf_results, gkf_predictions = evaluate_groupkfold(
        MAIN_CONDITIONS + SHUFFLE_CONDITIONS[1:],
        text_by_condition,
        modalities,
        y,
        groups,
        sample_ids,
        args.seed,
    )
    repeated = evaluate_repeated_splits(
        MAIN_CONDITIONS + SHUFFLE_CONDITIONS[1:],
        text_by_condition,
        modalities,
        y,
        groups,
        args.repeats,
        args.seed,
    )
    repeated_summary = summarize_repeated(repeated)
    repeated_deltas = compute_deltas(repeated)

    context_table.to_csv(args.output_dir / "context_table.csv", index=False)
    gkf_results.to_csv(args.output_dir / "speaker_groupkfold_results.csv", index=False)
    gkf_predictions.to_csv(args.output_dir / "speaker_groupkfold_predictions.csv", index=False)
    repeated.to_csv(args.output_dir / "repeated_group_results.csv", index=False)
    repeated_summary.to_csv(args.output_dir / "repeated_group_summary.csv", index=False)
    repeated_deltas.to_csv(args.output_dir / "repeated_group_deltas.csv", index=False)
    pd.DataFrame(excluded).to_csv(args.output_dir / "excluded_samples.csv", index=False)
    (args.output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "data_dir": str(args.data_dir),
                "text_embedding_npz": str(args.text_embedding_npz),
                "notes": notes,
                "n_samples": int(len(y)),
                "repeats": int(args.repeats),
                "seed": int(args.seed),
                "text_representation": "fold-local TF-IDF unigram+bigram pilot, not final BERT context embeddings",
                "classifier": "LogisticRegression(class_weight='balanced', solver='liblinear', threshold=0.5)",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    pooled = gkf_results[gkf_results["protocol"] == "speaker_groupkfold_pooled"].copy()
    summary_lines = [
        "# Context and AV Pilot",
        "",
        "This is a CPU pilot using fold-local TF-IDF text features plus existing frozen audio/video tensors.",
        "It is intended to test whether the context-length and AV complement/substitution story is worth pursuing with BERT context embeddings.",
        "",
        "## Speaker GroupKFold pooled Macro-F1",
        "",
        markdown_table(pooled[["condition", "macro_f1", "accuracy", "roc_auc"]].sort_values("macro_f1", ascending=False)),
        "",
        "## Repeated grouped split summary",
        "",
        markdown_table(repeated_summary),
        "",
        "## Repeated grouped deltas",
        "",
        markdown_table(repeated_deltas),
    ]
    (args.output_dir / "summary.md").write_text("\n".join(summary_lines), encoding="utf-8")
    print(f"saved pilot outputs to {args.output_dir}")
    print(pooled[["condition", "macro_f1", "accuracy", "roc_auc"]].sort_values("macro_f1", ascending=False).head(12).to_string(index=False))
    print(repeated_deltas.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
