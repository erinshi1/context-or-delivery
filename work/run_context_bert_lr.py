from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluate import binary_metrics
from src.sarcasm_data import load_sarcasm_organized
from work.run_context_av_pilot import markdown_table


BERT_CONDITIONS = [
    "target",
    "context_1",
    "context_2",
    "context_3",
    "context_all",
    "target_context_1",
    "target_context_2",
    "target_context_3",
    "target_context_all",
    "context_1_target",
    "context_2_target",
    "context_3_target",
    "context_all_target",
    "target_random_context_3",
    "target_same_speaker_context_3",
    "target_same_show_context_3",
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
    ("context_1_target_text", "context_1_target", ()),
    ("context_2_target_text", "context_2_target", ()),
    ("context_3_target_text", "context_3_target", ()),
    ("context_all_target_text", "context_all_target", ()),
    ("target_text_av", "target", ("audio", "video")),
    ("target_text_audio", "target", ("audio",)),
    ("target_text_video", "target", ("video",)),
    ("target_context_1_text_av", "target_context_1", ("audio", "video")),
    ("target_context_2_text_av", "target_context_2", ("audio", "video")),
    ("target_context_3_text_av", "target_context_3", ("audio", "video")),
    ("target_context_all_text_av", "target_context_all", ("audio", "video")),
    ("context_1_target_text_av", "context_1_target", ("audio", "video")),
    ("context_2_target_text_av", "context_2_target", ("audio", "video")),
    ("context_3_target_text_av", "context_3_target", ("audio", "video")),
    ("context_all_target_text_av", "context_all_target", ("audio", "video")),
    ("context_1_target_text_audio", "context_1_target", ("audio",)),
    ("context_1_target_text_video", "context_1_target", ("video",)),
    ("context_2_target_text_audio", "context_2_target", ("audio",)),
    ("context_2_target_text_video", "context_2_target", ("video",)),
    ("context_3_target_text_audio", "context_3_target", ("audio",)),
    ("context_3_target_text_video", "context_3_target", ("video",)),
    ("context_all_target_text_audio", "context_all_target", ("audio",)),
    ("context_all_target_text_video", "context_all_target", ("video",)),
    ("audio_only", None, ("audio",)),
    ("video_only", None, ("video",)),
    ("av_only", None, ("audio", "video")),
    ("target_random_context_3_text", "target_random_context_3", ()),
    ("target_same_speaker_context_3_text", "target_same_speaker_context_3", ()),
    ("target_same_show_context_3_text", "target_same_show_context_3", ()),
]

LATE_CONDITIONS = [
    ("target_late_text_av", "target"),
    ("context3_late_text_av", "target_context_3"),
    ("contextall_late_text_av", "target_context_all"),
    ("natural_context3_late_text_av", "context_3_target"),
    ("natural_contextall_late_text_av", "context_all_target"),
]


def load_context_embeddings(embedding_dir: Path) -> tuple[dict[str, np.ndarray], list[str]]:
    embeddings: dict[str, np.ndarray] = {}
    sample_ids: list[str] | None = None
    for condition in BERT_CONDITIONS:
        path = embedding_dir / f"{condition}_bert_cls_last4.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
        data = np.load(path, allow_pickle=False)
        ids = [str(x) for x in data["sample_id"].tolist()]
        if sample_ids is None:
            sample_ids = ids
        elif ids != sample_ids:
            raise ValueError(f"sample_id order mismatch for {condition}")
        embeddings[condition] = data["embeddings"].astype(np.float32)
    assert sample_ids is not None
    return embeddings, sample_ids


def train_logreg(X_train, y_train, X_test, seed: int) -> tuple[np.ndarray, np.ndarray]:
    clf = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=5000,
                    solver="lbfgs",
                    random_state=seed,
                ),
            ),
        ]
    )
    clf.fit(X_train, y_train)
    prob = clf.predict_proba(X_test)[:, 1]
    pred = (prob >= 0.5).astype(int)
    return pred, prob


def make_features(
    context_embeddings: dict[str, np.ndarray],
    modalities: dict[str, np.ndarray],
    text_condition: str | None,
    dense_modalities: tuple[str, ...],
) -> np.ndarray:
    parts = []
    if text_condition is not None:
        parts.append(context_embeddings[text_condition])
    for name in dense_modalities:
        parts.append(modalities[name])
    if not parts:
        raise ValueError("empty feature condition")
    return np.hstack(parts).astype(np.float32)


def evaluate_groupkfold(
    conditions: list[tuple[str, str | None, tuple[str, ...]]],
    context_embeddings: dict[str, np.ndarray],
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
        X = make_features(context_embeddings, modalities, text_condition, dense_modalities)
        oof_pred = np.zeros_like(y)
        oof_prob = np.zeros(len(y), dtype=float)
        fold_scores = []
        for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups=groups), start=1):
            pred, prob = train_logreg(X[train_idx], y[train_idx], X[test_idx], seed + fold)
            oof_pred[test_idx] = pred
            oof_prob[test_idx] = prob
            metrics = binary_metrics(y[test_idx], pred, prob)
            fold_scores.append(metrics["macro_f1"])
            rows.append(
                {
                    "protocol": "speaker_groupkfold",
                    "condition": condition_name,
                    "fold": fold,
                    "n_test": int(len(test_idx)),
                    **{k: v for k, v in metrics.items() if k != "confusion_matrix"},
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


def evaluate_late_groupkfold(
    context_embeddings: dict[str, np.ndarray],
    modalities: dict[str, np.ndarray],
    y: np.ndarray,
    groups: np.ndarray,
    seed: int,
) -> pd.DataFrame:
    rows = []
    av = make_features(context_embeddings, modalities, None, ("audio", "video"))
    splitter = GroupKFold(n_splits=5)
    for condition_name, text_condition in LATE_CONDITIONS:
        text = make_features(context_embeddings, modalities, text_condition, ())
        oof_pred = np.zeros_like(y)
        oof_prob = np.zeros(len(y), dtype=float)
        for fold, (train_idx, test_idx) in enumerate(splitter.split(text, y, groups=groups), start=1):
            _, text_prob = train_logreg(text[train_idx], y[train_idx], text[test_idx], seed + fold)
            _, av_prob = train_logreg(av[train_idx], y[train_idx], av[test_idx], seed + fold)
            prob = (text_prob + av_prob) / 2
            pred = (prob >= 0.5).astype(int)
            oof_pred[test_idx] = pred
            oof_prob[test_idx] = prob
            metrics = binary_metrics(y[test_idx], pred, prob)
            rows.append(
                {
                    "protocol": "speaker_groupkfold",
                    "condition": condition_name,
                    "fold": fold,
                    "n_test": int(len(test_idx)),
                    **{k: v for k, v in metrics.items() if k != "confusion_matrix"},
                }
            )
        pooled = binary_metrics(y, oof_pred, oof_prob)
        rows.append(
            {
                "protocol": "speaker_groupkfold_pooled",
                "condition": condition_name,
                "fold": 0,
                "n_test": int(len(y)),
                **{k: v for k, v in pooled.items() if k != "confusion_matrix"},
            }
        )
    return pd.DataFrame(rows)


def evaluate_repeated(
    conditions: list[tuple[str, str | None, tuple[str, ...]]],
    context_embeddings: dict[str, np.ndarray],
    modalities: dict[str, np.ndarray],
    y: np.ndarray,
    groups: np.ndarray,
    repeats: int,
    seed: int,
) -> pd.DataFrame:
    feature_cache = {
        condition_name: make_features(context_embeddings, modalities, text_condition, dense_modalities)
        for condition_name, text_condition, dense_modalities in conditions
    }
    rows = []
    for repeat in range(repeats):
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed + repeat)
        train_idx, test_idx = next(splitter.split(np.zeros(len(y)), y, groups=groups))
        for condition_name, X in feature_cache.items():
            pred, prob = train_logreg(X[train_idx], y[train_idx], X[test_idx], seed + repeat)
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


def evaluate_late_repeated(
    context_embeddings: dict[str, np.ndarray],
    modalities: dict[str, np.ndarray],
    y: np.ndarray,
    groups: np.ndarray,
    repeats: int,
    seed: int,
) -> pd.DataFrame:
    av = make_features(context_embeddings, modalities, None, ("audio", "video"))
    text_cache = {
        condition_name: make_features(context_embeddings, modalities, text_condition, ())
        for condition_name, text_condition in LATE_CONDITIONS
    }
    rows = []
    for repeat in range(repeats):
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed + repeat)
        train_idx, test_idx = next(splitter.split(np.zeros(len(y)), y, groups=groups))
        for condition_name, text in text_cache.items():
            _, text_prob = train_logreg(text[train_idx], y[train_idx], text[test_idx], seed + repeat)
            _, av_prob = train_logreg(av[train_idx], y[train_idx], av[test_idx], seed + repeat)
            prob = (text_prob + av_prob) / 2
            pred = (prob >= 0.5).astype(int)
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
        ("natural_context_1_minus_target", "context_1_target_text", "target_text"),
        ("natural_context_2_minus_target", "context_2_target_text", "target_text"),
        ("natural_context_3_minus_target", "context_3_target_text", "target_text"),
        ("natural_context_all_minus_target", "context_all_target_text", "target_text"),
        ("av_minus_target", "target_text_av", "target_text"),
        ("audio_minus_target", "target_text_audio", "target_text"),
        ("video_minus_target", "target_text_video", "target_text"),
        ("natural_context1_av_minus_context1", "context_1_target_text_av", "context_1_target_text"),
        ("natural_context2_av_minus_context2", "context_2_target_text_av", "context_2_target_text"),
        ("natural_context1_audio_minus_context1", "context_1_target_text_audio", "context_1_target_text"),
        ("natural_context1_video_minus_context1", "context_1_target_text_video", "context_1_target_text"),
        ("natural_context2_audio_minus_context2", "context_2_target_text_audio", "context_2_target_text"),
        ("natural_context2_video_minus_context2", "context_2_target_text_video", "context_2_target_text"),
        ("context3_av_minus_context3", "target_context_3_text_av", "target_context_3_text"),
        ("natural_context3_av_minus_context3", "context_3_target_text_av", "context_3_target_text"),
        ("natural_context3_audio_minus_context3", "context_3_target_text_audio", "context_3_target_text"),
        ("natural_context3_video_minus_context3", "context_3_target_text_video", "context_3_target_text"),
        ("natural_contextall_av_minus_contextall", "context_all_target_text_av", "context_all_target_text"),
        ("natural_contextall_audio_minus_contextall", "context_all_target_text_audio", "context_all_target_text"),
        ("natural_contextall_video_minus_contextall", "context_all_target_text_video", "context_all_target_text"),
        ("context3_av_minus_target_av", "target_context_3_text_av", "target_text_av"),
        ("natural_context3_av_minus_target_av", "context_3_target_text_av", "target_text_av"),
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--base-text-embedding-npz", type=Path, default=Path("outputs/bert_regenerated/bert_cls_last4_by_sample_id.npz"))
    parser.add_argument("--context-embedding-dir", type=Path, default=Path("outputs/context_bert_embeddings"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/context_bert_lr"))
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    modalities, y, sample_ids, metadata, excluded, notes = load_sarcasm_organized(args.data_dir, args.base_text_embedding_npz)
    context_embeddings, embedding_ids = load_context_embeddings(args.context_embedding_dir)
    if embedding_ids != sample_ids:
        raise ValueError("context embedding sample order does not match loaded modalities")
    groups = metadata["speaker"].astype(str).to_numpy()

    gkf_results, gkf_predictions = evaluate_groupkfold(MAIN_CONDITIONS, context_embeddings, modalities, y, groups, sample_ids, args.seed)
    late_gkf = evaluate_late_groupkfold(context_embeddings, modalities, y, groups, args.seed)
    repeated = evaluate_repeated(MAIN_CONDITIONS, context_embeddings, modalities, y, groups, args.repeats, args.seed)
    late_repeated = evaluate_late_repeated(context_embeddings, modalities, y, groups, args.repeats, args.seed)
    repeated_summary = summarize_repeated(repeated)
    late_repeated_summary = summarize_repeated(late_repeated)
    repeated_deltas = compute_deltas(repeated)

    gkf_results.to_csv(args.output_dir / "speaker_groupkfold_results.csv", index=False)
    gkf_predictions.to_csv(args.output_dir / "speaker_groupkfold_predictions.csv", index=False)
    late_gkf.to_csv(args.output_dir / "late_equal_groupkfold_results.csv", index=False)
    repeated.to_csv(args.output_dir / "repeated_group_results.csv", index=False)
    late_repeated.to_csv(args.output_dir / "late_equal_repeated_results.csv", index=False)
    repeated_summary.to_csv(args.output_dir / "repeated_group_summary.csv", index=False)
    late_repeated_summary.to_csv(args.output_dir / "late_equal_repeated_summary.csv", index=False)
    repeated_deltas.to_csv(args.output_dir / "repeated_group_deltas.csv", index=False)
    pd.DataFrame(excluded).to_csv(args.output_dir / "excluded_samples.csv", index=False)
    (args.output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "data_dir": str(args.data_dir),
                "base_text_embedding_npz": str(args.base_text_embedding_npz),
                "context_embedding_dir": str(args.context_embedding_dir),
                "n_samples": int(len(y)),
                "repeats": int(args.repeats),
                "seed": int(args.seed),
                "classifier": "LogisticRegression(class_weight='balanced', solver='lbfgs', threshold=0.5)",
                "loader_notes": notes,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    pooled = gkf_results[gkf_results["protocol"] == "speaker_groupkfold_pooled"]
    late_pooled = late_gkf[late_gkf["protocol"] == "speaker_groupkfold_pooled"]
    summary = [
        "# BERT Context and AV Logistic Regression",
        "",
        "Text features are BERT [CLS] vectors averaged over the last four hidden layers for each context condition.",
        "Audio and video are the existing mean-pooled Wav2Vec2 and TimeSformer tensors.",
        "",
        "## Speaker GroupKFold pooled Macro-F1",
        "",
        markdown_table(pooled[["condition", "macro_f1", "accuracy", "roc_auc"]].sort_values("macro_f1", ascending=False)),
        "",
        "## Repeated speaker-grouped split summary",
        "",
        markdown_table(repeated_summary),
        "",
        "## Repeated grouped deltas",
        "",
        markdown_table(repeated_deltas),
        "",
        "## Late equal fusion, GroupKFold pooled",
        "",
        markdown_table(late_pooled[["condition", "macro_f1", "accuracy", "roc_auc"]].sort_values("macro_f1", ascending=False)),
        "",
        "## Late equal fusion, repeated split summary",
        "",
        markdown_table(late_repeated_summary),
    ]
    (args.output_dir / "summary.md").write_text("\n".join(summary), encoding="utf-8")

    print(f"saved BERT LR outputs to {args.output_dir}")
    print(pooled[["condition", "macro_f1", "accuracy", "roc_auc"]].sort_values("macro_f1", ascending=False).head(12).to_string(index=False))
    print(repeated_deltas.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
