from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluate import binary_metrics
from src.sarcasm_data import load_sarcasm_organized
from work.run_context_bert_lr import load_context_embeddings, make_features


CONDITIONS = [
    ("target_text", "target", ()),
    ("context_3_target_text", "context_3_target", ()),
    ("context_all_target_text", "context_all_target", ()),
    ("target_text_tav", "target", ("audio", "video")),
    ("target_text_audio", "target", ("audio",)),
    ("target_text_video", "target", ("video",)),
    ("context_3_target_text_tav", "context_3_target", ("audio", "video")),
    ("context_3_target_text_audio", "context_3_target", ("audio",)),
    ("context_3_target_text_video", "context_3_target", ("video",)),
    ("context_all_target_text_tav", "context_all_target", ("audio", "video")),
]


def fit_predict_mlp(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    clf = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                MLPClassifier(
                    hidden_layer_sizes=(64,),
                    activation="relu",
                    alpha=1e-3,
                    batch_size=64,
                    learning_rate_init=1e-3,
                    max_iter=180,
                    early_stopping=True,
                    validation_fraction=0.15,
                    n_iter_no_change=12,
                    random_state=seed,
                ),
            ),
        ]
    )
    clf.fit(X_train, y_train)
    prob = clf.predict_proba(X_test)[:, 1]
    pred = (prob >= 0.5).astype(int)
    return pred, prob


def evaluate(
    context_embeddings: dict[str, np.ndarray],
    modalities: dict[str, np.ndarray],
    y: np.ndarray,
    groups: np.ndarray,
    sample_ids: list[str],
    seeds: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    pred_rows = []
    splitter = GroupKFold(n_splits=5)
    for seed in seeds:
        for condition_name, text_condition, dense_modalities in CONDITIONS:
            X = make_features(context_embeddings, modalities, text_condition, dense_modalities)
            oof_pred = np.zeros_like(y)
            oof_prob = np.zeros(len(y), dtype=float)
            fold_scores = []
            for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups=groups), start=1):
                pred, prob = fit_predict_mlp(X[train_idx], y[train_idx], X[test_idx], seed + fold)
                oof_pred[test_idx] = pred
                oof_prob[test_idx] = prob
                metrics = binary_metrics(y[test_idx], pred, prob)
                fold_scores.append(metrics["macro_f1"])
                rows.append(
                    {
                        "model": "mlp_64",
                        "seed": seed,
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
                    "model": "mlp_64",
                    "seed": seed,
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
                        "model": "mlp_64",
                        "seed": seed,
                        "condition": condition_name,
                        "sample_id": sid,
                        "label": int(y[idx]),
                        "pred": int(oof_pred[idx]),
                        "prob": float(oof_prob[idx]),
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(pred_rows)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    pooled = results[results["protocol"].eq("speaker_groupkfold_pooled")]
    summary = (
        pooled.groupby("condition", as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            mean_macro_f1=("macro_f1", "mean"),
            sd_macro_f1=("macro_f1", "std"),
            mean_accuracy=("accuracy", "mean"),
            mean_roc_auc=("roc_auc", "mean"),
        )
        .sort_values("mean_macro_f1", ascending=False)
    )
    wide = pooled.pivot(index="seed", columns="condition", values="macro_f1")
    comparisons = [
        ("target_tav_minus_target", "target_text_tav", "target_text"),
        ("target_audio_minus_target", "target_text_audio", "target_text"),
        ("target_video_minus_target", "target_text_video", "target_text"),
        ("c3_tav_minus_c3", "context_3_target_text_tav", "context_3_target_text"),
        ("c3_audio_minus_c3", "context_3_target_text_audio", "context_3_target_text"),
        ("c3_video_minus_c3", "context_3_target_text_video", "context_3_target_text"),
        ("all_tav_minus_all", "context_all_target_text_tav", "context_all_target_text"),
    ]
    delta_rows = []
    for name, a, b in comparisons:
        delta = wide[a] - wide[b]
        delta_rows.append(
            {
                "comparison": name,
                "mean_delta": float(delta.mean()),
                "sd_delta": float(delta.std(ddof=1)),
                "positive_rate": float((delta > 0).mean()),
            }
        )
    return summary, pd.DataFrame(delta_rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--base-text-embedding-npz", type=Path, default=Path("outputs/bert_regenerated/bert_cls_last4_by_sample_id.npz"))
    parser.add_argument("--context-embedding-dir", type=Path, default=Path("outputs/context_bert_embeddings"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/context_mlp_robustness"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2027])
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    modalities, y, sample_ids, metadata, excluded, notes = load_sarcasm_organized(args.data_dir, args.base_text_embedding_npz)
    context_embeddings, embedding_ids = load_context_embeddings(args.context_embedding_dir)
    if embedding_ids != sample_ids:
        raise ValueError("context embedding sample order does not match loaded modalities")
    groups = metadata["speaker"].astype(str).to_numpy()

    results, predictions = evaluate(context_embeddings, modalities, y, groups, sample_ids, args.seeds)
    summary, deltas = summarize(results)
    results.to_csv(args.output_dir / "speaker_groupkfold_mlp_results.csv", index=False)
    predictions.to_csv(args.output_dir / "speaker_groupkfold_mlp_predictions.csv", index=False)
    summary.to_csv(args.output_dir / "speaker_groupkfold_mlp_summary.csv", index=False)
    deltas.to_csv(args.output_dir / "speaker_groupkfold_mlp_deltas.csv", index=False)
    (args.output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "data_dir": str(args.data_dir),
                "base_text_embedding_npz": str(args.base_text_embedding_npz),
                "context_embedding_dir": str(args.context_embedding_dir),
                "seeds": args.seeds,
                "model": "MLPClassifier(hidden_layer_sizes=(64,), alpha=1e-3, early_stopping=True, threshold=0.5)",
                "n_samples": int(len(y)),
                "loader_notes": notes,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(summary.to_string(index=False))
    print(deltas.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
