"""Recompute the short-paper tables and paired uncertainty from saved results."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
TEXT = ["target_text"] + [f"context_{k}_target_text" for k in (1, 2, 3)] + ["context_all_target_text"]
CONTEXTS = ["target", "context_1_target", "context_2_target", "context_3_target", "context_all_target"]
CONTROLS = ["target_context_3_text", "target_random_context_3_text", "target_same_speaker_context_3_text", "target_same_show_context_3_text"]
COMPARISONS = [
    ("All context - target", TEXT[4], TEXT[0], 0),
    ("C3 - target", TEXT[3], TEXT[0], 1),
    ("Target T+A+V - target", "target_text_av", TEXT[0], 2),
    ("A+V after C1", TEXT[1] + "_av", TEXT[1], 5),
    ("A+V after C2", TEXT[2] + "_av", TEXT[2], 6),
    ("A+V after C3", TEXT[3] + "_av", TEXT[3], 11),
    ("A+V after all context", TEXT[4] + "_av", TEXT[4], 14),
    ("Target-first C3 - random", CONTROLS[0], CONTROLS[1], 19),
    ("Target-first C3 - same speaker", CONTROLS[0], CONTROLS[2], 20),
    ("Target-first C3 - same show", CONTROLS[0], CONTROLS[3], 21),
]


def macro_f1(y, pred):
    y, pred = np.asarray(y), np.asarray(pred)
    return float(np.mean([2 * np.sum((y == c) & (pred == c)) /
                          max(1, np.sum(y == c) + np.sum(pred == c)) for c in (0, 1)]))


def cluster_ci(y, a, b, speakers, seed, count):
    units = np.unique(speakers)
    # Aggregate confusion counts per speaker before resampling whole speakers.
    counts = np.array([[[np.sum((speakers == s) & (y == c) & (p == c)),
                         np.sum((speakers == s) & (y == c)) + np.sum((speakers == s) & (p == c))]
                        for p in (a, b) for c in (0, 1)] for s in units])
    chosen = np.random.default_rng(seed).integers(0, len(units), (count, len(units)))
    total = counts[chosen].sum(axis=1)
    scores = np.divide(2 * total[:, :, 0], total[:, :, 1],
                       out=np.zeros_like(total[:, :, 0], dtype=float), where=total[:, :, 1] != 0)
    delta = scores[:, :2].mean(axis=1) - scores[:, 2:].mean(axis=1)
    return np.percentile(delta, [2.5, 97.5])


def validate(manifest, predictions):
    if manifest.sample_id.duplicated().any() or len(manifest) != 684:
        raise ValueError("Expected 684 unique target IDs")
    if manifest.label.value_counts().to_dict() != {0: 344, 1: 340}:
        raise ValueError("Unexpected label balance")
    for condition, part in predictions.groupby("condition", sort=False):
        if part.sample_id.duplicated().any() or set(part.sample_id) != set(manifest.sample_id):
            raise ValueError(f"Invalid prediction IDs for {condition}")
        check = manifest[["sample_id", "label"]].merge(part[["sample_id", "label"]], on="sample_id")
        if not (check.label_x == check.label_y).all():
            raise ValueError(f"Mismatched labels for {condition}")
    folds = pd.read_csv(ROOT / "fold_assignments.csv", dtype={"sample_id": str})
    if folds.sample_id.duplicated().any() or set(folds.sample_id) != set(manifest.sample_id):
        raise ValueError("Invalid fold assignments")
    joined = manifest.merge(folds, on="sample_id", validate="one_to_one")
    if joined.groupby("speaker").fold.nunique().max() != 1:
        raise ValueError("Speaker crosses folds")
    repeat = pd.read_csv(ROOT / "repeated_split_assignments.csv", dtype={"sample_id": str})
    for _, part in repeat.groupby("seed"):
        if part.sample_id.duplicated().any() or set(part.sample_id) != set(manifest.sample_id):
            raise ValueError("Invalid repeated partition")
        part = part.merge(manifest[["sample_id", "speaker"]], on="sample_id")
        if part.groupby("speaker")["partition"].nunique().max() != 1:
            raise ValueError("Speaker crosses repeated partition")
    for path in sorted((ROOT / "features").glob("*.npz")):
        z = np.load(path, allow_pickle=False)
        if list(z["sample_id"].astype(str)) != list(manifest.sample_id):
            raise ValueError(f"Feature IDs differ in {path.name}")
        if z["embeddings"].shape != (684, 768) or not np.isfinite(z["embeddings"]).all():
            raise ValueError(f"Invalid feature matrix {path.name}")


def refit_text(manifest, output):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    folds = pd.read_csv(ROOT / "fold_assignments.csv", dtype={"sample_id": str}).set_index("sample_id")
    fold = folds.loc[manifest.sample_id, "fold"].to_numpy()
    y = manifest.label.to_numpy()
    rows = []
    conditions = list(zip(TEXT, CONTEXTS)) + [(c, c.removesuffix("_text")) for c in CONTROLS]
    for condition, feature in conditions:
        X = np.load(ROOT / "features" / f"{feature}_bert_cls_last4.npz", allow_pickle=False)["embeddings"]
        pred = np.zeros(len(y), dtype=int)
        for k in range(1, 6):
            test = fold == k
            model = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced",
                                   solver="lbfgs", C=1, max_iter=5000, random_state=2027 + k))
            model.fit(X[~test], y[~test])
            pred[test] = (model.predict_proba(X[test])[:, 1] >= 0.5).astype(int)
        rows.append({"condition": condition, "macro_f1": macro_f1(y, pred)})
    pd.DataFrame(rows).to_csv(output / "text_refit.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reproduced")
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--refit-text", action="store_true")
    args = parser.parse_args()
    if args.bootstrap < 1:
        parser.error("--bootstrap must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(ROOT / "sample_manifest.csv", dtype={"sample_id": str})
    pred = pd.read_csv(ROOT / "results/lr/speaker_groupkfold_predictions.csv", dtype={"sample_id": str})
    validate(manifest, pred)
    wide = pred.pivot(index="sample_id", columns="condition", values="pred").loc[manifest.sample_id]
    y, speakers = manifest.label.to_numpy(), manifest.speaker.astype(str).to_numpy()
    scores = {c: macro_f1(y, wide[c]) for c in wide}
    main_rows, modality_rows = [], []
    for i, c in enumerate(TEXT):
        av = "target_text_av" if i == 0 else c + "_av"
        audio = "target_text_audio" if i == 0 else c + "_audio"
        video = "target_text_video" if i == 0 else c + "_video"
        label = ["Target only", "C1 + target", "C2 + target", "C3 + target", "All context + target"][i]
        main_rows.append({"context": label, "text": scores[c], "text_av": scores[av], "delta_av": scores[av] - scores[c]})
        modality_rows.append({"context": label, "text_audio": scores[audio], "delta_audio": scores[audio] - scores[c],
                              "text_video": scores[video], "delta_video": scores[video] - scores[c]})
    pd.DataFrame(main_rows).to_csv(args.output_dir / "table_1_context_modality.csv", index=False)
    pd.DataFrame(modality_rows).to_csv(args.output_dir / "table_2_audio_video.csv", index=False)
    mlp = pd.read_csv(ROOT / "results/mlp/speaker_groupkfold_mlp_predictions.csv", dtype={"sample_id": str})
    mlp_rows = []
    for c, part in mlp.groupby("condition"):
        values = [macro_f1(g.label.to_numpy(), g.pred.to_numpy()) for _, g in part.groupby("seed")]
        mlp_rows.append({"condition": c, "mean_macro_f1": np.mean(values), "sd_macro_f1": np.std(values, ddof=1)})
    pd.DataFrame(mlp_rows).to_csv(args.output_dir / "table_3_mlp.csv", index=False)
    repeated = pd.read_csv(ROOT / "results/lr/repeated_group_results.csv")
    rw = repeated.pivot(index="repeat", columns="condition", values="macro_f1")
    repeat_rows = []
    for label, a, b, _ in COMPARISONS:
        if label == "C3 - target":
            continue
        d = rw[a] - rw[b]
        repeat_rows.append({"comparison": label, "mean_delta": d.mean(), "sd_delta": d.std(ddof=1), "positive_rate": (d > 0).mean()})
    pd.DataFrame(repeat_rows).to_csv(args.output_dir / "table_4_repeated_splits.csv", index=False)
    ci_rows = []
    for label, a, b, offset in COMPARISONS:
        low, high = cluster_ci(y, wide[a].to_numpy(), wide[b].to_numpy(), speakers, 2127 + offset, args.bootstrap)
        ci_rows.append({"comparison": label, "condition_a": a, "condition_b": b, "delta": scores[a] - scores[b],
                        "ci_low": low, "ci_high": high, "resamples": args.bootstrap, "bootstrap_seed": 2127 + offset})
    pd.DataFrame(ci_rows).to_csv(args.output_dir / "paired_speaker_bootstrap.csv", index=False)
    if args.refit_text:
        refit_text(manifest, args.output_dir)
    print("Verified 684 targets, label alignment, 16 feature matrices, and speaker-separated partitions.")
    print(pd.DataFrame(main_rows).round(3).to_string(index=False))
    print(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
