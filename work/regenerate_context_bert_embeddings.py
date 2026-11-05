from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.sarcasm_data import load_sarcasm_organized
from work.run_context_av_pilot import build_context_table, add_group_random_contexts


DEFAULT_CONDITIONS = [
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


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()


def encode_texts(
    texts: list[str],
    tokenizer,
    model,
    batch_size: int,
    max_length: int,
    device: torch.device,
) -> np.ndarray:
    vectors = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            outputs = model(**encoded)
            hidden = outputs.hidden_states[-4:]
            cls = torch.stack([layer[:, 0, :] for layer in hidden], dim=0).mean(dim=0)
            vectors.append(cls.cpu().numpy().astype(np.float32))
    return np.vstack(vectors).astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--base-text-embedding-npz", type=Path, default=Path("outputs/bert_regenerated/bert_cls_last4_by_sample_id.npz"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/context_bert_embeddings"))
    parser.add_argument("--model", default="bert-base-uncased")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    modalities, y, sample_ids, metadata, excluded, notes = load_sarcasm_organized(args.data_dir, args.base_text_embedding_npz)
    context_table = build_context_table(args.data_dir, sample_ids, args.seed)
    context_table = add_group_random_contexts(context_table, metadata, args.seed)
    context_table = metadata[["sample_id", "speaker", "show", "label"]].merge(context_table, on="sample_id", how="left")
    for n in [1, 2, 3]:
        context_table[f"context_{n}_target"] = (
            context_table[f"context_{n}"].fillna("").astype(str) + " [SEP] " + context_table["target"].fillna("").astype(str)
        ).str.strip()
    context_table["context_all_target"] = (
        context_table["context_all"].fillna("").astype(str) + " [SEP] " + context_table["target"].fillna("").astype(str)
    ).str.strip()
    context_table.to_csv(args.output_dir / "context_table.csv", index=False)
    pd.DataFrame(excluded).to_csv(args.output_dir / "excluded_samples.csv", index=False)

    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=args.local_files_only)
    model = AutoModel.from_pretrained(args.model, output_hidden_states=True, local_files_only=args.local_files_only)
    model.to(device)
    model.eval()

    saved = []
    for condition in DEFAULT_CONDITIONS:
        out_path = args.output_dir / f"{condition}_bert_cls_last4.npz"
        if out_path.is_file():
            data = np.load(out_path, allow_pickle=False)
            saved.append({"condition": condition, "path": str(out_path), "shape": list(data["embeddings"].shape), "skipped_existing": True})
            continue
        texts = context_table[condition].fillna("").astype(str).tolist()
        matrix = encode_texts(texts, tokenizer, model, args.batch_size, args.max_length, device)
        np.savez_compressed(
            out_path,
            sample_id=np.asarray(sample_ids),
            embeddings=matrix,
            condition=np.asarray([condition]),
        )
        saved.append({"condition": condition, "path": str(out_path), "shape": list(matrix.shape)})

    hashes = []
    for condition in DEFAULT_CONDITIONS:
        for sid, text in zip(sample_ids, context_table[condition].fillna("").astype(str)):
            hashes.append({"sample_id": sid, "condition": condition, "text_sha1": sha1_text(text), "text": text})
    pd.DataFrame(hashes).to_csv(args.output_dir / "text_hashes.csv", index=False)

    (args.output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "data_dir": str(args.data_dir),
                "base_text_embedding_npz": str(args.base_text_embedding_npz),
                "model": args.model,
                "max_length": args.max_length,
                "batch_size": args.batch_size,
                "device": str(device),
                "seed": args.seed,
                "n_samples": int(len(sample_ids)),
                "conditions": DEFAULT_CONDITIONS,
                "saved": saved,
                "loader_notes": notes,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"saved {len(saved)} condition embeddings for {len(sample_ids)} samples to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
