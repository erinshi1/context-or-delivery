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


def read_text_file(text_dir: Path, sample_id: str) -> str | None:
    path = text_dir / f"{sample_id}.txt"
    if not path.is_file():
        return None
    raw = path.read_text(encoding="utf-8", errors="replace")
    return raw.rsplit("\t", 1)[0] if "\t" in raw else raw


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/bert_regenerated"))
    parser.add_argument("--model", default="bert-base-uncased")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--source", choices=["metadata", "text_files"], default="metadata")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    meta_path = args.data_dir / "core" / "final_context_videos" / "SE-MUStARD-Final.csv"
    metadata = pd.read_csv(meta_path)
    labeled = metadata[metadata["SARCASM"].notna()].copy().reset_index(drop=True)
    labeled["sample_id"] = labeled["KEY"].astype(str)
    text_dir = args.data_dir / "core" / "texts"

    texts = []
    for _, row in labeled.iterrows():
        sample_id = str(row["sample_id"])
        if args.source == "text_files":
            text = read_text_file(text_dir, sample_id)
            if text is None:
                text = str(row["SENTENCE"])
        else:
            text = str(row["SENTENCE"])
        texts.append(text)

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=args.local_files_only)
    model = AutoModel.from_pretrained(args.model, output_hidden_states=True, local_files_only=args.local_files_only)
    model.eval()

    vectors = []
    with torch.no_grad():
        for start in range(0, len(texts), args.batch_size):
            batch_texts = texts[start : start + args.batch_size]
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            outputs = model(**encoded)
            hidden = outputs.hidden_states[-4:]
            cls = torch.stack([layer[:, 0, :] for layer in hidden], dim=0).mean(dim=0)
            vectors.append(cls.cpu().numpy().astype(np.float32))

    matrix = np.vstack(vectors).astype(np.float32)
    ids = labeled["sample_id"].astype(str).tolist()
    info = pd.DataFrame(
        {
            "sample_id": ids,
            "row_index": np.arange(len(ids)),
            "speaker": labeled["SPEAKER"].astype(str),
            "show": labeled["SHOW"].astype(str),
            "label": labeled["SARCASM"].astype(str).str.lower().map({"true": 1, "false": 0}),
            "text_source": args.source,
            "text_sha1": [sha1_text(text) for text in texts],
            "text": texts,
        }
    )
    np.savez_compressed(args.output_dir / "bert_cls_last4_by_sample_id.npz", sample_id=np.asarray(ids), embeddings=matrix)
    info.to_csv(args.output_dir / "bert_cls_last4_by_sample_id_metadata.csv", index=False)
    (args.output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "model": args.model,
                "source": args.source,
                "max_length": args.max_length,
                "batch_size": args.batch_size,
                "rows": len(ids),
                "embedding_dim": int(matrix.shape[1]),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"saved {matrix.shape} to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
