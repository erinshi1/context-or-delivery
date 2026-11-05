from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def looks_like_sarcasm_organized(data_dir: Path) -> bool:
    return (
        (data_dir / "core" / "audio_tensor").is_dir()
        and (data_dir / "core" / "video_tensor").is_dir()
        and (data_dir / "core" / "final_context_videos" / "SE-MUStARD-Final.csv").is_file()
        and (data_dir / "embeddings" / "bert" / "bert-output.jsonl").is_file()
    )


def _mean_pool_tensor(path: Path) -> np.ndarray:
    try:
        tensor = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        tensor = torch.load(path, map_location="cpu")
    if not torch.is_tensor(tensor):
        tensor = torch.as_tensor(tensor)
    arr = tensor.detach().cpu().float()
    if arr.ndim == 1:
        return arr.numpy()
    return arr.reshape(-1, arr.shape[-1]).mean(dim=0).numpy()


def _load_tensor_dir(path: Path) -> dict[str, np.ndarray]:
    features = {}
    for file in sorted(path.glob("*.pkl")):
        if file.name.startswith("._"):
            continue
        features[file.stem] = _mean_pool_tensor(file).astype(np.float32)
    return features


def _load_bert_cls(path: Path) -> list[np.ndarray]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            cls_feature = obj["features"][0]
            layer_values = [layer["values"] for layer in cls_feature["layers"]]
            values = np.asarray(layer_values, dtype=np.float32).mean(axis=0)
            rows.append(values)
    return rows


def _read_texts(path: Path) -> dict[str, str]:
    texts = {}
    for file in sorted(path.glob("*.txt")):
        if file.name.startswith("._"):
            continue
        raw = file.read_text(encoding="utf-8", errors="replace")
        text = raw.rsplit("\t", 1)[0] if "\t" in raw else raw
        texts[file.stem] = text
    return texts


def _load_regenerated_bert_npz(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=False)
    ids = [str(x) for x in data["sample_id"].tolist()]
    embeddings = data["embeddings"].astype(np.float32)
    return {sid: embeddings[i] for i, sid in enumerate(ids)}


def load_sarcasm_organized(
    data_dir: Path,
    text_embedding_npz: Path | None = None,
) -> tuple[dict[str, np.ndarray], np.ndarray, list[str], pd.DataFrame, list[dict], dict]:
    core = data_dir / "core"
    meta_path = core / "final_context_videos" / "SE-MUStARD-Final.csv"
    metadata_all = pd.read_csv(meta_path)
    labeled = metadata_all[metadata_all["SARCASM"].notna()].copy().reset_index(drop=True)
    labeled["sample_id"] = labeled["KEY"].astype(str)
    labeled["label"] = labeled["SARCASM"].astype(str).str.lower().map({"true": 1, "false": 0}).astype(int)
    labeled["dialogue_id"] = labeled["KEY"].astype(str)
    labeled["speaker"] = labeled["SPEAKER"].astype(str)
    labeled["show"] = labeled["SHOW"].astype(str)
    labeled["utterance"] = labeled["SENTENCE"].astype(str)

    bert_rows = _load_bert_cls(data_dir / "embeddings" / "bert" / "bert-output.jsonl")
    bert_warning = None
    if text_embedding_npz is not None:
        bert_by_id = _load_regenerated_bert_npz(text_embedding_npz)
        text_embedding_note = f"ID-bearing regenerated BERT [CLS] vectors from {text_embedding_npz}"
    elif len(bert_rows) == len(labeled):
        bert_by_id = dict(zip(labeled["sample_id"], bert_rows))
        text_embedding_note = "BERT JSONL [CLS] vector averaged over the saved last four layers from embeddings/bert/bert-output.jsonl"
    else:
        bert_by_id = {}
        bert_warning = f"BERT JSONL rows ({len(bert_rows)}) do not match labeled metadata rows ({len(labeled)})."
        text_embedding_note = "BERT JSONL could not be aligned"

    audio_by_id = _load_tensor_dir(core / "audio_tensor")
    video_by_id = _load_tensor_dir(core / "video_tensor")
    text_by_id = _read_texts(core / "texts")

    candidate_ids = set(labeled["sample_id"])
    available_ids = candidate_ids & set(audio_by_id) & set(video_by_id)
    if bert_by_id:
        available_ids &= set(bert_by_id)
    sample_ids = [sid for sid in labeled["sample_id"].tolist() if sid in available_ids]

    excluded = []
    for sid in labeled["sample_id"]:
        reasons = []
        if sid not in audio_by_id:
            reasons.append("missing_audio_tensor")
        if sid not in video_by_id:
            reasons.append("missing_video_tensor")
        if bert_by_id and sid not in bert_by_id:
            reasons.append("missing_bert_embedding")
        if not bert_by_id:
            reasons.append("bert_unaligned")
        if reasons:
            excluded.append({"sample_id": sid, "reason": ";".join(reasons)})

    meta = labeled.set_index("sample_id").loc[sample_ids].reset_index()
    text_matrix = np.vstack([bert_by_id[sid] for sid in sample_ids]).astype(np.float32)
    audio_matrix = np.vstack([audio_by_id[sid] for sid in sample_ids]).astype(np.float32)
    video_matrix = np.vstack([video_by_id[sid] for sid in sample_ids]).astype(np.float32)
    y = meta["label"].to_numpy(dtype=int)
    modalities = {"text": text_matrix, "audio": audio_matrix, "video": video_matrix}
    meta["source_text"] = [text_by_id.get(sid, "") for sid in sample_ids]
    notes = {
        "dataset_loader": "sarcasm_organized",
        "text_embedding": text_embedding_note,
        "audio_embedding": "mean pooled Wav2Vec2 tensor over sequence dimension",
        "video_embedding": "mean pooled TimeSformer tensor over sequence dimension",
        "metadata_labeled_rows": int(len(labeled)),
        "bert_jsonl_rows": int(len(bert_rows)),
        "bert_warning": bert_warning,
        "text_embedding_npz": str(text_embedding_npz) if text_embedding_npz is not None else None,
    }
    return modalities, y, sample_ids, meta, excluded, notes
