# Context or Delivery?

Reproduction materials for **Context or Delivery? A Controlled Study of Conversational and Audiovisual Cues for Sarcasm Detection**, IEEE ISM short paper, Yuanyuan Shi.

The study compares conversational text context with target-level audio and video on 684 aligned MUStARD-derived samples, using speaker-separated evaluation and frozen representations.

## Included materials

- `features/`: 16 sample-ID-indexed frozen BERT feature matrices, including both input orders and replacement-context controls.
- `sample_manifest.csv`: labels, speakers, shows, context lengths, and transcript hashes for the 684 targets.
- `fold_assignments.csv` and `repeated_split_assignments.csv`: the exact five-fold and 20 repeated speaker partitions.
- `results/lr/` and `results/mlp/`: saved metrics and out-of-fold predictions from the reported experiments.
- `results/paper/`: paper tables and paired speaker-cluster intervals reconstructed by `reproduce.py`.
- `work/` and `src/`: the experiment and feature-generation scripts used in the study.
- `config/experiment.json`: seeds, model settings, and feature provenance.

## Recompute tables and uncertainty on CPU

Python 3.10 or newer is recommended. Run from the repository root:

```bash
python -m pip install -r requirements.txt
python reproduce.py
```

This validates sample alignment and speaker separation, recomputes pooled Macro-F1 and the four paper tables from saved predictions/results, and generates 5,000-resample paired speaker-cluster percentile intervals. It does not refit models during bootstrap. Outputs go to `reproduced/`.

To refit the five text-only context conditions and the four target-first control conditions from the included BERT matrices:

```bash
python -m pip install -r requirements-training.txt
python reproduce.py --refit-text
```

Use `--bootstrap 200` for a quick software check; the paper uses 5,000 resamples. Exact refitting scores can vary with numerical libraries and scikit-learn versions. The supplied OOF predictions preserve the evaluated decisions.

## Full audiovisual training

The saved audiovisual predictions and results are included, but the original Wav2Vec2 and TimeSformer sequence tensors are not in this release. This repository supports independent statistical reanalysis and text-head refitting; it does not currently provide all inputs needed to refit the audiovisual systems or reconstruct upstream audiovisual extraction.

The original scripts accept the organized experiment layout:

```text
/path/to/sarcasm_organized/
  core/audio_tensor/<sample_id>.pkl
  core/video_tensor/<sample_id>.pkl
  core/final_context_videos/SE-MUStARD-Final.csv
  core/texts/<sample_id>.txt
  embeddings/bert/bert-output.jsonl
```

With these inputs, install `requirements-training.txt` and `requirements-encoders.txt`. Generate target vectors with `work/regenerate_bert_embeddings_by_id.py`, then context vectors with `work/regenerate_context_bert_embeddings.py`. Run the fixed LR evaluation with:

```bash
python work/run_context_bert_lr.py \
  --data-dir /path/to/sarcasm_organized \
  --base-text-embedding-npz outputs/bert_regenerated/bert_cls_last4_by_sample_id.npz \
  --context-embedding-dir features \
  --output-dir reproduced/full_lr --seed 2027 --repeats 20

python work/run_context_mlp_robustness.py \
  --data-dir /path/to/sarcasm_organized \
  --base-text-embedding-npz outputs/bert_regenerated/bert_cls_last4_by_sample_id.npz \
  --context-embedding-dir features \
  --output-dir reproduced/full_mlp --seeds 42 123 2027
```

The original data source is [MUStARD](https://github.com/soujanyaporia/MUStARD). [MUStARD++](https://github.com/cfiltnlp/MUStARD_Plus_Plus) is a separate later 1,202-instance extension and was not evaluated in this study. The 690 original targets become 684 after excluding six missing video tensors; the exclusions are listed in `results/lr/excluded_samples.csv`.

## Interpretation and reporting corrections

The main context curve uses context-first inputs. Replacement controls use target-first inputs and must be compared against the genuine target-first C3 baseline (Macro-F1 0.588), not the context-first score (0.631). Each control has a separately fitted classifier. The corrected control intervals include zero.

Donors were generated once before splitting with seed 2027. These exploratory controls do not guarantee donor-text isolation across folds. Speaker separation refers to target speakers. Global permutation can contain fixed points; matched donor sampling excludes the target itself.

Late fusion averages two probability estimates: one text classifier and one classifier on concatenated audio/video. Repeated splits use GroupShuffleSplit seeds 2027 through 2046. These descriptions correct wording in earlier manuscript drafts; they reflect the saved experiment code.

The MLP uses base seeds 42, 123, and 2027, with `random_state = base_seed + fold`. Its internal early-stopping holdout is sample-random within each outer training fold, not separately grouped by speaker. Outer test folds remain speaker-separated.

The audiovisual tensors were mean-pooled to 768 dimensions each. The exact upstream checkpoints and extraction preprocessing are not recoverable from the archived evaluation scripts. Conclusions are limited to these fixed representations and simple fusion heads.

## Data references

S. Castro et al. Towards Multimodal Sarcasm Detection (An Obviously Perfect Paper). ACL, 2019, pp. 4619-4629.

A. Ray et al. A Multimodal Corpus for Emotion Recognition in Sarcasm. LREC, 2022, pp. 6992-7003.
