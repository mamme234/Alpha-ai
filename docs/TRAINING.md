# AlphaAI Training Foundation

AlphaAI ships a **real** training foundation so it can eventually own its own
weights. Today it trains a small tokenizer and a small language model from real
data on CPU — enough to prove every code path works end to end. No AlphaAI
weights exist yet; the `alphaai-x` model is registered with
`status: "not_trained"` and the `alphaai` engine reports that truthfully.

The foundation lives in `alphaai/training/` and is driven by
`alphaai train <action>`.

## Layout

| Path | Purpose |
| ---- | ------- |
| `datasets/` | datasets and their manifests (records, splits, licenses) |
| `tokenizer/` | trained tokenizer (`tokenizer.json`) and its manifest |
| `checkpoints/` | saved training checkpoints (`model.pt` + metadata) |
| `training/` | run metadata and logs per experiment |
| `evaluation/` | evaluation prompts and generated reports |
| `configs/training/` | training configurations |
| `scripts/training/`, `scripts/evaluation/` | thin wrappers around the same actions |

## Pipeline

```shell
alphaai train status                       # what exists and whether it validates
alphaai train validate  --dataset alphaai-sample
alphaai train prepare   --dataset alphaai-sample
alphaai train tokenize  --dataset alphaai-sample --vocab-size 4096
alphaai train finetune  --dataset alphaai-sample --max-steps 30
alphaai train evaluate  --dataset alphaai-sample
```

Each action is a real step, not a stub:

* **validate** — checks the dataset manifest and every record against the schema.
* **prepare** — de-duplicates records and writes real `train` / `valid` split
  files.
* **tokenize** — trains a byte-level BPE tokenizer on the corpus and reports the
  vocabulary size, merge count and a round-trip fidelity check.
* **finetune** — trains a small transformer language model on the tokenised
  corpus, reporting the real first/final loss and writing checkpoints with real
  `model.pt` state.
* **evaluate** — runs dataset/tokenizer/language-model checks, computes a real
  perplexity over the evaluation split, and writes a JSON report under
  `evaluation/reports/`.

Reported numbers are measured, never invented. When a step cannot run (missing
dataset, missing tokenizer, no trainable engine) it raises a structured
`training_error` with a remediation.

## Adding a dataset

Create `datasets/<name>/` with a `dataset.json` manifest and the split files it
names:

```json
{
  "name": "my-dataset",
  "version": "1.0.0",
  "license": "Apache-2.0",
  "source": "where the data came from",
  "format": "jsonl",
  "splits": { "train": "train.jsonl", "valid": "valid.jsonl" },
  "min_record_chars": 5
}
```

Records are JSON objects with an `instruction` and a `response` (or a `text`
field). Run `alphaai train validate --dataset my-dataset` to check it.

## Honesty

* Loss and perplexity are measured from the actual run.
* `checkpoints.py` verifies each checkpoint by hashing every stored file; a
  metadata-only checkpoint is rejected, not silently accepted.
* The foundation never imports DeepSeek's weights or claims to have trained them.
