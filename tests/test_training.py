"""Training foundation tests: datasets, tokenizer, checkpoints, tracking, eval."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from alphaai.core.errors import DatasetError, DatasetValidationError, CheckpointError
from alphaai.training import checkpoints as checkpoints_mod
from alphaai.training import dataset as dataset_mod
from alphaai.training import evaluation as evaluation_mod
from alphaai.training import experiment as experiment_mod
from alphaai.training import model_card as model_card_mod
from alphaai.training import tokenizer as tokenizer_mod
from alphaai.training import pipeline as pipeline_mod

from .conftest import REPO_ROOT


def test_seed_dataset_is_valid_and_hashed() -> None:
    directory = REPO_ROOT / "datasets" / "alphaai-sample"
    report = dataset_mod.validate_dataset(directory)
    assert report.ok is True
    assert report.records >= 20
    assert report.manifest_ok is True
    assert set(report.splits) == {"train", "valid"}
    for split in report.splits.values():
        assert len(split["sha256"]) == 64


def test_dataset_validation_rejects_broken_records(echo_dataset: Path) -> None:
    (echo_dataset / "train.jsonl").write_text('{"instruction": "", "response": ""}\n', encoding="utf-8")
    report = dataset_mod.validate_dataset(echo_dataset, check_manifest=False)
    assert report.ok is False
    assert any(problem["problem"] for problem in report.problems)


def test_prepare_dedupes_and_writes_manifest(echo_dataset: Path) -> None:
    records = dataset_mod.load_records(echo_dataset / "train.jsonl")
    records = records + [records[0]]  # introduce a duplicate
    dataset_mod.write_jsonl(echo_dataset / "train.jsonl", records)
    spec = dataset_mod.load_spec(echo_dataset)
    collected = dataset_mod.load_records(dataset_mod.resolve_split_path(spec, "train"))
    deduped, removed = dataset_mod.deduplicate(collected)
    assert removed == 1
    splits = dataset_mod.split_records(deduped, valid_ratio=0.25, test_ratio=0.0, seed=spec.seed)
    dataset_mod.write_jsonl(echo_dataset / "train.jsonl", splits["train"])
    dataset_mod.write_jsonl(echo_dataset / "valid.jsonl", splits["valid"])
    manifest = dataset_mod.build_manifest(echo_dataset)
    assert manifest["records_total"] == len(deduped)
    report = dataset_mod.validate_dataset(echo_dataset)
    assert report.manifest_ok is True


def test_splits_are_deterministic(echo_dataset: Path) -> None:
    records = dataset_mod.load_records(echo_dataset / "train.jsonl")
    first = dataset_mod.split_records(records, valid_ratio=0.3, seed=42)
    second = dataset_mod.split_records(records, valid_ratio=0.3, seed=42)
    assert [record.training_text for record in first["train"]] == [
        record.training_text for record in second["train"]
    ]
    third = dataset_mod.split_records(records, valid_ratio=0.3, seed=43)
    assert [record.training_text for record in third["train"]] != [
        record.training_text for record in first["train"]
    ]


def test_missing_dataset_metadata_raises(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()
    with pytest.raises(DatasetError):
        dataset_mod.load_spec(tmp_path / "empty")


def test_tokenizer_trains_and_round_trips(tmp_path: Path) -> None:
    documents = [
        "AlphaAI routes requests to local engines.",
        "The router never invents a model that does not exist.",
        "Tools are permission checked and deny by default.",
    ]
    config = tokenizer_mod.TokenizerConfig(vocab_size=320, min_frequency=2, max_chars=10_000)
    artifact = tokenizer_mod.train_tokenizer(documents, config)
    assert artifact.vocab_size > tokenizer_mod.BYTE_VOCAB_SIZE
    assert artifact.merges
    report = tokenizer_mod.roundtrip_report(artifact, documents)
    assert report["roundtrip_rate"] == 1.0

    manifest = tokenizer_mod.save_tokenizer(artifact, tmp_path / "tokenizer")
    assert manifest["vocab_size"] == artifact.vocab_size
    reloaded = tokenizer_mod.load_tokenizer(tmp_path / "tokenizer")
    assert reloaded.vocab_size == artifact.vocab_size
    assert reloaded.special_ids == artifact.special_ids
    assert reloaded.decode(reloaded.encode("AlphaAI")) == "AlphaAI"


def test_tokenizer_requires_corpus() -> None:
    with pytest.raises(DatasetValidationError):
        tokenizer_mod.train_tokenizer([], tokenizer_mod.TokenizerConfig())


def test_checkpoint_manager_retention_and_verify(tmp_path: Path) -> None:
    manager = checkpoints_mod.CheckpointManager(tmp_path / "ckpt", keep=2)
    for step in (10, 20, 30):
        manager.save_metadata(f"step-{step:05d}", {"loss": 1.0 / step}, step=step)
    names = [info.name for info in manager.list()]
    assert names == ["step-00020", "step-00030"]
    assert manager.latest().name == "step-00030"
    report = manager.verify()
    assert report["ok"] is True and report["checked"] >= 2

    with pytest.raises(CheckpointError):
        manager.load("step-00030")  # metadata-only checkpoint has no tensors


def test_checkpoint_state_round_trip_when_torch_present(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    manager = checkpoints_mod.CheckpointManager(tmp_path / "ckpt", keep=1)
    info = manager.save_state("step-00001", {"w": torch.zeros(4)}, step=1, metadata={"loss": 0.5})
    assert info.kind == "state"
    state, meta = manager.load("step-00001")
    assert meta["step"] == 1
    assert tuple(state["w"].shape) == (4,)


def test_experiment_tracker_writes_real_files(tmp_path: Path) -> None:
    tracker = experiment_mod.ExperimentTracker.start(tmp_path / "experiments", "unit-run", config={"lr": 1e-3})
    tracker.log_metrics(1, {"loss": 1.5})
    tracker.log_metrics(2, {"loss": 1.2})
    payload = tracker.finish(summary={"final_loss": 1.2})
    assert payload["status"] == "completed"
    assert (tracker.directory / "config.json").exists()
    assert (tracker.directory / "run.json").exists()
    assert len(tracker.metrics()) == 2
    runs = experiment_mod.list_runs(tmp_path / "experiments")
    assert runs and runs[0]["run_id"] == tracker.run_id
    assert runs[0]["result"]["summary"]["final_loss"] == 1.2


def test_evaluation_reports_skips_honestly(config, echo_dataset: Path) -> None:
    suite = evaluation_mod.default_suite()
    report = suite.run(
        {
            "dataset_dir": str(echo_dataset),
            "tokenizer_dir": str(config.paths.tokenizer_dir),
            "runtime": None,
            "checkpoint": None,
            "prompts": ["hello"],
        }
    )
    by_name = {entry["name"]: entry for entry in report["results"]}
    assert by_name["dataset"]["ok"] is True
    assert by_name["tokenizer"]["skipped_reason"]
    assert by_name["language_model"]["skipped_reason"]
    assert by_name["perplexity"]["skipped_reason"]
    assert report["skipped"] == 3


def test_model_card_records_ownership(echo_dataset: Path, tmp_path: Path) -> None:
    metadata = model_card_mod.build_metadata(
        model_name="AlphaAI-X-reference",
        model_owner="AlphaAI",
        checkpoint=None,
        dataset_dir=echo_dataset,
        tokenizer_dir=None,
        hparams={"dim": 96},
        metrics={"parameters": 1234},
        status="reference",
    )
    assert metadata.is_alphaai_owned
    paths = model_card_mod.write_model_card(metadata, tmp_path / "card")
    payload = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    assert payload["model_owner"] == "AlphaAI"
    assert payload["engine_owner"] == "AlphaAI"
    assert payload["datasets"][0]["name"] == "unit-sample"
    assert "Weights created by" in Path(paths["markdown"]).read_text(encoding="utf-8")
    entry = model_card_mod.candidate_model_entry(metadata)
    assert entry["status"] == "reference"


def test_reference_training_loop_produces_real_checkpoints(echo_dataset: Path, tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    tokenizer_dir = tmp_path / "tokenizer"
    documents = [record.training_text for record in dataset_mod.load_records(echo_dataset / "train.jsonl")]
    artifact = tokenizer_mod.train_tokenizer(documents, tokenizer_mod.TokenizerConfig(vocab_size=300, max_chars=4000))
    tokenizer_mod.save_tokenizer(artifact, tokenizer_dir)

    manager = checkpoints_mod.CheckpointManager(tmp_path / "checkpoints", keep=2)
    tracker = experiment_mod.ExperimentTracker.start(tmp_path / "experiments", "unit-train")
    report = pipeline_mod.train_reference(
        config=pipeline_mod.ReferenceModelConfig(
            vocab_size=artifact.vocab_size,
            dim=32,
            n_layers=1,
            n_heads=2,
            block_size=16,
            max_steps=4,
            batch_size=2,
            gradient_accumulation_steps=1,
            eval_every=2,
            checkpoint_every=4,
            max_records=9,
        ),
        dataset_dir=echo_dataset,
        tokenizer_dir=tokenizer_dir,
        checkpoints=manager,
        tracker=tracker,
    )
    assert report["parameters"] > 0
    assert len(report["steps"]) == 4
    assert report["checkpoints"], "a real checkpoint must be written"
    model_path = Path(report["checkpoints"][-1]["path"]) / "model.pt"
    assert model_path.exists() and model_path.stat().st_size > 0
    state, metadata = manager.load(report["checkpoints"][-1]["name"])
    assert any(key.startswith("token_embedding") for key in state)
    assert metadata["metadata"]["note"].startswith("AlphaAI reference training run")

    model, loaded_config = pipeline_mod.load_reference_model(
        report["checkpoints"][-1]["path"], tokenizer_dir
    )
    assert loaded_config.vocab_size == artifact.vocab_size
    assert model is not None
