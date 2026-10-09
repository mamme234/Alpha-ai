"""Fetch a registered AlphaAI model's weights at *image build* time.

Why this exists
---------------
``render.yaml`` mounts a persistent disk at ``/data`` and Render (like every
container host) gives the build no access to that disk. If the weights are only
downloaded at boot, the first boot of a fresh service downloads 491 MB before
the port ever opens, and the platform's start-up window is the difference
between a deployment that comes up and one that is declared dead.

So the download happens while the image is built instead:

* the URL, file name, size and SHA-256 all come from
  ``configs/models/<model-id>.json`` — the same provenance record
  ``alphaai models install`` verifies at boot, never a hard-coded link,
* the bytes are verified (size + SHA-256) before the build can succeed,
* the weights are written to ``/opt/model-cache`` in the image and copied onto
  the persistent disk by ``deploy/entrypoint.sh`` on the first boot,
* the weights are **never** written into the repository: this script is run by
  the Dockerfile, not by any git-visible step, and ``.gitignore`` excludes
  ``models/*/`` regardless.

It is safe to run twice: an already-fetched, verified file is left alone.

Usage::

    python deploy/prefetch_model.py qwen2.5-0.5b-instruct-gguf \\
        --spec-dir configs --dest /opt/model-cache
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

CHUNK = 1024 * 1024


class PrefetchError(RuntimeError):
    """The weights could not be fetched or did not match their provenance."""


def load_provenance(spec_path: Path) -> dict:
    """Return the ``provenance`` block of a model spec, or raise."""

    try:
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise PrefetchError(f"model spec not found: {spec_path}") from exc
    except json.JSONDecodeError as exc:
        raise PrefetchError(f"model spec is not valid JSON: {spec_path}: {exc}") from exc

    provenance = spec.get("provenance") or {}
    for key in ("download_url", "filename"):
        if not provenance.get(key):
            raise PrefetchError(
                f"{spec_path} declares no provenance.{key}; AlphaAI only downloads "
                "weights from the repository recorded in its metadata."
            )
    return provenance


def digest(path: Path) -> tuple[int, str]:
    """Return ``(size, sha256)`` of a file without loading it into memory."""

    sha = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            size += len(block)
            sha.update(block)
    return size, sha.hexdigest()


def verified(path: Path, provenance: dict) -> bool:
    """True when ``path`` already matches the recorded size and SHA-256."""

    expected_size = provenance.get("file_size_bytes")
    expected_sha = provenance.get("sha256")
    if not path.is_file():
        return False
    if expected_size and path.stat().st_size != int(expected_size):
        return False
    if not expected_sha:
        # No checksum recorded: a size match is the strongest claim available.
        return bool(expected_size)
    size, sha = digest(path)
    return sha == str(expected_sha).lower()


def download(url: str, target: Path) -> None:
    """Stream ``url`` into ``target`` (written last, so a partial file never
    passes as a complete one)."""

    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "AlphaAI/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as out:
            while True:
                block = response.read(CHUNK)
                if not block:
                    break
                out.write(block)
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        partial.unlink(missing_ok=True)
        raise PrefetchError(f"could not download {url}: {exc}") from exc
    partial.replace(target)


def prefetch(model_id: str, spec_dir: Path, dest_root: Path) -> Path:
    """Ensure the model's weights sit verified under ``dest_root/<model_id>/``."""

    provenance = load_provenance(spec_dir / "models" / f"{model_id}.json")
    target = dest_root / model_id / str(provenance["filename"])

    if verified(target, provenance):
        print(f"alphaai prefetch: {target} already verified")
        return target

    print(f"alphaai prefetch: fetching {provenance['download_url']}")
    download(str(provenance["download_url"]), target)

    expected_size = provenance.get("file_size_bytes")
    if expected_size and target.stat().st_size != int(expected_size):
        actual = target.stat().st_size
        target.unlink(missing_ok=True)
        raise PrefetchError(
            f"{target.name} is {actual} bytes, expected {expected_size}; "
            "the download was incomplete and has been removed."
        )

    size, sha = digest(target)
    expected_sha = str(provenance.get("sha256") or "").lower()
    if expected_sha and sha != expected_sha:
        target.unlink(missing_ok=True)
        raise PrefetchError(
            f"{target.name} failed its SHA-256 check: got {sha}, expected {expected_sha}."
        )
    print(f"alphaai prefetch: verified {target} ({size} bytes, sha256 {sha[:16]}…)")
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_id", help="registered model id, e.g. qwen2.5-0.5b-instruct-gguf")
    parser.add_argument("--spec-dir", default="configs", type=Path)
    parser.add_argument("--dest", default=Path("/opt/model-cache"), type=Path)
    args = parser.parse_args(argv)
    try:
        prefetch(args.model_id, args.spec_dir, args.dest)
    except PrefetchError as exc:
        print(f"alphaai prefetch: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
