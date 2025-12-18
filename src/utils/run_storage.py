"""Run storage utilities with lightweight versioning and metadata.

Features
--------
- Creates timestamped run directories under ``runs/``.
- Saves HF ``Dataset`` objects and pipeline results as JSONL.
- Records metadata including git commit, model info, dataset config, and arbitrary
  extra fields you pass in.
- Provides helpers to load and summarise past runs.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

from datasets import Dataset


def _get_git_commit(root: Optional[Path] = None) -> Optional[str]:
    """Return the current git commit hash, or None if unavailable."""

    root = root or Path(__file__).resolve().parents[2]
    try:
        commit = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=str(root), stderr=subprocess.DEVNULL
            )
            .decode("utf-8")
            .strip()
        )
        return commit
    except Exception:
        return None


def create_run_dir(
    base_dir: str = "runs",
    run_name: Optional[str] = None,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> Path:
    """Create a new timestamped run directory and write ``meta.json``.

    Args:
        base_dir: Root directory for runs (relative to repo root).
        run_name: Optional human-friendly name suffix.
        extra_meta: Optional additional metadata to merge into ``meta.json``.
            Common fields you may want to pass:
            - ``model_name`` (e.g. ``\"llama3:8b\"``)
            - ``dataset_config`` (e.g. ``\"coin_flip_default\"``)
            - key generation parameters (``temperature``, ``max_tokens``)

    Returns:
        Path to the created run directory.
    """

    ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    label = run_name or "run"
    run_dir = Path(base_dir) / f"{ts}_{label}"
    run_dir.mkdir(parents=True, exist_ok=False)

    meta: Dict[str, Any] = {
        "created_at_utc": ts,
        "run_name": label,
        "git_commit": _get_git_commit(),
    }
    if extra_meta:
        # Shallow merge – explicit keys from extra_meta win
        meta.update(extra_meta)

    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return run_dir


def save_dataset_snapshot(
    dataset: Dataset,
    run_dir: Path,
    name: str = "dataset",
    max_rows: Optional[int] = None,
) -> Path:
    """Save a snapshot of a dataset as JSONL in the run directory.

    Args:
        dataset: HF Dataset to save.
        run_dir: Run directory created by :func:`create_run_dir`.
        name: Base file name (without extension).
        max_rows: If set, only save the first N rows (useful for git-tracked snapshots).

    Returns:
        Path to the written file.
    """

    run_dir = Path(run_dir)
    ds = dataset
    if max_rows is not None and len(ds) > max_rows:
        ds = ds.select(range(max_rows))

    out_path = run_dir / f"{name}.jsonl"
    ds.to_json(str(out_path), orient="records", lines=True)
    return out_path


def save_results_jsonl(
    results: Iterable[Dict[str, Any]],
    run_dir: Path,
    name: str = "results",
) -> Path:
    """Save a list of result dicts as JSONL.

    Args:
        results: Iterable of dictionaries (e.g., from ``run_experiment``).
        run_dir: Run directory.
        name: Base file name.

    Returns:
        Path to the written results file.
    """

    run_dir = Path(run_dir)
    out_path = run_dir / f"{name}.jsonl"

    with out_path.open("w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row) + "\n")

    return out_path


# ---------------------------------------------------------------------------
# Convenience helpers for inspecting past runs
# ---------------------------------------------------------------------------


def load_run(run_dir: Union[str, Path]) -> Dict[str, Any]:
    """Load metadata and (optionally) results for a given run directory.

    This is intentionally lightweight – it only reads a small prefix of the
    results file to avoid loading huge experiments by accident.
    """

    run_path = Path(run_dir)
    meta_path = run_path / "meta.json"
    results_path = run_path / "results.jsonl"

    if not meta_path.exists():
        raise FileNotFoundError(f"No meta.json found in {run_path}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    preview_results: List[Dict[str, Any]] = []
    if results_path.exists():
        with results_path.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= 10:  # preview first 10 rows
                    break
                try:
                    preview_results.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    return {
        "meta": meta,
        "results_preview": preview_results,
        "run_dir": str(run_path),
    }


def list_runs(base_dir: Union[str, Path] = "runs") -> List[Path]:
    """Return a sorted list of existing run directories."""

    base = Path(base_dir)
    if not base.exists():
        return []

    return sorted(
        [p for p in base.iterdir() if p.is_dir()],
        key=lambda p: p.name,
    )
