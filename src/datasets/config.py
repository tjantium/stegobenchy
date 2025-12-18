"""Config-driven dataset factory.

Datasets are defined in `config/datasets.yaml` and can be loaded by name.
This keeps experiment / demo code from hard-coding dataset parameters.
"""

from pathlib import Path
from typing import Any, Dict

import yaml
from datasets import Dataset

from . import (
    generate_coin_flip_dataset,
    generate_coin_flip_with_context,
    generate_paraphrase_dataset,
    generate_encoded_paraphrase_dataset,
    generate_robust_paraphrase_dataset,
    generate_stego_cover_dataset,
    generate_robustness_evaluation_dataset,
    generate_monitoring_robustness_dataset,
)

_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "datasets.yaml"


def load_dataset_config() -> Dict[str, Any]:
    """Load the dataset config YAML into a dict.

    Returns:
        Mapping from dataset name → config dict.
    """

    if not _CONFIG_PATH.exists():
        raise FileNotFoundError(f"Dataset config not found at {_CONFIG_PATH}")

    with _CONFIG_PATH.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise ValueError("datasets.yaml must contain a top-level mapping")

    return data


def get_dataset(name: str) -> Dataset:
    """Create a dataset by name, using `config/datasets.yaml`.

    Args:
        name: Key in the YAML file (e.g. "coin_flip_default").

    Returns:
        A `datasets.Dataset` instance.
    """

    cfg = load_dataset_config()
    if name not in cfg:
        raise KeyError(f"Unknown dataset config '{name}'. Available: {list(cfg.keys())}")

    entry = cfg[name]
    ds_type = entry.get("type")
    params: Dict[str, Any] = entry.get("params", {})

    if ds_type == "coin_flip":
        return generate_coin_flip_dataset(**params)
    if ds_type == "coin_flip_with_context":
        return generate_coin_flip_with_context(**params)
    if ds_type == "paraphrase":
        base_texts = params.pop("base_texts", ["Example text 1", "Example text 2"])
        return generate_paraphrase_dataset(base_texts=base_texts, **params)
    if ds_type == "encoded_paraphrase":
        base_texts = params.pop("base_texts", ["Example text 1", "Example text 2"])
        hidden_info = params.pop("hidden_info", ["secret1", "secret2"])
        return generate_encoded_paraphrase_dataset(
            base_texts=base_texts,
            hidden_info=hidden_info,
            **params,
        )
    if ds_type == "robust_paraphrase":
        base_texts = params.pop("base_texts", ["Example text 1", "Example text 2"])
        return generate_robust_paraphrase_dataset(base_texts=base_texts, **params)
    if ds_type == "stego_cover":
        return generate_stego_cover_dataset(**params)
    if ds_type == "robustness_evaluation":
        raise NotImplementedError(
            "robustness_evaluation requires a base_dataset; call the function directly."
        )
    if ds_type == "monitoring_robustness":
        return generate_monitoring_robustness_dataset(**params)

    raise ValueError(f"Unknown dataset type '{ds_type}' for config '{name}'")
