from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final


@dataclass(frozen=True)
class DatasetConfig:
    """
    Dataset-specific mapping from logical model keys -> on-disk NPZ filenames.

    Notes:
    - "preds" are the base model prediction matrices. Some may only exist for
      one split (e.g. koko has test-nn.npz but no train-nn.npz).
    - "ensemble3" defines the *exact* 3 predictors used by all 3-way ensembles
      (mean/weighted mean/torch_mean/torch_* ensemble scripts).
    """

    name: str
    preds: dict[str, dict[str, str]]  # preds[split][model_key] -> filename
    ensemble3: tuple[str, str, str]


DATA_ROOT: Final[Path] = Path("data")
SUPPORTED_DATASETS: Final[tuple[str, ...]] = ("yso-fi", "yso-en", "koko")


def _preds_for_yso() -> dict[str, dict[str, str]]:
    # yso-* layouts share the same filenames
    return {
        "train": {
            "bonsai": "train-bonsai.npz",
            "fasttext": "train-fasttext.npz",
            "mllm": "train-mllm.npz",
            "nn": "train-nn.npz",
        },
        "test": {
            "bonsai": "test-bonsai.npz",
            "fasttext": "test-fasttext.npz",
            "mllm": "test-mllm.npz",
            "nn": "test-nn.npz",
        },
    }


DATASETS: Final[dict[str, DatasetConfig]] = {
    "yso-fi": DatasetConfig(
        name="yso-fi",
        preds=_preds_for_yso(),
        ensemble3=("bonsai", "fasttext", "mllm"),
    ),
    "yso-en": DatasetConfig(
        name="yso-en",
        preds=_preds_for_yso(),
        ensemble3=("bonsai", "fasttext", "mllm"),
    ),
    "koko": DatasetConfig(
        name="koko",
        preds={
            "train": {
                "bonsai_gemma3": "train-bonsai-gemma3.npz",
                "bonsai_ovis2": "train-bonsai-ovis2.npz",
                "mllm": "train-mllm.npz",
                # Intentionally no train-nn.npz (allowed).
            },
            "test": {
                "bonsai_gemma3": "test-bonsai-gemma3.npz",
                "bonsai_ovis2": "test-bonsai-ovis2.npz",
                "mllm": "test-mllm.npz",
                "nn": "test-nn.npz",
            },
        },
        ensemble3=("bonsai_gemma3", "bonsai_ovis2", "mllm"),
    ),
}


def get_dataset_config(dataset: str) -> DatasetConfig:
    if dataset not in DATASETS:
        raise ValueError(
            f"Unknown dataset {dataset!r}. Supported: {', '.join(SUPPORTED_DATASETS)}"
        )
    return DATASETS[dataset]


def truth_path(dataset: str, split: str) -> Path:
    if split not in {"train", "test"}:
        raise ValueError("split must be 'train' or 'test'")
    return DATA_ROOT / dataset / f"{split}-output.npz"


def pred_path(dataset: str, split: str, model_key: str) -> Path:
    cfg = get_dataset_config(dataset)
    if split not in cfg.preds:
        raise ValueError(f"Unknown split {split!r}")
    if model_key not in cfg.preds[split]:
        raise KeyError(f"Model {model_key!r} not available for dataset={dataset} split={split}")
    return DATA_ROOT / dataset / cfg.preds[split][model_key]


def ensemble3_keys(dataset: str) -> tuple[str, str, str]:
    return get_dataset_config(dataset).ensemble3


def all_pred_keys(dataset: str, split: str) -> list[str]:
    cfg = get_dataset_config(dataset)
    return sorted(cfg.preds.get(split, {}).keys())
