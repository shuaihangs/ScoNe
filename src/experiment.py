import os
import re
from pathlib import Path

import pandas as pd
import torch

from .config import *
from .data import (load_rows_from_csv, print_dataset_counts,
                   split_examples_by_dataset, split_examples_by_dataset_kfold,
                   validate_required_splits)
from .model import load_frozen_lm
from .utils import set_seed

def slugify(value):
    value = re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_")
    return value.lower()


def normalize_seed(value):
    numeric = float(value)
    if not numeric.is_integer():
        raise ValueError(f"Seed must be an integer, got {value!r}.")
    return str(int(numeric))


def normalize_fold(value):
    numeric = float(value)
    if not numeric.is_integer() or numeric < 1:
        raise ValueError(f"CV fold must be a positive integer, got {value!r}.")
    return str(int(numeric))


def fold_suffix(cv_fold=1, cv_folds=1):
    if int(cv_folds) <= 1:
        return ""
    return f"_fold{int(cv_fold)}of{int(cv_folds)}"


def config_value(config, key):
    if key not in config:
        raise KeyError(f"Missing tuning config key: {key}")
    return config[key]


def config_slug(config):
    return slugify(config_value(config, "name"))


def checkpoint_path(
    config,
    model_name,
    train_dataset,
    seed,
    cv_fold=1,
    cv_folds=1,
):
    return os.path.join(
        CHECKPOINT_DIR,
        (
            f"best_{config_slug(config)}_{slugify(model_name)}"
            f"_train_{slugify(train_dataset)}_seed{seed}"
            f"{fold_suffix(cv_fold, cv_folds)}.pt"
        ),
    )


def history_path(
    config,
    model_name,
    train_dataset,
    seed,
    cv_fold=1,
    cv_folds=1,
):
    return os.path.join(
        HISTORY_DIR,
        (
            f"history_{config_slug(config)}_{slugify(model_name)}"
            f"_train_{slugify(train_dataset)}_seed{seed}"
            f"{fold_suffix(cv_fold, cv_folds)}.csv"
        ),
    )


def ensure_output_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(HISTORY_DIR, exist_ok=True)


def configure_output_dir(output_dir):
    if output_dir is None:
        return

    global OUTPUT_DIR, CHECKPOINT_DIR, HISTORY_DIR, FEATURE_CACHE_DIR
    OUTPUT_DIR = os.path.normpath(output_dir)
    CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
    HISTORY_DIR = os.path.join(OUTPUT_DIR, "histories")
    FEATURE_CACHE_DIR = os.path.join(OUTPUT_DIR, "feature_cache")


def configure_determinism(seed):
    set_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def matches_requested(value, requested_values):
    if requested_values is None:
        return True

    requested = {str(x) for x in requested_values}
    requested_slugs = {slugify(x) for x in requested_values}
    return value in requested or slugify(value) in requested_slugs


def select_models(requested):
    return [
        model_name
        for model_name in MODEL_NAMES
        if matches_requested(model_name, requested)
    ]


def select_datasets(requested):
    return [
        dataset_name
        for dataset_name in DATASET_NAMES
        if matches_requested(dataset_name, requested)
    ]


def print_split_counts(splits):
    cv_fold = splits.get("cv_fold", 1)
    cv_folds = splits.get("cv_folds", 1)
    print(f"\nConfigured dataset groups: fold {cv_fold}/{cv_folds}")
    print("------------------------------------")

    for dataset_name in splits["dataset_names"]:
        dataset = splits["datasets"][dataset_name]
        counts = [
            f"rows={len(dataset['rows'])}",
        ]
        if "outer_train_rows" in dataset:
            counts.append(
                f"outer_train_rows={len(dataset['outer_train_rows'])}"
            )
        counts.extend(
            [
                f"train_rows={len(dataset['train_rows'])}",
                f"validation_rows={len(dataset['validation_rows'])}",
            ]
        )
        if "test_rows" in dataset:
            counts.append(f"test_rows={len(dataset['test_rows'])}")
        counts.append(f"examples={len(dataset['examples'])}")
        print(f"{dataset_name}: {', '.join(counts)}")


def build_experiment_splits(rows, cv_folds, seed):
    if cv_folds < 1:
        raise ValueError("--cv-folds must be at least 1.")
    if cv_folds == 1:
        return [
            split_examples_by_dataset(
                rows,
                dataset_names=DATASET_NAMES,
                validation_ratio=VALIDATION_RATIO,
                seed=seed,
            )
        ]
    return split_examples_by_dataset_kfold(
        rows,
        dataset_names=DATASET_NAMES,
        n_splits=cv_folds,
        validation_ratio=VALIDATION_RATIO,
        seed=seed,
    )


def print_split_audit(split_folds):
    print("\nSplit integrity audit")
    print("---------------------")
    for splits in split_folds:
        validate_required_splits(splits, DATASET_NAMES)
        print_split_counts(splits)

    if len(split_folds) > 1:
        print(
            "\nNested grouped CV audit passed: the original outer 20% test "
            "partition is fixed and untouched in every fold; every row and "
            "normalized question in the outer 80% appears in inner validation "
            "exactly once; train/validation/test questions are disjoint."
        )
    else:
        print("\nQuestion-grouped holdout audit passed.")


def sanity_check_train_loader(loaders):
    sanity_batch = next(iter(loaders["train"]))
    required_keys = [
        "pos_answer_mask",
        "neg_answer_mask",
    ]

    for key in required_keys:
        if key not in sanity_batch:
            raise KeyError(
                f"Missing {key} in train batch. "
                "Your data.py is not returning answer masks correctly."
            )

    print("Answer-mask sanity check passed.")
    print(f"pos_answer_mask shape: {sanity_batch['pos_answer_mask'].shape}")
    print(f"neg_answer_mask shape: {sanity_batch['neg_answer_mask'].shape}")
    print(
        "Mean positive answer tokens per sample:",
        sanity_batch["pos_answer_mask"].sum(dim=1).float().mean().item(),
    )
    print(
        "Mean negative answer tokens per sample:",
        sanity_batch["neg_answer_mask"].sum(dim=1).float().mean().item(),
    )
def atomic_csv(rows: list[dict[str, object]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    pd.DataFrame(rows).to_csv(temporary_path, index=False)
    os.replace(temporary_path, path)


def assert_private_feature_cache(output_dir: Path) -> Path:
    """Reject cache paths that could resolve outside this experiment."""
    output_dir = output_dir.absolute()
    cache_dir = output_dir / "feature_cache"

    if output_dir.exists() and output_dir.is_symlink():
        raise RuntimeError(f"Output directory may not be a symlink: {output_dir}")
    if cache_dir.exists() and cache_dir.is_symlink():
        raise RuntimeError(f"Feature cache may not be a symlink: {cache_dir}")
    if cache_dir.exists():
        linked_entries = [path for path in cache_dir.rglob("*") if path.is_symlink()]
        if linked_entries:
            raise RuntimeError(
                "Feature cache contains symlinks: "
                + ", ".join(str(path) for path in linked_entries)
            )

    configured_cache = Path(FEATURE_CACHE_DIR).absolute()
    if configured_cache != cache_dir:
        raise RuntimeError(
            f"Refusing external cache {configured_cache}; expected {cache_dir}."
        )
    return cache_dir


def checkpoint_file(
    config: dict[str, object],
    model_name: str,
    train_dataset: str,
    seed: int,
    cv_fold: int = 1,
    cv_folds: int = 1,
) -> Path:
    return Path(
        checkpoint_path(
            config,
            model_name,
            train_dataset,
            seed,
            cv_fold=cv_fold,
            cv_folds=cv_folds,
        )
    )


def history_file(
    config: dict[str, object],
    model_name: str,
    train_dataset: str,
    seed: int,
    cv_fold: int = 1,
    cv_folds: int = 1,
) -> Path:
    return Path(
        history_path(
            config,
            model_name,
            train_dataset,
            seed,
            cv_fold=cv_fold,
            cv_folds=cv_folds,
        )
    )
