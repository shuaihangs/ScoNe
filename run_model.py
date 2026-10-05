"""Train SCoNE using alpha * BCE + (1-alpha) * soft-neighbour loss / 6."""

import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import argparse
import fcntl
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import torch

from src import experiment, training
from src.reproducibility import record_run

def build_scaled_alpha_configs(
    alpha_values: list[float],
    *,
    soft_loss_divisor: float,
    k_neighbours: int,
    contrastive_temperature: float,
    positiveness_temperature: float,
) -> list[dict[str, object]]:
    """Build alpha * BCE + (1-alpha) * soft/divisor configurations."""
    if soft_loss_divisor <= 0.0:
        raise ValueError("--soft-loss-divisor must be positive.")

    configs = []
    for alpha in sorted({float(value) for value in alpha_values}):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("Every --scaled-alpha-values entry must be in [0, 1].")
        scaled_soft_coefficient = (1.0 - alpha) / soft_loss_divisor
        config = training.make_config(
            scaled_soft_coefficient,
            k_neighbours=k_neighbours,
            contrastive_temperature=contrastive_temperature,
            positiveness_temperature=positiveness_temperature,
        )
        config.update(
            {
                "name": (
                    "single_stage_scaled_alpha_"
                    f"{training.coefficient_slug(alpha)}_soft_div_"
                    f"{training.coefficient_slug(soft_loss_divisor)}_k"
                    f"{int(config['k_neighbours'])}"
                ),
                "lambda_bce": alpha,
                "lambda_soft_neighbour_contrastive": scaled_soft_coefficient,
                "uses_bce": alpha > 0.0,
                "loss_mixing_alpha": alpha,
                "soft_neighbour_loss_divisor": soft_loss_divisor,
                "stage1_loss_type": (
                    "alpha_bce_plus_one_minus_alpha_soft_neighbour_"
                    "contrastive_divided_by_constant"
                ),
            }
        )
        configs.append(config)
    return configs

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--train-datasets", nargs="+", default=None)
    parser.add_argument("--csv-path", default=experiment.CSV_PATH)
    parser.add_argument("--alpha-values", "--scaled-alpha-values", dest="alpha_values",
                        nargs="+", type=float, default=[0, .25, .5, .75, 1])
    parser.add_argument("--k-neighbours", type=int, choices=[10, 30, 50], default=10)
    parser.add_argument("--soft-loss-divisor", type=float, default=6.0)
    parser.add_argument("--contrastive-temperature", type=float, default=.2)
    parser.add_argument("--positiveness-temperature", type=float, default=.2)
    parser.add_argument("--cv-folds", type=int, choices=[1, 5], default=5)
    parser.add_argument("--folds", nargs="+", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    if args.contrastive_temperature <= 0 or args.positiveness_temperature <= 0:
        parser.error("temperatures must be positive")
    if args.folds and (len(set(args.folds)) != len(args.folds) or
                      any(f < 1 or f > args.cv_folds for f in args.folds)):
        parser.error("--folds must be distinct indices within 1..cv-folds")
    return args

def result_key(row: dict[str, object]) -> tuple[object, ...]:
    return (
        str(row["model_name"]),
        str(row["config_name"]),
        str(row["train_dataset"]),
        experiment.normalize_seed(row["seed"]),
        experiment.normalize_fold(row.get("cv_fold", 1)),
        experiment.normalize_fold(row.get("cv_folds", 1)),
    )


def expected_eval_datasets(train_dataset: str, cv_folds: int) -> set[str]:
    source_names = (
        {
            f"{train_dataset}_train",
            f"{train_dataset}_inner_val",
            f"{train_dataset}_test",
        }
        if cv_folds > 1
        else {f"{train_dataset}_train", f"{train_dataset}_val"}
    )
    return {
        *source_names,
        *(name for name in experiment.DATASET_NAMES if name != train_dataset),
    }


def load_resume_rows(path: Path) -> tuple[list[dict[str, object]], set[tuple]]:
    if not path.exists():
        return [], set()
    frame = pd.read_csv(path)
    required = {
        "model_name",
        "config_name",
        "train_dataset",
        "eval_dataset",
        "seed",
        "cv_fold",
        "cv_folds",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError("Cannot resume summary; missing: " + ", ".join(sorted(missing)))

    completed = set()
    for _, group in frame.groupby(
        [
            "model_name",
            "config_name",
            "train_dataset",
            "seed",
            "cv_fold",
            "cv_folds",
        ],
        dropna=False,
    ):
        first = group.iloc[0].to_dict()
        if set(group["eval_dataset"].astype(str)) == expected_eval_datasets(
            str(first["train_dataset"]), int(first["cv_folds"])
        ):
            completed.add(result_key(first))
    rows = [row for row in frame.to_dict("records") if result_key(row) in completed]
    print(
        f"Resume: retained {len(rows)} rows from {len(completed)} complete cells.",
        flush=True,
    )
    return rows, completed


def save_cv_summary(rows: list[dict[str, object]], path: Path, cv_folds: int) -> None:
    if cv_folds <= 1 or not rows:
        return
    frame = pd.DataFrame(rows)
    group_columns = [
        "stage",
        "config_name",
        "model_name",
        "train_dataset",
        "seed",
        "cv_folds",
        "eval_dataset",
    ]
    metric_columns = [
        "loss",
        "accuracy",
        "energy_auc",
        "logit_auc",
        "energy_gap",
    ]
    metadata_columns = [
        "lambda_bce",
        "lambda_soft_neighbour_contrastive",
        "loss_mixing_alpha",
        "soft_neighbour_loss_divisor",
        "neighbour_backend",
        "k_neighbours",
        "soft_neighbour_contrastive_temperature",
        "soft_neighbour_positiveness_temperature",
        "soft_neighbour_weight_source",
        "classifier_role",
        "final_classifier_retained",
        "projection_frozen",
        "max_epochs",
        "split_strategy",
    ]
    aggregate_rows = []
    for key, group in frame.groupby(group_columns, dropna=False):
        if group["cv_fold"].nunique() != cv_folds:
            continue
        row = dict(zip(group_columns, key))
        row["n_folds"] = int(group["cv_fold"].nunique())
        for column in metadata_columns:
            if column in group.columns:
                row[column] = group.iloc[0][column]
        for column in metric_columns:
            if column in group.columns:
                values = pd.to_numeric(group[column], errors="coerce").dropna()
                row[f"{column}_mean"] = values.mean()
                row[f"{column}_std"] = values.std(ddof=1)
        aggregate_rows.append(row)
    if aggregate_rows:
        experiment.atomic_csv(aggregate_rows, path)
        print(f"Updated five-fold summary: {path}", flush=True)


def print_settings(args, models, datasets, configs, output_dir: Path) -> None:
    folds = args.folds or list(range(1, args.cv_folds + 1))
    cells = len(models) * len(datasets) * len(configs) * len(folds)
    evaluations_per_cell = len(experiment.DATASET_NAMES) + (
        2 if args.cv_folds > 1 else 1
    )
    print("============================================================")
    print("Single-stage BCE + soft-neighbour contrastive experiment")
    print("============================================================")
    print(f"Models ({len(models)}): {models}")
    print(f"Training datasets ({len(datasets)}): {datasets}")
    print(f"Coefficient settings ({len(configs)}):")
    for config in configs:
        if "loss_mixing_alpha" in config:
            print(
                f"  alpha={config['loss_mixing_alpha']:.3f}: "
                f"BCE coefficient={config['lambda_bce']:.6f}, "
                "SoftNeighbourContrastive coefficient="
                f"{config['lambda_soft_neighbour_contrastive']:.6f} "
                f"(raw loss divided by {config['soft_neighbour_loss_divisor']:g}), "
                f"k={config['k_neighbours']}"
            )
        else:
            print(
                f"  BCE={config['lambda_bce']:.3f}, "
                "SoftNeighbourContrastive="
                f"{config['lambda_soft_neighbour_contrastive']:.3f}, "
                f"k={config['k_neighbours']}"
            )
    print(
        "Question-grouped split: fixed outer 80/20 with "
        f"{args.cv_folds}-fold CV inside the 80%, seed={args.seed}"
    )
    print(f"Requested folds: {folds}")
    print(f"Epochs: {args.epochs} (full training, no early stopping)")
    print("Classifier: jointly trained Linear(320, 1), retained for evaluation")
    print(f"Training cells: {cells}")
    print(f"Evaluation rows: {cells * evaluations_per_cell}")
    print(f"Output directory: {output_dir}")
    print(f"Private feature cache: {output_dir / 'feature_cache'}")
    print("============================================================")


def main() -> None:
    args = parse_args()
    models = experiment.select_models(args.models)
    datasets = experiment.select_datasets(args.train_datasets)
    configs = build_scaled_alpha_configs(
        args.alpha_values, soft_loss_divisor=args.soft_loss_divisor,
        k_neighbours=args.k_neighbours,
        contrastive_temperature=args.contrastive_temperature,
        positiveness_temperature=args.positiveness_temperature,
    )
    if not models:
        raise ValueError("No matching models selected.")
    if not datasets:
        raise ValueError("No matching training datasets selected.")

    output_dir = Path(args.output_dir).expanduser().absolute()
    print_settings(args, models, datasets, configs, output_dir)
    experiment.configure_determinism(args.seed)

    print("\nLoading rows and auditing the grouped split...", flush=True)
    rows = experiment.load_rows_from_csv(args.csv_path)
    split_folds = experiment.build_experiment_splits(
        rows,
        cv_folds=args.cv_folds,
        seed=args.seed,
    )
    experiment.print_dataset_counts(
        rows=split_folds[0]["rows"],
        examples=split_folds[0]["examples"],
    )
    experiment.print_split_audit(split_folds)
    if args.folds:
        requested = set(args.folds)
        split_folds = [
            split for split in split_folds if int(split["cv_fold"]) in requested
        ]

    if args.dry_run:
        print("\nDry run complete. No output or cache was created.", flush=True)
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    experiment_lock = (output_dir / ".run.lock").open("a")
    try:
        fcntl.flock(experiment_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(f"Waiting for active experiment to release {output_dir}", flush=True)
        fcntl.flock(experiment_lock, fcntl.LOCK_EX)
    print(f"Acquired experiment lock: {output_dir}", flush=True)

    if not args.resume and any(p.name != ".run.lock" for p in output_dir.iterdir()):
        raise RuntimeError("Output directory is not empty; use --resume or a new directory.")
    record_run(args, output_dir)
    candidate_cache = output_dir / "feature_cache"
    if (
        not args.resume
        and candidate_cache.exists()
        and any(candidate_cache.iterdir())
    ):
        raise RuntimeError(
            "A non-empty cache already exists. Use a new --output-dir or "
            "pass --resume for the same experiment."
        )

    experiment.configure_output_dir(str(output_dir))
    private_cache = experiment.assert_private_feature_cache(output_dir)
    experiment.ensure_output_dirs()
    print(f"\nCache isolation check passed: {private_cache}", flush=True)

    summary_path = output_dir / "experiment_summary.csv"
    cv_summary_path = output_dir / "experiment_summary_cv_averaged.csv"
    if args.resume:
        result_rows, completed_keys = load_resume_rows(summary_path)
    else:
        result_rows, completed_keys = [], set()

    training_args = SimpleNamespace(
        seed=args.seed,
        max_epochs=args.epochs,
        patience=0,
        min_delta=0.0,
        eval_every_epoch=False,
    )

    for model_name in models:
        print("\n============================================================", flush=True)
        print(f"Loading frozen base LM: {model_name}", flush=True)
        print("============================================================", flush=True)
        tokenizer, base_model = experiment.load_frozen_lm(model_name, experiment.DEVICE)
        base_model.eval()
        try:
            for config in configs:
                for train_dataset in datasets:
                    for splits in split_folds:
                        cv_fold = int(splits.get("cv_fold", 1))
                        cv_folds = int(splits.get("cv_folds", 1))
                        key = (
                            model_name,
                            str(config["name"]),
                            train_dataset,
                            experiment.normalize_seed(args.seed),
                            experiment.normalize_fold(cv_fold),
                            experiment.normalize_fold(cv_folds),
                        )
                        checkpoint = experiment.checkpoint_file(
                            config,
                            model_name,
                            train_dataset,
                            args.seed,
                            cv_fold,
                            cv_folds,
                        )
                        history = experiment.history_file(
                            config,
                            model_name,
                            train_dataset,
                            args.seed,
                            cv_fold,
                            cv_folds,
                        )
                        if (
                            args.resume
                            and key in completed_keys
                            and checkpoint.exists()
                            and history.exists()
                        ):
                            print(
                                "\nSkipping complete single-stage cell: "
                                f"{model_name} | {config['name']} | "
                                f"{train_dataset} | fold {cv_fold}/{cv_folds}",
                                flush=True,
                            )
                            continue

                        result_rows = [
                            row for row in result_rows if result_key(row) != key
                        ]
                        completed_keys.discard(key)
                        new_rows = training.run_experiment(
                            config=config,
                            model_name=model_name,
                            train_dataset=train_dataset,
                            tokenizer=tokenizer,
                            base_model=base_model,
                            splits=splits,
                            args=training_args,
                        )
                        result_rows.extend(new_rows)
                        completed_keys.add(key)
                        experiment.atomic_csv(result_rows, summary_path)
                        save_cv_summary(result_rows, cv_summary_path, args.cv_folds)
        finally:
            del base_model
            del tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    experiment.atomic_csv(result_rows, summary_path)
    save_cv_summary(result_rows, cv_summary_path, args.cv_folds)
    print("\nAll requested single-stage experiments are complete.", flush=True)


if __name__ == "__main__":
    main()
