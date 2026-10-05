import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import experiment
from .data import build_dataloaders
from .evaluation import evaluate_loader, print_metrics
from .model import build_energy_model, forward_energy
from .soft_neighbour import soft_neighbour_contrastive_loss
from .utils import set_seed

def coefficient_slug(value: float) -> str:
    text = f"{value:.8f}".rstrip("0").rstrip(".")
    return text.replace(".", "p")


def make_config(
    soft_coefficient: float,
    *,
    k_neighbours: int,
    contrastive_temperature: float,
    positiveness_temperature: float,
) -> dict[str, object]:
    bce_coefficient = 1.0 - soft_coefficient
    uses_soft_neighbours = soft_coefficient > 0.0
    return {
        "name": (
            "stage1_bce_"
            f"{coefficient_slug(bce_coefficient)}_soft_neighbour_"
            f"{coefficient_slug(soft_coefficient)}_k{k_neighbours}"
        ),
        "neighbour_backend": "llm_hidden" if uses_soft_neighbours else "none",
        "k_neighbours": k_neighbours if uses_soft_neighbours else 0,
        "lambda_bce": bce_coefficient,
        "lambda_pair_rank": 0.0,
        "lambda_inbatch_rank": 0.0,
        "lambda_neighbour_rank": 0.0,
        "lambda_soft_neighbour_contrastive": soft_coefficient,
        "rank_margin": 0.0,
        "neighbour_margin": 0.0,
        "soft_neighbour_contrastive_temperature": contrastive_temperature,
        "soft_neighbour_positiveness_temperature": positiveness_temperature,
        "soft_neighbour_weight_source": "projected_claim_cosine_stop_gradient",
        "stage1_temporary_classifier": f"Linear({5 * experiment.PROJ_DIM}, 1)",
        "stage1_representation": f"{5 * experiment.PROJ_DIM}d_projected_claim",
        "stage1_loss_type": "bce_plus_soft_neighbour_contrastive",
        "dropout": experiment.DROPOUT,
        "weight_decay": experiment.WEIGHT_DECAY,
        "uses_bce": bce_coefficient > 0.0,
        "uses_pair": False,
        "uses_inbatch": False,
        "uses_neighbour": uses_soft_neighbours,
        "uses_soft_neighbour_contrastive": uses_soft_neighbours,
    }


def _forward_claims(batch, prefix, base_model, energy_model):
    return forward_energy(
        base_model,
        energy_model,
        batch.get(f"{prefix}_input_ids"),
        batch.get(f"{prefix}_attention_mask"),
        experiment.DEVICE,
        answer_mask=batch.get(f"{prefix}_answer_mask"),
        raw_layer_reprs=batch.get(f"{prefix}_raw_layer_reprs"),
    )


def train_epoch(
    *,
    loader,
    base_model,
    energy_model,
    classifier_head,
    optimizer,
    config: dict[str, object],
) -> dict[str, float]:
    base_model.eval()
    energy_model.proj.train()
    energy_model.energy_head.eval()
    classifier_head.train()

    lambda_bce = float(config["lambda_bce"])
    lambda_soft = float(config["lambda_soft_neighbour_contrastive"])
    total_weighted_loss = 0.0
    total_bce_loss = 0.0
    total_soft_loss = 0.0
    total_pairs = 0

    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        positive_output = _forward_claims(batch, "pos", base_model, energy_model)
        negative_output = _forward_claims(batch, "neg", base_model, energy_model)

        positive_logits = classifier_head(
            positive_output["features"]
        ).squeeze(-1)
        negative_logits = classifier_head(
            negative_output["features"]
        ).squeeze(-1)
        bce_loss = 0.5 * (
            F.binary_cross_entropy_with_logits(
                positive_logits,
                torch.zeros_like(positive_logits),
            )
            + F.binary_cross_entropy_with_logits(
                negative_logits,
                torch.ones_like(negative_logits),
            )
        )

        soft_loss = torch.zeros(
            (),
            device=positive_logits.device,
            dtype=positive_logits.dtype,
        )
        if lambda_soft > 0.0:
            if not bool(batch.get("has_neighbours", False)):
                raise RuntimeError("Active soft-neighbour loss received no neighbours.")
            neighbour_positive_output = _forward_claims(
                batch,
                "neigh_pos",
                base_model,
                energy_model,
            )
            neighbour_negative_output = _forward_claims(
                batch,
                "neigh_neg",
                base_model,
                energy_model,
            )
            soft_loss = soft_neighbour_contrastive_loss(
                positive_output["raw_features"],
                negative_output["raw_features"],
                neighbour_positive_output["raw_features"],
                neighbour_negative_output["raw_features"],
                batch["k_list"],
                temperature=float(
                    config["soft_neighbour_contrastive_temperature"]
                ),
                positiveness_temperature=float(
                    config["soft_neighbour_positiveness_temperature"]
                ),
            )

        total_loss = lambda_bce * bce_loss + lambda_soft * soft_loss
        if not torch.isfinite(total_loss):
            raise FloatingPointError(
                "Non-finite training loss: "
                f"BCE={float(bce_loss.detach())}, "
                f"soft={float(soft_loss.detach())}."
            )
        total_loss.backward()
        trainable_parameters = [
            *energy_model.proj.parameters(),
            *classifier_head.parameters(),
        ]
        torch.nn.utils.clip_grad_norm_(trainable_parameters, 1.0)
        optimizer.step()

        pair_count = int(positive_logits.numel())
        total_weighted_loss += float(total_loss.detach().cpu()) * pair_count
        total_bce_loss += float(bce_loss.detach().cpu()) * pair_count
        total_soft_loss += float(soft_loss.detach().cpu()) * pair_count
        total_pairs += pair_count

    denominator = max(total_pairs, 1)
    return {
        "total_loss": total_weighted_loss / denominator,
        "bce_loss": total_bce_loss / denominator,
        "soft_neighbour_contrastive_loss": total_soft_loss / denominator,
    }


def build_loaders(
    *,
    config: dict[str, object],
    model_name: str,
    train_dataset: str,
    tokenizer,
    base_model,
    energy_model,
    splits: dict[str, object],
):
    ood_datasets = [
        dataset_name
        for dataset_name in experiment.DATASET_NAMES
        if dataset_name != train_dataset
    ]
    feature_cache_path = None
    if experiment.CACHE_FROZEN_LLM_FEATURES:
        feature_cache_path = os.path.join(
            experiment.FEATURE_CACHE_DIR,
            (
                "raw_layer_reprs_"
                f"{experiment.slugify(model_name)}"
                f"_max{experiment.MAX_LENGTH}"
                f"_short{int(bool(experiment.USE_SHORT_ANSWER_IN_TEXT))}.pt"
            ),
        )
        print(f"Frozen feature cache path: {feature_cache_path}", flush=True)

    loaders = build_dataloaders(
        splits=splits,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_datasets=ood_datasets,
        max_length=experiment.MAX_LENGTH,
        batch_size=experiment.BATCH_SIZE,
        use_short_answer=experiment.USE_SHORT_ANSWER_IN_TEXT,
        num_workers=0,
        k_neighbours=int(config["k_neighbours"]),
        neighbour_backend=str(config["neighbour_backend"]),
        neighbour_llm_base_model=base_model,
        neighbour_llm_device=experiment.DEVICE,
        neighbour_llm_batch_size=experiment.NEIGHBOUR_LLM_BATCH_SIZE,
        cache_frozen_features=experiment.CACHE_FROZEN_LLM_FEATURES,
        feature_cache_base_model=base_model,
        feature_cache_energy_model=energy_model,
        feature_cache_device=experiment.DEVICE,
        feature_cache_path=feature_cache_path,
        feature_cache_batch_size=experiment.FEATURE_CACHE_BATCH_SIZE,
    )
    experiment.sanity_check_train_loader(loaders)
    return loaders


def run_experiment(
    config,
    model_name,
    train_dataset,
    tokenizer,
    base_model,
    splits,
    args,
):
    cv_fold = int(splits.get("cv_fold", 1))
    cv_folds = int(splits.get("cv_folds", 1))
    training_seed = int(args.seed) + cv_fold - 1
    set_seed(training_seed)

    print("\n==============================================", flush=True)
    print(f"Configuration: {config['name']}", flush=True)
    print(f"Base model: {model_name}", flush=True)
    print(f"Train dataset: {train_dataset}", flush=True)
    print(f"Training seed: {training_seed}", flush=True)
    print(f"CV fold: {cv_fold}/{cv_folds}", flush=True)
    print("==============================================", flush=True)

    energy_model = build_energy_model(
        base_model=base_model,
        device=experiment.DEVICE,
        proj_dim=experiment.PROJ_DIM,
        dropout=float(config["dropout"]),
        normalize_projected_states=experiment.NORMALIZE_PROJECTED_STATES,
        use_feature_standardization=experiment.USE_FEATURE_STANDARDIZATION,
    )
    if energy_model.num_features != 5 * experiment.PROJ_DIM:
        raise AssertionError(
            f"Expected {5 * experiment.PROJ_DIM} projected features, "
            f"got {energy_model.num_features}."
        )
    classifier_head = energy_model.energy_head

    loaders = build_loaders(
        config=config,
        model_name=model_name,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        base_model=base_model,
        energy_model=energy_model,
        splits=splits,
    )
    optimizer = torch.optim.AdamW(
        [
            *energy_model.proj.parameters(),
            *classifier_head.parameters(),
        ],
        lr=experiment.LR,
        weight_decay=float(config["weight_decay"]),
    )

    history_rows = []
    for epoch in range(int(args.max_epochs)):
        losses = train_epoch(
            loader=loaders["train"],
            base_model=base_model,
            energy_model=energy_model,
            classifier_head=classifier_head,
            optimizer=optimizer,
            config=config,
        )
        row = {
            "epoch": epoch,
            "model_name": model_name,
            "train_dataset": train_dataset,
            "config_name": config["name"],
            "seed": args.seed,
            "training_seed": training_seed,
            "cv_fold": cv_fold,
            "cv_folds": cv_folds,
            "lambda_bce": config["lambda_bce"],
            "lambda_soft_neighbour_contrastive": config[
                "lambda_soft_neighbour_contrastive"
            ],
            "soft_neighbour_contrastive_temperature": config[
                "soft_neighbour_contrastive_temperature"
            ],
            "soft_neighbour_positiveness_temperature": config[
                "soft_neighbour_positiveness_temperature"
            ],
            **losses,
        }
        for metadata_key in (
            "loss_mixing_alpha",
            "soft_neighbour_loss_divisor",
        ):
            if metadata_key in config:
                row[metadata_key] = config[metadata_key]
        history_rows.append(row)
        print(
            f"Epoch {epoch:03d} | Objective={losses['total_loss']:.4f} | "
            f"BCE={losses['bce_loss']:.4f} | "
            "SoftNeighbourContrastive="
            f"{losses['soft_neighbour_contrastive_loss']:.4f}",
            flush=True,
        )

    history_path = experiment.history_file(
        config,
        model_name,
        train_dataset,
        args.seed,
        cv_fold,
        cv_folds,
    )
    experiment.atomic_csv(history_rows, history_path)

    checkpoint_path = experiment.checkpoint_file(
        config,
        model_name,
        train_dataset,
        args.seed,
        cv_fold,
        cv_folds,
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": int(args.max_epochs) - 1,
            "model_state_dict": energy_model.state_dict(),
            "classifier_head_state_dict": classifier_head.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model_name": model_name,
            "train_dataset": train_dataset,
            "config_name": config["name"],
            "experiment_config": dict(config),
            "seed": args.seed,
            "training_seed": training_seed,
            "cv_fold": cv_fold,
            "cv_folds": cv_folds,
            "loss_type": config["stage1_loss_type"],
            "lambda_bce": config["lambda_bce"],
            "lambda_soft_neighbour_contrastive": config[
                "lambda_soft_neighbour_contrastive"
            ],
            "stage1_temporary_classifier": config["stage1_temporary_classifier"],
            "classifier_role": (
                "final_joint_classifier"
            ),
            "final_classifier_retained": True,
            "stage1_original_nonlinear_head_trained": False,
            "eval_every_epoch": False,
            "early_stopping_patience": None,
            "monitor_auc": float("nan"),
            "mean_eval_auc": float("nan"),
        },
        checkpoint_path,
    )
    print(f"Saved final checkpoint: {checkpoint_path}", flush=True)

    energy_model.energy_head = classifier_head
    energy_model.eval()
    common = {
        "config_name": config["name"],
        "model_name": model_name,
        "train_dataset": train_dataset,
        "seed": args.seed,
        "training_seed": training_seed,
        "cv_fold": cv_fold,
        "cv_folds": cv_folds,
        "split_strategy": splits["split_strategy"],
        "checkpoint_path": str(checkpoint_path),
        "best_epoch": int(args.max_epochs) - 1,
        "stopped_epoch": int(args.max_epochs) - 1,
        "early_stopped": False,
        "max_epochs": int(args.max_epochs),
        "lambda_bce": config["lambda_bce"],
        "lambda_pair_rank": 0.0,
        "lambda_inbatch_rank": 0.0,
        "lambda_neighbour_rank": 0.0,
        "lambda_soft_neighbour_contrastive": config[
            "lambda_soft_neighbour_contrastive"
        ],
        "neighbour_backend": config["neighbour_backend"],
        "k_neighbours": config["k_neighbours"],
        "soft_neighbour_contrastive_temperature": config[
            "soft_neighbour_contrastive_temperature"
        ],
        "soft_neighbour_positiveness_temperature": config[
            "soft_neighbour_positiveness_temperature"
        ],
        "soft_neighbour_weight_source": config["soft_neighbour_weight_source"],
        "stage1_temporary_classifier": config["stage1_temporary_classifier"],
        "stage1_original_nonlinear_head_trained": False,
        "stage": (
            "single_stage_joint_classifier"
        ),
        "classifier_role": (
            "final_joint_classifier"
        ),
        "final_classifier_retained": True,
        "projection_frozen": False,
    }
    for metadata_key in (
        "loss_mixing_alpha",
        "soft_neighbour_loss_divisor",
    ):
        if metadata_key in config:
            common[metadata_key] = config[metadata_key]
    result_rows = []
    print("\nFinal evaluation:", flush=True)
    evaluation_datasets = list(dict.fromkeys(
        [*loaders["eval_datasets"], *loaders["monitor_datasets"]]
    ))
    for eval_dataset in evaluation_datasets:
        metrics = evaluate_loader(
            loaders["eval"][eval_dataset],
            base_model,
            energy_model,
            experiment.DEVICE,
        )
        print_metrics(eval_dataset, metrics)
        result_rows.append({**common, "eval_dataset": eval_dataset, **metrics})

    del loaders
    del optimizer
    del classifier_head
    del energy_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result_rows
