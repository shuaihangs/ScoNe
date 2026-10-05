"""Soft semantic-neighbour contrastive objectives for SCONE representations."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _validate_inputs(
    positive_features: torch.Tensor,
    negative_features: torch.Tensor,
    neighbour_positive_features: torch.Tensor,
    neighbour_negative_features: torch.Tensor,
    k_list: torch.Tensor,
) -> list[int]:
    feature_tensors = {
        "positive_features": positive_features,
        "negative_features": negative_features,
        "neighbour_positive_features": neighbour_positive_features,
        "neighbour_negative_features": neighbour_negative_features,
    }
    for name, tensor in feature_tensors.items():
        if tensor.ndim != 2:
            raise ValueError(f"{name} must have shape [N, D], got {tensor.shape}.")

    if positive_features.shape != negative_features.shape:
        raise ValueError(
            "positive_features and negative_features must have matching shapes."
        )
    if neighbour_positive_features.shape != neighbour_negative_features.shape:
        raise ValueError(
            "Neighbour positive and negative features must have matching shapes."
        )
    if positive_features.size(1) != neighbour_positive_features.size(1):
        raise ValueError("Anchor and neighbour feature dimensions must match.")
    if k_list.ndim != 1 or k_list.numel() != positive_features.size(0):
        raise ValueError("k_list must contain one neighbour count per anchor pair.")

    counts = [int(value) for value in k_list.detach().cpu().tolist()]
    if any(value < 0 for value in counts):
        raise ValueError("Neighbour counts cannot be negative.")
    if sum(counts) != neighbour_positive_features.size(0):
        raise ValueError(
            "The sum of k_list must equal the number of flattened neighbours."
        )
    return counts


def _soft_positive_cross_entropy(
    anchor: torch.Tensor,
    positive_candidates: torch.Tensor,
    negative_candidates: torch.Tensor,
    *,
    temperature: float,
    positiveness_temperature: float,
) -> torch.Tensor:
    """Cross-entropy against a soft distribution over positive neighbours."""
    positive_similarity = positive_candidates @ anchor
    negative_similarity = negative_candidates @ anchor

    logits = torch.cat(
        [positive_similarity, negative_similarity],
        dim=0,
    ) / temperature
    log_probabilities = F.log_softmax(logits, dim=0)

    with torch.no_grad():
        positive_weights = F.softmax(
            positive_similarity.detach() / positiveness_temperature,
            dim=0,
        )

    return -(positive_weights * log_probabilities[: positive_candidates.size(0)]).sum()


def soft_neighbour_contrastive_loss(
    positive_features: torch.Tensor,
    negative_features: torch.Tensor,
    neighbour_positive_features: torch.Tensor,
    neighbour_negative_features: torch.Tensor,
    k_list: torch.Tensor,
    *,
    temperature: float = 0.2,
    positiveness_temperature: float = 0.2,
) -> torch.Tensor:
    """Shape SCONE's projection with soft, label-aware semantic neighbours."""
    if temperature <= 0.0:
        raise ValueError("temperature must be positive.")
    if positiveness_temperature <= 0.0:
        raise ValueError("positiveness_temperature must be positive.")

    counts = _validate_inputs(
        positive_features,
        negative_features,
        neighbour_positive_features,
        neighbour_negative_features,
        k_list,
    )

    positive_features = F.normalize(positive_features, p=2, dim=-1, eps=1e-8)
    negative_features = F.normalize(negative_features, p=2, dim=-1, eps=1e-8)
    neighbour_positive_features = F.normalize(
        neighbour_positive_features,
        p=2,
        dim=-1,
        eps=1e-8,
    )
    neighbour_negative_features = F.normalize(
        neighbour_negative_features,
        p=2,
        dim=-1,
        eps=1e-8,
    )

    losses = []
    start = 0
    for anchor_idx, count in enumerate(counts):
        end = start + count
        if count > 0:
            local_positive_neighbours = neighbour_positive_features[start:end]
            local_negative_neighbours = neighbour_negative_features[start:end]

            truthful_negatives = torch.cat(
                [negative_features, local_negative_neighbours],
                dim=0,
            )
            hallucinated_negatives = torch.cat(
                [positive_features, local_positive_neighbours],
                dim=0,
            )

            losses.append(
                _soft_positive_cross_entropy(
                    positive_features[anchor_idx],
                    local_positive_neighbours,
                    truthful_negatives,
                    temperature=temperature,
                    positiveness_temperature=positiveness_temperature,
                )
            )
            losses.append(
                _soft_positive_cross_entropy(
                    negative_features[anchor_idx],
                    local_negative_neighbours,
                    hallucinated_negatives,
                    temperature=temperature,
                    positiveness_temperature=positiveness_temperature,
                )
            )
        start = end

    if not losses:
        return (positive_features.sum() + negative_features.sum()) * 0.0

    return torch.stack(losses).mean()
