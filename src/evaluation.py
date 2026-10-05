import torch
import torch.nn.functional as F

from sklearn.metrics import accuracy_score, roc_auc_score

from .model import forward_energy


def safe_auc(labels, scores):
    """Safe ROC-AUC."""
    try:
        return float(roc_auc_score(labels, scores))
    except Exception:
        return float("nan")


def safe_mean(x):
    """Safe mean for numpy arrays / tensors."""
    try:
        if len(x) == 0:
            return float("nan")
        return float(x.mean())
    except Exception:
        return float("nan")


@torch.no_grad()
def evaluate_loader(
    loader,
    base_model,
    energy_model,
    device,
):
    """Evaluate hallucination detection."""

    base_model.eval()
    energy_model.eval()

    all_labels = []
    all_logits = []
    all_probs = []

    all_pos_logits = []
    all_neg_logits = []

    scalar_logs = {}

    total_loss = 0.0
    total_count = 0

    def collect_scalar_logs(out):
        """Collect compact scalar diagnostics from model output."""
        skip_keys = {
            "energy_logit",
            "hallucination_prob",
            "raw_features",
            "features",
        }

        for key, value in out.items():
            if key in skip_keys:
                continue

            if torch.is_tensor(value) and value.dim() == 1:
                if key not in scalar_logs:
                    scalar_logs[key] = []
                scalar_logs[key].append(value.detach().cpu())

    for batch in loader:
        if "pos_input_ids" in batch and "neg_input_ids" in batch:
            pos_out = forward_energy(
                base_model,
                energy_model,
                batch.get("pos_input_ids", None),
                batch.get("pos_attention_mask", None),
                device,
                answer_mask=batch.get("pos_answer_mask", None),
                raw_layer_reprs=batch.get("pos_raw_layer_reprs", None),
            )

            neg_out = forward_energy(
                base_model,
                energy_model,
                batch.get("neg_input_ids", None),
                batch.get("neg_attention_mask", None),
                device,
                answer_mask=batch.get("neg_answer_mask", None),
                raw_layer_reprs=batch.get("neg_raw_layer_reprs", None),
            )

            pos_logits = pos_out["energy_logit"]
            neg_logits = neg_out["energy_logit"]

            pos_labels = torch.zeros_like(pos_logits)
            neg_labels = torch.ones_like(neg_logits)

            logits = torch.cat([pos_logits, neg_logits], dim=0)
            labels = torch.cat([pos_labels, neg_labels], dim=0)

            loss = F.binary_cross_entropy_with_logits(
                logits,
                labels,
            )

            probs = torch.sigmoid(logits)

            bs = labels.size(0)
            total_loss += float(loss.detach().cpu().item()) * bs
            total_count += bs

            all_labels.append(labels.detach().cpu())
            all_logits.append(logits.detach().cpu())
            all_probs.append(probs.detach().cpu())

            all_pos_logits.append(pos_logits.detach().cpu())
            all_neg_logits.append(neg_logits.detach().cpu())

            collect_scalar_logs(pos_out)
            collect_scalar_logs(neg_out)

        elif (
            "input_ids" in batch or "raw_layer_reprs" in batch
        ) and "labels" in batch:
            labels = batch["labels"].to(device).float()

            out = forward_energy(
                base_model,
                energy_model,
                batch.get("input_ids", None),
                batch.get("attention_mask", None),
                device,
                answer_mask=batch.get("answer_mask", None),
                raw_layer_reprs=batch.get("raw_layer_reprs", None),
            )

            logits = out["energy_logit"]
            probs = out["hallucination_prob"]

            loss = F.binary_cross_entropy_with_logits(
                logits,
                labels,
            )

            bs = labels.size(0)
            total_loss += float(loss.detach().cpu().item()) * bs
            total_count += bs

            all_labels.append(labels.detach().cpu())
            all_logits.append(logits.detach().cpu())
            all_probs.append(probs.detach().cpu())

            collect_scalar_logs(out)

        else:
            raise KeyError(
                "Unsupported batch format. Expected either paired keys "
                "pos_input_ids/neg_input_ids or single keys input_ids/labels."
            )

    all_labels = torch.cat(all_labels, dim=0).float().numpy()
    all_logits = torch.cat(all_logits, dim=0).float().numpy()
    all_probs = torch.cat(all_probs, dim=0).float().numpy()

    pred_labels = (all_probs >= 0.5).astype(int)

    pos_mask = all_labels == 0
    neg_mask = all_labels == 1

    metrics = {
        "loss": total_loss / max(total_count, 1),
        "accuracy": float(accuracy_score(all_labels, pred_labels)),

        "energy_auc": safe_auc(all_labels, all_probs),
        "logit_auc": safe_auc(all_labels, all_logits),

        "mean_prob_positive": safe_mean(all_probs[pos_mask]),
        "mean_prob_negative": safe_mean(all_probs[neg_mask]),

        "mean_logit_positive": safe_mean(all_logits[pos_mask]),
        "mean_logit_negative": safe_mean(all_logits[neg_mask]),

        "mean_pos_energy": safe_mean(all_logits[pos_mask]),
        "mean_neg_energy": safe_mean(all_logits[neg_mask]),
        "energy_gap": (
            safe_mean(all_logits[neg_mask])
            -
            safe_mean(all_logits[pos_mask])
        ),
    }

    if len(all_pos_logits) > 0 and len(all_neg_logits) > 0:
        pos_logits_all = torch.cat(all_pos_logits, dim=0).float()
        neg_logits_all = torch.cat(all_neg_logits, dim=0).float()

        metrics["pairwise_accuracy"] = float(
            (pos_logits_all < neg_logits_all).float().mean().item()
        )

    for key, values in scalar_logs.items():
        values = torch.cat(values, dim=0).float()
        metrics[f"{key}_mean"] = float(values.mean().item())
        metrics[f"{key}_std"] = float(values.std(unbiased=False).item())

    return metrics


def print_metrics(name, metrics):
    """Print core metrics safely."""

    print(f"\n=== {name} ===")

    print(f"Loss:                         {metrics.get('loss', float('nan')):.4f}")
    print(f"Accuracy at threshold 0.5:    {metrics.get('accuracy', float('nan')):.4f}")
    print(f"ROC-AUC using P(hallucinated): {metrics.get('energy_auc', float('nan')):.4f}")
    print(f"ROC-AUC using raw energy logit:{metrics.get('logit_auc', float('nan')):.4f}")

    if "pairwise_accuracy" in metrics:
        print(f"Pairwise accuracy:            {metrics['pairwise_accuracy']:.4f}")

    print(f"Mean P(hallu) positive:       {metrics.get('mean_prob_positive', float('nan')):.4f}")
    print(f"Mean P(hallu) negative:       {metrics.get('mean_prob_negative', float('nan')):.4f}")

    print(f"Mean pos energy/logit:        {metrics.get('mean_pos_energy', float('nan')):.4f}")
    print(f"Mean neg energy/logit:        {metrics.get('mean_neg_energy', float('nan')):.4f}")
    print(f"Energy gap:                   {metrics.get('energy_gap', float('nan')):.4f}")

    optional_keys = [
        "proj_layer1_norm_mean",
        "proj_early_middle_norm_mean",
        "proj_middle_norm_mean",
        "proj_late_middle_norm_mean",
        "proj_last_norm_mean",

        "selected_layer1_idx_mean",
        "selected_early_middle_idx_mean",
        "selected_middle_idx_mean",
        "selected_late_middle_idx_mean",
        "selected_last_idx_mean",
    ]

    for key in optional_keys:
        if key in metrics:
            print(f"{key}: {metrics[key]:.4f}")
