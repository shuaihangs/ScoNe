import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

from .utils import masked_mean
from .config import MODEL_REVISIONS


class LearnedClaimEnergy(nn.Module):
    """Projection-based EBM using contextual answer-token pooling."""

    def __init__(
        self,
        hidden_size: int,
        proj_dim: int = 64,
        dropout: float = 0.4,
        normalize_projected_states: bool = False,
        use_feature_standardization: bool = False,
        total_num_layers: int = None,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.proj_dim = proj_dim
        self.normalize_projected_states = normalize_projected_states

        self.total_num_layers = total_num_layers
        self.num_selected_layers = 5

        if total_num_layers is not None:
            self.selected_layer_indices = self.get_selected_layer_indices(
                total_num_layers
            )
        else:
            self.selected_layer_indices = None

        self.selected_layer_names = [
            "layer1",
            "early_middle",
            "middle",
            "late_middle",
            "last",
        ]

        self.proj = nn.Sequential(
            nn.Linear(hidden_size, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, proj_dim),
        )

        self.num_features = self.num_selected_layers * proj_dim

        self.feature_names = []
        for layer_name in self.selected_layer_names:
            for j in range(proj_dim):
                self.feature_names.append(f"{layer_name}_proj_{j}")

        self.register_buffer(
            "feature_mean",
            torch.zeros(self.num_features, dtype=torch.float32),
        )
        self.register_buffer(
            "feature_std",
            torch.ones(self.num_features, dtype=torch.float32),
        )
        self.register_buffer(
            "feature_standardization_enabled",
            torch.tensor(float(use_feature_standardization), dtype=torch.float32),
        )

        # Preserve the reported seed sequence after removing the unused legacy head.
        for fan_in, fan_out in [(self.num_features, 256), (256, 128), (128, 64), (64, 1)]:
            nn.Linear(fan_in, fan_out)
        self.energy_head = nn.Linear(self.num_features, 1)

    def get_selected_layer_indices(self, total_layers: int):
        if total_layers <= 0:
            raise ValueError("total_layers must be positive.")

        indices = [
            0,
            total_layers // 4,
            total_layers // 2,
            (3 * total_layers) // 4,
            total_layers - 1,
        ]

        indices = [
            max(0, min(int(idx), total_layers - 1))
            for idx in indices
        ]

        return indices

    def get_feature_names(self):
        return list(self.feature_names)

    @torch.no_grad()
    def set_feature_stats(self, mean, std, eps: float = 1e-6):
        mean = torch.as_tensor(
            mean,
            dtype=torch.float32,
            device=self.feature_mean.device,
        )
        std = torch.as_tensor(
            std,
            dtype=torch.float32,
            device=self.feature_std.device,
        )

        if mean.numel() != self.num_features:
            raise ValueError(
                f"mean must have {self.num_features} values, got {mean.numel()}"
            )

        if std.numel() != self.num_features:
            raise ValueError(
                f"std must have {self.num_features} values, got {std.numel()}"
            )

        mean = mean.reshape(self.num_features)
        std = std.reshape(self.num_features).clamp_min(eps)

        self.feature_mean.copy_(mean)
        self.feature_std.copy_(std)
        self.feature_standardization_enabled.fill_(1.0)

    def standardize_features(self, features):
        enabled = bool(self.feature_standardization_enabled.item())

        if not enabled:
            return features

        mean = self.feature_mean.to(
            device=features.device,
            dtype=features.dtype,
        )
        std = self.feature_std.to(
            device=features.device,
            dtype=features.dtype,
        )

        return (features - mean) / (std + 1e-6)

    def get_raw_layer_reprs(
        self,
        hidden_states,
        attention_mask,
        answer_mask=None,
    ):
        """Convert token-level hidden states into selected layer vectors."""

        if answer_mask is not None:
            pool_mask = answer_mask.to(
                device=attention_mask.device,
                dtype=attention_mask.dtype,
            )

            empty_answer = pool_mask.sum(dim=1) == 0

            if empty_answer.any():
                pool_mask = pool_mask.clone()
                pool_mask[empty_answer] = attention_mask[empty_answer]
        else:
            pool_mask = attention_mask

        all_layer_reprs = [
            masked_mean(h, pool_mask)
            for h in hidden_states[1:]
        ]

        total_layers = len(all_layer_reprs)

        if self.selected_layer_indices is None:
            self.total_num_layers = total_layers
            self.selected_layer_indices = self.get_selected_layer_indices(
                total_layers
            )

        selected_layer_reprs = [
            all_layer_reprs[i]
            for i in self.selected_layer_indices
        ]

        raw_layer_reprs = torch.stack(selected_layer_reprs, dim=1)

        return raw_layer_reprs

    def project_layer_reprs(self, raw_layer_reprs):
        bsz, num_layers, hidden_size = raw_layer_reprs.shape

        if num_layers != self.num_selected_layers:
            raise ValueError(
                f"Expected {self.num_selected_layers} selected layers, "
                f"got {num_layers}."
            )

        if hidden_size != self.hidden_size:
            raise ValueError(
                f"Expected hidden_size={self.hidden_size}, got {hidden_size}."
            )

        proj_param = next(self.proj.parameters())
        proj_device = proj_param.device
        proj_dtype = proj_param.dtype

        flat = raw_layer_reprs.reshape(bsz * num_layers, hidden_size)
        flat = flat.to(
            device=proj_device,
            dtype=proj_dtype,
            non_blocking=True,
        )

        projected = self.proj(flat)
        projected = projected.reshape(bsz, num_layers, self.proj_dim)

        if self.normalize_projected_states:
            projected = F.normalize(
                projected,
                p=2,
                dim=-1,
                eps=1e-8,
            )

        return projected

    def extract_features_from_raw_layer_reprs(self, raw_layer_reprs):
        projected = self.project_layer_reprs(raw_layer_reprs)

        raw_features = projected.reshape(
            projected.size(0),
            self.num_selected_layers * self.proj_dim,
        )

        raw_features = torch.nan_to_num(
            raw_features,
            nan=0.0,
            posinf=1e6,
            neginf=-1e6,
        )

        layer_norms = torch.linalg.vector_norm(projected, dim=-1)

        feature_dict = {
            "proj_layer1_norm": layer_norms[:, 0],
            "proj_early_middle_norm": layer_norms[:, 1],
            "proj_middle_norm": layer_norms[:, 2],
            "proj_late_middle_norm": layer_norms[:, 3],
            "proj_last_norm": layer_norms[:, 4],
        }

        device = raw_features.device
        dtype = raw_features.dtype

        for name, idx in zip(
            self.selected_layer_names,
            self.selected_layer_indices,
        ):
            feature_dict[f"selected_{name}_idx"] = torch.full(
                (raw_features.size(0),),
                float(idx),
                device=device,
                dtype=dtype,
            )

        return raw_features, feature_dict

    def energy_from_raw_layer_reprs(self, raw_layer_reprs):
        raw_features, feature_dict = self.extract_features_from_raw_layer_reprs(
            raw_layer_reprs
        )

        features = self.standardize_features(raw_features)

        energy_logit = self.energy_head(features).squeeze(-1)
        hallucination_prob = torch.sigmoid(energy_logit)

        return {
            **feature_dict,
            "raw_features": raw_features,
            "features": features,
            "energy_logit": energy_logit,
            "hallucination_prob": hallucination_prob,
        }

    def forward(
        self,
        hidden_states,
        attention_mask,
        answer_mask=None,
    ):
        raw_layer_reprs = self.get_raw_layer_reprs(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            answer_mask=answer_mask,
        )

        return self.energy_from_raw_layer_reprs(raw_layer_reprs)


def _is_cuda_device(device):
    return str(device).startswith("cuda")


def load_frozen_lm(model_name, device):
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=MODEL_REVISIONS[model_name])

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    is_cuda = _is_cuda_device(device)

    try:
        base_model = AutoModel.from_pretrained(
            model_name,
            revision=MODEL_REVISIONS[model_name],
            torch_dtype=torch.float16 if is_cuda else torch.float32,
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
        ).to(device)
    except TypeError:
        base_model = AutoModel.from_pretrained(
            model_name,
            revision=MODEL_REVISIONS[model_name],
            torch_dtype=torch.float16 if is_cuda else torch.float32,
            low_cpu_mem_usage=True,
        ).to(device)

    base_model.config.pad_token_id = tokenizer.pad_token_id
    base_model.config.use_cache = False

    base_model.eval()

    for p in base_model.parameters():
        p.requires_grad = False

    return tokenizer, base_model


def get_hidden_size(base_model):
    hidden_size = getattr(base_model.config, "hidden_size", None)

    if hidden_size is None:
        hidden_size = getattr(base_model.config, "n_embd", None)

    if hidden_size is None:
        raise ValueError(
            "Could not infer hidden size from base_model.config. "
            "Expected `hidden_size` or `n_embd`."
        )

    return hidden_size


def get_num_layers(base_model):
    num_layers = getattr(base_model.config, "num_hidden_layers", None)

    if num_layers is None:
        num_layers = getattr(base_model.config, "n_layer", None)

    if num_layers is None:
        raise ValueError(
            "Could not infer number of layers from base_model.config. "
            "Expected `num_hidden_layers` or `n_layer`."
        )

    return int(num_layers)


def build_energy_model(
    base_model,
    device,
    proj_dim=64,
    dropout=0.4,
    normalize_projected_states=False,
    use_feature_standardization=False,
):
    hidden_size = get_hidden_size(base_model)
    total_num_layers = get_num_layers(base_model)

    return LearnedClaimEnergy(
        hidden_size=hidden_size,
        proj_dim=proj_dim,
        dropout=dropout,
        normalize_projected_states=normalize_projected_states,
        use_feature_standardization=use_feature_standardization,
        total_num_layers=total_num_layers,
    ).to(device)


def get_hidden_states(base_model, input_ids, attention_mask, device):
    input_ids = input_ids.to(device, non_blocking=True)
    attention_mask = attention_mask.to(device, non_blocking=True)

    with torch.inference_mode():
        outputs = base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )

    return attention_mask, outputs.hidden_states


def forward_energy(
    base_model,
    energy_model,
    input_ids=None,
    attention_mask=None,
    device=None,
    answer_mask=None,
    raw_layer_reprs=None,
):
    if device is None:
        device = next(energy_model.parameters()).device

    if raw_layer_reprs is not None:
        raw_layer_reprs = raw_layer_reprs.to(device, non_blocking=True)
        return energy_model.energy_from_raw_layer_reprs(raw_layer_reprs)

    if input_ids is None or attention_mask is None:
        raise ValueError(
            "forward_energy requires either raw_layer_reprs or "
            "input_ids + attention_mask."
        )

    attention_mask, hidden_states = get_hidden_states(
        base_model,
        input_ids,
        attention_mask,
        device,
    )

    if answer_mask is not None:
        answer_mask = answer_mask.to(device, non_blocking=True)

    return energy_model(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        answer_mask=answer_mask,
    )


def load_detector_checkpoint(path, base_model, device):
    """Load the retained linear detector from either checkpoint format."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint['experiment_config']
    model = build_energy_model(base_model, device, proj_dim=64,
                               dropout=float(config['dropout']))
    state = checkpoint['model_state_dict']
    state = {key: value for key, value in state.items()
             if not key.startswith('energy_head.')}
    head = checkpoint.get('classifier_head_state_dict',
                          checkpoint.get('temporary_bce_head_state_dict'))
    if head is None:
        raise ValueError('Checkpoint has no retained linear classifier.')
    state.update({f'energy_head.{key}': value for key, value in head.items()})
    model.load_state_dict(state, strict=True)
    return model.eval()
