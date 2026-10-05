import csv
import hashlib
import os
import random
from typing import Dict, List, Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.neighbors import NearestNeighbors


_NEIGHBOUR_INDEX_CACHE = {}
QUESTION_GROUPED_SPLIT_STRATEGY = "question_grouped_seed_plus_dataset_index"
QUESTION_GROUPED_CV_SPLIT_STRATEGY = (
    "question_grouped_outer_holdout_inner_kfold_seed_plus_dataset_index"
)


def question_identity(question: str) -> str:
    """Canonical identity used to keep repeated question variants together."""
    return " ".join(str(question).split()).casefold()


def get_selected_layer_indices(total_layers: int):
    if total_layers <= 0:
        raise ValueError("total_layers must be positive.")

    indices = [
        0,
        total_layers // 4,
        total_layers // 2,
        (3 * total_layers) // 4,
        total_layers - 1,
    ]

    return [
        max(0, min(int(idx), total_layers - 1))
        for idx in indices
    ]


@torch.no_grad()
def build_llm_hidden_question_embeddings(
    questions,
    tokenizer,
    base_model,
    device,
    batch_size=16,
    max_length=128,
):
    """Build question-only KNN features from the frozen base LLM hidden states."""

    if tokenizer is None or base_model is None:
        raise ValueError(
            "llm_hidden neighbours require both tokenizer and base_model."
        )

    base_model.eval()
    device = torch.device(device)

    texts = [f"Question: {q}" for q in questions]
    embeddings = []
    selected_indices = None

    total_batches = (len(texts) + batch_size - 1) // batch_size

    for start in range(0, len(texts), batch_size):
        end = min(start + batch_size, len(texts))
        batch_texts = texts[start:end]
        batch_id = start // batch_size + 1

        if batch_id == 1 or batch_id == total_batches or batch_id % 25 == 0:
            print(
                "  LLM-hidden neighbour embedding batch "
                f"{batch_id}/{total_batches}",
                flush=True,
            )

        enc = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc = {
            key: value.to(device)
            for key, value in enc.items()
        }

        outputs = base_model(
            **enc,
            output_hidden_states=True,
            use_cache=False,
        )

        hidden_states = outputs.hidden_states[1:]
        total_layers = len(hidden_states)

        if selected_indices is None:
            selected_indices = get_selected_layer_indices(total_layers)
            print(
                "  LLM-hidden selected layer indices: "
                f"{selected_indices}",
                flush=True,
            )

        attention_mask = enc["attention_mask"]
        mask = attention_mask.unsqueeze(-1)

        pooled_layers = []

        for layer_idx in selected_indices:
            hidden = hidden_states[layer_idx]
            layer_mask = mask.to(
                device=hidden.device,
                dtype=hidden.dtype,
            )
            denom = layer_mask.sum(dim=1).clamp_min(1.0)
            pooled = (hidden * layer_mask).sum(dim=1) / denom
            pooled_layers.append(pooled.float())

        batch_embeddings = torch.cat(pooled_layers, dim=-1)
        batch_embeddings = F.normalize(
            batch_embeddings,
            p=2,
            dim=-1,
            eps=1e-8,
        )
        embeddings.append(batch_embeddings.cpu())

    return torch.cat(embeddings, dim=0).numpy()


def load_rows_from_csv(csv_path: str) -> List[Dict[str, Any]]:
    rows = []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            dataset = str(row.get("dataset", "")).strip()
            question = str(row.get("question", "")).strip()
            short_answer = str(row.get("short_answer", "")).strip()
            positive = str(row.get("positive", "")).strip()
            negative = str(row.get("negative", "")).strip()

            if not dataset or not question or not positive or not negative:
                continue

            rows.append(
                {
                    "dataset": dataset,
                    "question": question,
                    "short_answer": short_answer,
                    "positive": positive,
                    "negative": negative,
                }
            )

    return rows


def normalize_dataset_key(dataset_name: str) -> str:
    name = str(dataset_name).strip().lower()
    name = name.replace("-", "_")
    name = name.replace(" ", "_")

    if "hotpot" in name:
        return "hotpotqa"

    if "truthful" in name:
        return "truthfulqa"

    if "halueval" in name or "halu_eval" in name or "hallu_eval" in name:
        return "halueval"

    if "trivia" in name:
        return "triviaqa"

    if "squad" in name:
        return "squadqa"

    return name


def rows_to_claim_examples(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    examples = []

    for row in rows:
        base = {
            "dataset": row["dataset"],
            "question": row["question"],
            "short_answer": row.get("short_answer", ""),
        }

        examples.append(
            {
                **base,
                "claim": row["positive"],
                "label": 0.0,
                "label_name": "truthful",
            }
        )

        examples.append(
            {
                **base,
                "claim": row["negative"],
                "label": 1.0,
                "label_name": "hallucinated",
            }
        )

    return examples


def split_rows_by_question(
    rows: List[Dict[str, Any]],
    validation_ratio: float,
    rng: random.Random,
):
    """Split rows while keeping every normalized question in one partition."""
    if not rows:
        return [], []

    question_groups = {}
    for row in rows:
        key = question_identity(row["question"])
        question_groups.setdefault(key, []).append(row)

    groups = list(question_groups.values())
    rng.shuffle(groups)

    if validation_ratio <= 0.0 or len(groups) <= 1:
        return [row for group in groups for row in group], []

    target_validation_rows = max(1, int(len(rows) * validation_ratio))
    target_validation_rows = min(target_validation_rows, len(rows) - 1)

    validation_groups = []
    train_groups = []
    validation_row_count = 0

    for group in groups:
        if validation_row_count + len(group) <= target_validation_rows:
            validation_groups.append(group)
            validation_row_count += len(group)
        else:
            train_groups.append(group)

    if not validation_groups:
        validation_groups.append(train_groups.pop(0))

    validation_rows = [
        row
        for group in validation_groups
        for row in group
    ]
    train_rows = [
        row
        for group in train_groups
        for row in group
    ]

    return train_rows, validation_rows


def split_rows_by_question_kfold(
    rows: List[Dict[str, Any]],
    n_splits: int,
    rng: random.Random,
):
    """Build balanced K-fold partitions without splitting question groups."""
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2 for cross-validation.")
    if not rows:
        raise ValueError("Cannot build cross-validation folds from no rows.")

    question_groups = {}
    for row in rows:
        key = question_identity(row["question"])
        question_groups.setdefault(key, []).append(row)

    groups = list(question_groups.values())
    if len(groups) < n_splits:
        raise ValueError(
            f"Cannot create {n_splits} folds from only "
            f"{len(groups)} unique question groups."
        )

    rng.shuffle(groups)
    groups.sort(key=len, reverse=True)

    fold_groups = [[] for _ in range(n_splits)]
    fold_row_counts = [0 for _ in range(n_splits)]

    for group in groups:
        fold_idx = min(
            range(n_splits),
            key=lambda idx: (
                fold_row_counts[idx],
                len(fold_groups[idx]),
                idx,
            ),
        )
        fold_groups[fold_idx].append(group)
        fold_row_counts[fold_idx] += len(group)

    folds = []
    for validation_fold_idx in range(n_splits):
        validation_rows = [
            row
            for group in fold_groups[validation_fold_idx]
            for row in group
        ]
        train_rows = [
            row
            for fold_idx, groups_in_fold in enumerate(fold_groups)
            if fold_idx != validation_fold_idx
            for group in groups_in_fold
            for row in group
        ]
        folds.append((train_rows, validation_rows))

    return folds


def split_examples_by_dataset(
    rows: List[Dict[str, Any]],
    dataset_names=None,
    validation_ratio: float = 0.2,
    seed: int = 42,
):
    if dataset_names is None:
        dataset_names = sorted({
            normalize_dataset_key(row["dataset"])
            for row in rows
        })
    else:
        dataset_names = [
            normalize_dataset_key(name)
            for name in dataset_names
        ]

    datasets = {
        name: {
            "rows": [],
            "examples": [],
        }
        for name in dataset_names
    }

    for row in rows:
        dataset_key = normalize_dataset_key(row["dataset"])

        if dataset_key not in datasets:
            continue

        datasets[dataset_key]["rows"].append(row)

    all_examples = []

    for dataset_idx, dataset in enumerate(datasets.values()):
        dataset_rows = list(dataset["rows"])
        rng = random.Random(seed + dataset_idx)
        train_rows, validation_rows = split_rows_by_question(
            rows=dataset_rows,
            validation_ratio=validation_ratio,
            rng=rng,
        )

        dataset["train_rows"] = train_rows
        dataset["validation_rows"] = validation_rows
        dataset["examples"] = rows_to_claim_examples(dataset["rows"])
        dataset["train_examples"] = rows_to_claim_examples(train_rows)
        dataset["validation_examples"] = rows_to_claim_examples(validation_rows)
        all_examples.extend(dataset["examples"])

    return {
        "rows": rows,
        "examples": all_examples,
        "dataset_names": dataset_names,
        "split_strategy": QUESTION_GROUPED_SPLIT_STRATEGY,
        "cv_fold": 1,
        "cv_folds": 1,
        "datasets": datasets,
    }


def split_examples_by_dataset_kfold(
    rows: List[Dict[str, Any]],
    dataset_names=None,
    n_splits: int = 5,
    validation_ratio: float = 0.2,
    seed: int = 42,
):
    """Split 80/20, then return grouped CV folds from only the outer 80%."""
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2 for cross-validation.")

    if dataset_names is None:
        dataset_names = sorted({
            normalize_dataset_key(row["dataset"])
            for row in rows
        })
    else:
        dataset_names = [
            normalize_dataset_key(name)
            for name in dataset_names
        ]

    rows_by_dataset = {name: [] for name in dataset_names}
    for row in rows:
        dataset_key = normalize_dataset_key(row["dataset"])
        if dataset_key in rows_by_dataset:
            rows_by_dataset[dataset_key].append(row)

    outer_partitions_by_dataset = {}
    folds_by_dataset = {}
    for dataset_idx, dataset_name in enumerate(dataset_names):
        outer_train_rows, test_rows = split_rows_by_question(
            rows=rows_by_dataset[dataset_name],
            validation_ratio=validation_ratio,
            rng=random.Random(seed + dataset_idx),
        )
        if not outer_train_rows or not test_rows:
            raise ValueError(
                f"{dataset_name} requires non-empty outer train and test "
                "partitions for nested cross-validation."
            )
        outer_partitions_by_dataset[dataset_name] = (
            outer_train_rows,
            test_rows,
        )
        folds_by_dataset[dataset_name] = split_rows_by_question_kfold(
            rows=outer_train_rows,
            n_splits=n_splits,
            rng=random.Random(seed + dataset_idx),
        )

    all_examples = rows_to_claim_examples([
        row
        for dataset_name in dataset_names
        for row in rows_by_dataset[dataset_name]
    ])
    split_folds = []

    for fold_idx in range(n_splits):
        datasets = {}
        for dataset_name in dataset_names:
            dataset_rows = rows_by_dataset[dataset_name]
            outer_train_rows, test_rows = outer_partitions_by_dataset[
                dataset_name
            ]
            train_rows, validation_rows = folds_by_dataset[dataset_name][fold_idx]
            datasets[dataset_name] = {
                "rows": dataset_rows,
                "examples": rows_to_claim_examples(dataset_rows),
                "outer_train_rows": outer_train_rows,
                "outer_train_examples": rows_to_claim_examples(outer_train_rows),
                "train_rows": train_rows,
                "validation_rows": validation_rows,
                "test_rows": test_rows,
                "train_examples": rows_to_claim_examples(train_rows),
                "validation_examples": rows_to_claim_examples(validation_rows),
                "test_examples": rows_to_claim_examples(test_rows),
            }

        split_folds.append(
            {
                "rows": rows,
                "examples": all_examples,
                "dataset_names": dataset_names,
                "split_strategy": QUESTION_GROUPED_CV_SPLIT_STRATEGY,
                "outer_validation_ratio": validation_ratio,
                "cv_fold": fold_idx + 1,
                "cv_folds": n_splits,
                "datasets": datasets,
            }
        )

    validate_grouped_cv_splits(split_folds)
    return split_folds


def validate_grouped_cv_splits(split_folds):
    """Validate fixed outer holdout and grouped folds within the outer train set."""
    if len(split_folds) < 2:
        raise ValueError("Cross-validation requires at least two folds.")

    expected_fold_count = len(split_folds)
    dataset_names = list(split_folds[0]["dataset_names"])

    for fold_idx, split in enumerate(split_folds, start=1):
        if split.get("cv_fold") != fold_idx:
            raise ValueError(
                f"Expected cv_fold={fold_idx}, got {split.get('cv_fold')}."
            )
        if split.get("cv_folds") != expected_fold_count:
            raise ValueError("Inconsistent cv_folds metadata.")
        if list(split["dataset_names"]) != dataset_names:
            raise ValueError("Inconsistent dataset ordering across CV folds.")

    for dataset_name in dataset_names:
        expected_rows = split_folds[0]["datasets"][dataset_name]["rows"]
        expected_row_ids = {id(row) for row in expected_rows}
        expected_questions = {
            question_identity(row["question"])
            for row in expected_rows
        }
        first_dataset = split_folds[0]["datasets"][dataset_name]
        expected_outer_train_rows = first_dataset["outer_train_rows"]
        expected_test_rows = first_dataset["test_rows"]
        expected_outer_train_row_ids = {
            id(row)
            for row in expected_outer_train_rows
        }
        expected_test_row_ids = {id(row) for row in expected_test_rows}
        expected_outer_train_questions = {
            question_identity(row["question"])
            for row in expected_outer_train_rows
        }
        expected_test_questions = {
            question_identity(row["question"])
            for row in expected_test_rows
        }

        if not expected_outer_train_rows or not expected_test_rows:
            raise ValueError(
                f"{dataset_name} has an empty outer train or test partition."
            )
        if expected_outer_train_row_ids & expected_test_row_ids:
            raise ValueError(f"Outer row leakage in {dataset_name}.")
        if expected_outer_train_questions & expected_test_questions:
            raise ValueError(f"Outer question leakage in {dataset_name}.")
        if (
            expected_outer_train_row_ids | expected_test_row_ids
            != expected_row_ids
        ):
            raise ValueError(f"Incomplete outer row coverage in {dataset_name}.")
        if (
            expected_outer_train_questions | expected_test_questions
            != expected_questions
        ):
            raise ValueError(
                f"Incomplete outer question coverage in {dataset_name}."
            )

        validation_row_counts = {
            row_id: 0
            for row_id in expected_outer_train_row_ids
        }
        validation_question_counts = {
            question: 0
            for question in expected_outer_train_questions
        }

        for split in split_folds:
            dataset = split["datasets"][dataset_name]
            outer_train_rows = dataset["outer_train_rows"]
            train_rows = dataset["train_rows"]
            validation_rows = dataset["validation_rows"]
            test_rows = dataset["test_rows"]
            if not train_rows or not validation_rows or not test_rows:
                raise ValueError(
                    f"{dataset_name} has an empty partition in "
                    f"fold {split['cv_fold']}."
                )

            outer_train_row_ids = {id(row) for row in outer_train_rows}
            test_row_ids = {id(row) for row in test_rows}
            if outer_train_row_ids != expected_outer_train_row_ids:
                raise ValueError(
                    f"Outer train partition changed in {dataset_name}, fold "
                    f"{split['cv_fold']}."
                )
            if test_row_ids != expected_test_row_ids:
                raise ValueError(
                    f"Outer test partition changed in {dataset_name}, fold "
                    f"{split['cv_fold']}."
                )

            train_questions = {
                question_identity(row["question"])
                for row in train_rows
            }
            validation_questions = {
                question_identity(row["question"])
                for row in validation_rows
            }
            test_questions = {
                question_identity(row["question"])
                for row in test_rows
            }
            if not train_questions.isdisjoint(validation_questions):
                raise ValueError(
                    f"Inner question leakage in {dataset_name}, fold "
                    f"{split['cv_fold']}."
                )
            if not train_questions.isdisjoint(test_questions):
                raise ValueError(
                    f"Train/test question leakage in {dataset_name}, fold "
                    f"{split['cv_fold']}."
                )
            if not validation_questions.isdisjoint(test_questions):
                raise ValueError(
                    f"Validation/test question leakage in {dataset_name}, "
                    f"fold {split['cv_fold']}."
                )
            if (
                train_questions | validation_questions
                != expected_outer_train_questions
            ):
                raise ValueError(
                    f"Incomplete inner question coverage in {dataset_name}, fold "
                    f"{split['cv_fold']}."
                )

            train_row_ids = {id(row) for row in train_rows}
            validation_row_ids = {id(row) for row in validation_rows}
            if train_row_ids & validation_row_ids:
                raise ValueError(
                    f"Inner row leakage in {dataset_name}, fold "
                    f"{split['cv_fold']}."
                )
            if (
                train_row_ids | validation_row_ids
                != expected_outer_train_row_ids
            ):
                raise ValueError(
                    f"Incomplete inner row coverage in {dataset_name}, fold "
                    f"{split['cv_fold']}."
                )

            for row in validation_rows:
                validation_row_counts[id(row)] += 1
            for question in validation_questions:
                validation_question_counts[question] += 1

        if set(validation_row_counts.values()) != {1}:
            raise ValueError(
                f"Every outer-train {dataset_name} row must be inner validation "
                "data exactly once."
            )
        if set(validation_question_counts.values()) != {1}:
            raise ValueError(
                f"Every outer-train {dataset_name} question group must occur "
                "in exactly one inner validation fold."
            )

    return True


def print_dataset_counts(rows, examples):
    print("\nDataset row counts")
    print("------------------")

    counts = {}

    for row in rows:
        name = row["dataset"]
        counts[name] = counts.get(name, 0) + 1

    for name, count in sorted(counts.items()):
        print(f"{name}: rows={count}")

    print(f"Total rows: {len(rows)}")
    print(f"Total individual examples: {len(examples)}")


def validate_required_splits(splits, dataset_names=None):
    if dataset_names is None:
        dataset_names = splits["dataset_names"]

    dataset_names = [
        normalize_dataset_key(name)
        for name in dataset_names
    ]

    missing = [
        name
        for name in dataset_names
        if len(splits["datasets"].get(name, {}).get("rows", [])) == 0
    ]

    if missing:
        available = sorted({
            normalize_dataset_key(row["dataset"])
            for row in splits["rows"]
        })
        raise ValueError(
            "Missing required dataset rows for: "
            f"{', '.join(missing)}. "
            f"Available dataset keys in CSV: {', '.join(available)}."
        )


def format_question_claim(
    question: str,
    claim: str,
    short_answer: str = "",
    use_short_answer: bool = False,
) -> str:
    if use_short_answer and short_answer:
        return (
            f"Question: {question}\n"
            f"Known answer: {short_answer}\n"
            f"Claim: {claim}"
        )

    return f"Question: {question}\nClaim: {claim}"


def _split_prompt_and_answer_from_text(text: str):
    """Splits formatted text into:"""

    markers = ["\nClaim:", "\nAnswer:"]

    for marker in markers:
        idx = text.rfind(marker)

        if idx >= 0:
            marker_end = idx + len(marker)
            prompt = text[:marker_end]
            answer = text[marker_end:]
            return prompt, answer

    return "", text


def encode_text_with_answer_mask(
    tokenizer,
    text: str,
    max_length: int,
):
    """Tokenise a formatted question-claim text and build answer_mask."""

    prompt, answer = _split_prompt_and_answer_from_text(text)

    if answer and not answer.startswith(" "):
        answer = " " + answer

    prompt_ids = tokenizer(
        prompt,
        add_special_tokens=True,
        truncation=False,
    )["input_ids"]

    answer_ids = tokenizer(
        answer,
        add_special_tokens=False,
        truncation=False,
    )["input_ids"]

    if len(answer_ids) == 0:
        full = tokenizer(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=max_length,
        )["input_ids"]

        input_ids = full[:max_length]
        attention_mask = [1] * len(input_ids)
        answer_mask = [1] * len(input_ids)

    else:
        if len(answer_ids) >= max_length:
            answer_ids = answer_ids[:max_length]
            prompt_ids = []
        else:
            max_prompt_len = max_length - len(answer_ids)
            prompt_ids = prompt_ids[-max_prompt_len:]

        input_ids = prompt_ids + answer_ids
        attention_mask = [1] * len(input_ids)
        answer_mask = [0] * len(prompt_ids) + [1] * len(answer_ids)

    pad_len = max_length - len(input_ids)

    if pad_len > 0:
        input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
        attention_mask = attention_mask + [0] * pad_len
        answer_mask = answer_mask + [0] * pad_len

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "answer_mask": torch.tensor(answer_mask, dtype=torch.long),
    }


def tokenize_texts_with_answer_masks(
    tokenizer,
    texts: List[str],
    max_length: int,
):
    encoded = [
        encode_text_with_answer_mask(
            tokenizer=tokenizer,
            text=text,
            max_length=max_length,
        )
        for text in texts
    ]

    return {
        "input_ids": torch.stack([x["input_ids"] for x in encoded], dim=0),
        "attention_mask": torch.stack([x["attention_mask"] for x in encoded], dim=0),
        "answer_mask": torch.stack([x["answer_mask"] for x in encoded], dim=0),
    }


def feature_cache_key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _feature_cache_metadata(
    texts,
    base_model,
    energy_model,
    max_length,
    dtype,
):
    text_digest = hashlib.sha1(
        "\n".join(sorted(feature_cache_key(text) for text in texts)).encode("utf-8")
    ).hexdigest()

    base_config = getattr(base_model, "config", None)

    return {
        "format": "raw_selected_answer_layer_reprs_v1",
        "model_name": str(getattr(base_config, "_name_or_path", "")),
        "max_length": int(max_length),
        "hidden_size": int(getattr(energy_model, "hidden_size")),
        "num_selected_layers": int(getattr(energy_model, "num_selected_layers")),
        "selected_layer_indices": [
            int(i)
            for i in getattr(energy_model, "selected_layer_indices")
        ],
        "dtype": str(dtype).replace("torch.", ""),
        "text_digest": text_digest,
        "num_texts": int(len(set(texts))),
    }


def _metadata_matches(a, b):
    return (
        a.get("format") == b.get("format")
        and a.get("model_name") == b.get("model_name")
        and int(a.get("max_length", -1)) == int(b.get("max_length", -2))
        and int(a.get("hidden_size", -1)) == int(b.get("hidden_size", -2))
        and int(a.get("num_selected_layers", -1)) == int(b.get("num_selected_layers", -2))
        and list(a.get("selected_layer_indices", [])) == list(b.get("selected_layer_indices", [None]))
        and a.get("dtype") == b.get("dtype")
        and a.get("text_digest") == b.get("text_digest")
    )


@torch.no_grad()
def build_or_load_frozen_feature_cache(
    texts,
    tokenizer,
    base_model,
    energy_model,
    device,
    max_length=128,
    batch_size=16,
    cache_path=None,
    dtype=torch.float16,
):
    """Cache exactly the frozen representation used by the trainable EBM:"""

    unique_texts = list(dict.fromkeys(texts))

    if not unique_texts:
        return {}

    metadata = _feature_cache_metadata(
        texts=unique_texts,
        base_model=base_model,
        energy_model=energy_model,
        max_length=max_length,
        dtype=dtype,
    )

    if cache_path:
        cache_path = os.path.abspath(cache_path)
        if os.path.exists(cache_path):
            cached = torch.load(cache_path, map_location="cpu")
            cached_metadata = cached.get("metadata", {})
            if _metadata_matches(cached_metadata, metadata):
                print(
                    "Using cached frozen LLM answer-layer features: "
                    f"{cache_path}",
                    flush=True,
                )
                keys = cached["keys"]
                features = cached["features"]
                return {
                    key: features[i]
                    for i, key in enumerate(keys)
                }

            print(
                "Ignoring stale frozen feature cache because metadata changed: "
                f"{cache_path}",
                flush=True,
            )

    print(
        "Precomputing frozen LLM answer-layer features "
        f"for {len(unique_texts)} unique question-answer texts...",
        flush=True,
    )

    base_model.eval()
    energy_model.eval()
    device = torch.device(device)

    keys = []
    feature_batches = []
    total_batches = (len(unique_texts) + batch_size - 1) // batch_size

    for start in range(0, len(unique_texts), batch_size):
        end = min(start + batch_size, len(unique_texts))
        batch_texts = unique_texts[start:end]
        batch_id = start // batch_size + 1

        if batch_id == 1 or batch_id == total_batches or batch_id % 25 == 0:
            print(
                f"  frozen feature batch {batch_id}/{total_batches}",
                flush=True,
            )

        encoded = tokenize_texts_with_answer_masks(
            tokenizer=tokenizer,
            texts=batch_texts,
            max_length=max_length,
        )

        input_ids = encoded["input_ids"].to(device, non_blocking=True)
        attention_mask = encoded["attention_mask"].to(device, non_blocking=True)
        answer_mask = encoded["answer_mask"].to(device, non_blocking=True)

        outputs = base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )

        raw_layer_reprs = energy_model.get_raw_layer_reprs(
            hidden_states=outputs.hidden_states,
            attention_mask=attention_mask,
            answer_mask=answer_mask,
        )

        raw_layer_reprs = torch.nan_to_num(
            raw_layer_reprs.detach().cpu().to(dtype=dtype),
            nan=0.0,
            posinf=1e6,
            neginf=-1e6,
        )

        feature_batches.append(raw_layer_reprs)
        keys.extend(feature_cache_key(text) for text in batch_texts)

    features = torch.cat(feature_batches, dim=0).contiguous()

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        tmp_path = f"{cache_path}.tmp"
        torch.save(
            {
                "metadata": metadata,
                "keys": keys,
                "features": features,
            },
            tmp_path,
        )
        os.replace(tmp_path, cache_path)
        print(f"Saved frozen feature cache to: {cache_path}", flush=True)

    return {
        key: features[i]
        for i, key in enumerate(keys)
    }


class ClaimDataset(Dataset):
    """Individual examples for evaluation."""

    def __init__(
        self,
        examples,
        use_short_answer=False,
        feature_cache=None,
    ):
        self.examples = examples
        self.use_short_answer = use_short_answer
        self.feature_cache = feature_cache

    def __len__(self):
        return len(self.examples)

    def format_text(self, question, claim, short_answer=None):
        return format_question_claim(
            question=question,
            claim=claim,
            short_answer=short_answer,
            use_short_answer=self.use_short_answer,
        )

    def __getitem__(self, idx):
        ex = self.examples[idx]

        text = self.format_text(
            ex["question"],
            ex["claim"],
            ex.get("short_answer", ""),
        )

        item = {
            "text": text,
            "dataset": ex["dataset"],
            "question": ex["question"],
            "short_answer": ex.get("short_answer", ""),
            "claim": ex["claim"],
            "label": float(ex["label"]),
            "label_name": ex["label_name"],
        }

        if self.feature_cache is not None:
            item["raw_layer_reprs"] = self.feature_cache[feature_cache_key(text)]

        return item


class PairClaimDataset(Dataset):
    """Paired truthful/hallucinated examples for training."""

    def __init__(
        self,
        rows,
        use_short_answer=False,
        k_neighbours=5,
        neighbour_backend="llm_hidden",
        llm_tokenizer=None,
        llm_base_model=None,
        llm_device=None,
        llm_hidden_batch_size=16,
        llm_hidden_max_length=128,
        feature_cache=None,
    ):
        self.rows = rows
        self.use_short_answer = use_short_answer
        self.k_neighbours = k_neighbours
        self.neighbour_backend = neighbour_backend
        self.llm_tokenizer = llm_tokenizer
        self.llm_base_model = llm_base_model
        self.llm_device = llm_device
        self.llm_hidden_batch_size = llm_hidden_batch_size
        self.llm_hidden_max_length = llm_hidden_max_length
        self.feature_cache = feature_cache

        self.neighbour_indices = self._build_neighbour_indices(
            k=k_neighbours,
            backend=neighbour_backend,
        )

    def __len__(self):
        return len(self.rows)

    def format_text(self, question, claim, short_answer=None):
        return format_question_claim(
            question=question,
            claim=claim,
            short_answer=short_answer,
            use_short_answer=self.use_short_answer,
        )

    def _build_neighbour_indices(
        self,
        k,
        backend,
    ):
        """Build K nearest question neighbours among training questions."""
        backend = str(backend or "none").strip().lower()

        if backend not in {"none", "llm_hidden"}:
            raise ValueError(
                "Unknown neighbour backend: "
                f"{backend}. Use one of: none, llm_hidden."
            )

        if backend == "none" or k <= 0 or len(self.rows) <= 1:
            return [[] for _ in self.rows]

        unique_questions = []
        representative_row_indices = []
        question_group_by_key = {}
        row_group_indices = []

        for row_idx, row in enumerate(self.rows):
            key = question_identity(row["question"])
            group_idx = question_group_by_key.get(key)

            if group_idx is None:
                group_idx = len(unique_questions)
                question_group_by_key[key] = group_idx
                unique_questions.append(row["question"])
                representative_row_indices.append(row_idx)

            row_group_indices.append(group_idx)

        if len(unique_questions) <= 1:
            return [[] for _ in self.rows]

        question_digest = hashlib.sha1(
            "\n".join(question_identity(q) for q in unique_questions).encode("utf-8")
        ).hexdigest()
        cache_key = (
            "unique_question_knn_v1",
            backend,
            int(k),
            str(getattr(getattr(self.llm_base_model, "config", None), "_name_or_path", "")),
            int(self.llm_hidden_max_length),
            int(self.llm_hidden_batch_size),
            question_digest,
        )

        if cache_key in _NEIGHBOUR_INDEX_CACHE:
            print(
                f"Using cached {backend} question neighbours "
                f"(k={k}, unique_questions={len(unique_questions)}, "
                f"rows={len(self.rows)})."
            )
            return _NEIGHBOUR_INDEX_CACHE[cache_key]

        if backend == "llm_hidden":
            try:
                model_name = str(
                    getattr(
                        getattr(self.llm_base_model, "config", None),
                        "_name_or_path",
                        "frozen_base_lm",
                    )
                )

                print(
                    "Building frozen-LLM hidden-state question neighbours "
                    f"with {model_name} "
                    f"(k={k}, unique_questions={len(unique_questions)}, "
                    f"rows={len(self.rows)})..."
                )

                embeddings = build_llm_hidden_question_embeddings(
                    questions=unique_questions,
                    tokenizer=self.llm_tokenizer,
                    base_model=self.llm_base_model,
                    device=self.llm_device,
                    batch_size=self.llm_hidden_batch_size,
                    max_length=self.llm_hidden_max_length,
                )

                n_neighbours = min(k + 1, len(unique_questions))
                index = NearestNeighbors(
                    n_neighbors=n_neighbours,
                    metric="cosine",
                    algorithm="brute",
                )
                index.fit(embeddings)
                _, nearest_indices = index.kneighbors(embeddings)

            except Exception as e:
                raise RuntimeError(
                    "Could not build LLM-hidden neighbours. "
                    "Check that tokenizer/base_model/device were passed to "
                    "build_dataloaders."
                ) from e

        neighbours_by_group = []

        for group_idx, neighbour_group_indices in enumerate(nearest_indices):
            neighbour_group_indices = [
                int(j)
                for j in neighbour_group_indices
                if int(j) != group_idx
            ]
            neighbours_by_group.append([
                representative_row_indices[j]
                for j in neighbour_group_indices[:k]
            ])

        neighbours = [
            list(neighbours_by_group[group_idx])
            for group_idx in row_group_indices
        ]

        _NEIGHBOUR_INDEX_CACHE[cache_key] = neighbours
        return neighbours

    def __getitem__(self, idx):
        row = self.rows[idx]

        question = row["question"]
        short_answer = row.get("short_answer", "")

        pos_text = self.format_text(
            question=question,
            claim=row["positive"],
            short_answer=short_answer,
        )

        neg_text = self.format_text(
            question=question,
            claim=row["negative"],
            short_answer=short_answer,
        )

        neigh_pos_texts = []
        neigh_neg_texts = []

        for j in self.neighbour_indices[idx]:
            nrow = self.rows[j]

            neigh_pos_texts.append(
                self.format_text(
                    question=nrow["question"],
                    claim=nrow["positive"],
                    short_answer=nrow.get("short_answer", ""),
                )
            )

            neigh_neg_texts.append(
                self.format_text(
                    question=nrow["question"],
                    claim=nrow["negative"],
                    short_answer=nrow.get("short_answer", ""),
                )
            )

        item = {
            "dataset": row["dataset"],
            "question": question,
            "short_answer": short_answer,

            "positive": row["positive"],
            "negative": row["negative"],

            "pos_text": pos_text,
            "neg_text": neg_text,

            "neigh_pos_texts": neigh_pos_texts,
            "neigh_neg_texts": neigh_neg_texts,
        }

        if self.feature_cache is not None:
            item["pos_raw_layer_reprs"] = self.feature_cache[feature_cache_key(pos_text)]
            item["neg_raw_layer_reprs"] = self.feature_cache[feature_cache_key(neg_text)]
            item["neigh_pos_raw_layer_reprs"] = [
                self.feature_cache[feature_cache_key(text)]
                for text in neigh_pos_texts
            ]
            item["neigh_neg_raw_layer_reprs"] = [
                self.feature_cache[feature_cache_key(text)]
                for text in neigh_neg_texts
            ]

        return item


def build_question_neighbour_indices(
    rows,
    tokenizer,
    base_model,
    device,
    k=5,
    batch_size=16,
    max_length=128,
):
    """Build question-neighbour indices using the same implementation as training."""
    dataset = PairClaimDataset(
        rows=rows,
        use_short_answer=False,
        k_neighbours=k,
        neighbour_backend="llm_hidden",
        llm_tokenizer=tokenizer,
        llm_base_model=base_model,
        llm_device=device,
        llm_hidden_batch_size=batch_size,
        llm_hidden_max_length=max_length,
    )
    return dataset.neighbour_indices


def make_pair_collate_fn(tokenizer, max_length):
    def collate_fn(batch):
        pos_texts = [x["pos_text"] for x in batch]
        neg_texts = [x["neg_text"] for x in batch]

        neigh_pos_texts = []
        neigh_neg_texts = []
        k_list = []

        for x in batch:
            k = len(x["neigh_pos_texts"])
            k_list.append(k)
            neigh_pos_texts.extend(x["neigh_pos_texts"])
            neigh_neg_texts.extend(x["neigh_neg_texts"])

        pos_enc = tokenize_texts_with_answer_masks(
            tokenizer,
            pos_texts,
            max_length,
        )

        neg_enc = tokenize_texts_with_answer_masks(
            tokenizer,
            neg_texts,
            max_length,
        )

        out = {
            "pos_input_ids": pos_enc["input_ids"],
            "pos_attention_mask": pos_enc["attention_mask"],
            "pos_answer_mask": pos_enc["answer_mask"],

            "neg_input_ids": neg_enc["input_ids"],
            "neg_attention_mask": neg_enc["attention_mask"],
            "neg_answer_mask": neg_enc["answer_mask"],

            "k_list": torch.tensor(k_list, dtype=torch.long),
            "has_neighbours": sum(k_list) > 0,
            "raw_batch": batch,
        }

        if "pos_raw_layer_reprs" in batch[0]:
            out["pos_raw_layer_reprs"] = torch.stack(
                [x["pos_raw_layer_reprs"] for x in batch],
                dim=0,
            )
            out["neg_raw_layer_reprs"] = torch.stack(
                [x["neg_raw_layer_reprs"] for x in batch],
                dim=0,
            )

        if sum(k_list) > 0:
            if "pos_raw_layer_reprs" in batch[0]:
                out["neigh_pos_raw_layer_reprs"] = torch.stack(
                    [
                        raw
                        for x in batch
                        for raw in x["neigh_pos_raw_layer_reprs"]
                    ],
                    dim=0,
                )
                out["neigh_neg_raw_layer_reprs"] = torch.stack(
                    [
                        raw
                        for x in batch
                        for raw in x["neigh_neg_raw_layer_reprs"]
                    ],
                    dim=0,
                )
            else:
                neigh_pos_enc = tokenize_texts_with_answer_masks(
                    tokenizer,
                    neigh_pos_texts,
                    max_length,
                )
                neigh_neg_enc = tokenize_texts_with_answer_masks(
                    tokenizer,
                    neigh_neg_texts,
                    max_length,
                )
                out.update(
                    {
                        "neigh_pos_input_ids": neigh_pos_enc["input_ids"],
                        "neigh_pos_attention_mask": neigh_pos_enc["attention_mask"],
                        "neigh_pos_answer_mask": neigh_pos_enc["answer_mask"],
                        "neigh_neg_input_ids": neigh_neg_enc["input_ids"],
                        "neigh_neg_attention_mask": neigh_neg_enc["attention_mask"],
                        "neigh_neg_answer_mask": neigh_neg_enc["answer_mask"],
                    }
                )

        return out

    return collate_fn


def make_claim_collate_fn(tokenizer, max_length):
    def collate_fn(batch):
        texts = [x["text"] for x in batch]
        labels = torch.tensor([x["label"] for x in batch], dtype=torch.float)

        if "raw_layer_reprs" in batch[0]:
            return {
                "labels": labels,
                "raw_batch": batch,
                "raw_layer_reprs": torch.stack(
                    [x["raw_layer_reprs"] for x in batch],
                    dim=0,
                ),
            }

        enc = tokenize_texts_with_answer_masks(
            tokenizer,
            texts,
            max_length,
        )
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "answer_mask": enc["answer_mask"],
            "labels": labels,
            "raw_batch": batch,
        }

    return collate_fn


def build_dataloaders(
    splits,
    tokenizer,
    train_dataset,
    eval_datasets=None,
    max_length=128,
    batch_size=16,
    use_short_answer=False,
    num_workers=0,
    k_neighbours=5,
    neighbour_backend="llm_hidden",
    neighbour_llm_base_model=None,
    neighbour_llm_device=None,
    neighbour_llm_batch_size=16,
    cache_frozen_features=False,
    feature_cache_base_model=None,
    feature_cache_energy_model=None,
    feature_cache_device=None,
    feature_cache_path=None,
    feature_cache_batch_size=16,
):
    train_dataset = normalize_dataset_key(train_dataset)

    if eval_datasets is None:
        eval_datasets = [
            name
            for name in splits["dataset_names"]
            if name != train_dataset
        ]
    else:
        eval_datasets = [
            normalize_dataset_key(name)
            for name in eval_datasets
        ]

    if train_dataset not in splits["datasets"]:
        raise KeyError(f"Unknown train dataset: {train_dataset}")

    train_rows = splits["datasets"][train_dataset]["rows"]
    train_rows = splits["datasets"][train_dataset].get("train_rows", train_rows)

    if len(train_rows) == 0:
        raise ValueError(f"No rows available for train dataset: {train_dataset}")

    train_eval_name = f"{train_dataset}_train"
    is_nested_cv = "test_examples" in splits["datasets"][train_dataset]
    validation_eval_name = (
        f"{train_dataset}_inner_val"
        if is_nested_cv
        else f"{train_dataset}_val"
    )

    train_examples = splits["datasets"][train_dataset].get(
        "train_examples",
        rows_to_claim_examples(train_rows),
    )
    validation_examples = splits["datasets"][train_dataset].get(
        "validation_examples",
        [],
    )

    if len(validation_examples) == 0:
        validation_examples = train_examples

    eval_example_specs = [
        (train_eval_name, train_examples),
        (validation_eval_name, validation_examples),
    ]
    final_eval_names = [train_eval_name]

    if is_nested_cv:
        test_eval_name = f"{train_dataset}_test"
        test_examples = splits["datasets"][train_dataset]["test_examples"]
        if len(test_examples) == 0:
            raise ValueError(
                f"No outer test examples available for train dataset: "
                f"{train_dataset}"
            )
        eval_example_specs.append((test_eval_name, test_examples))
        final_eval_names.append(test_eval_name)
    else:
        final_eval_names.append(validation_eval_name)

    for dataset_name in eval_datasets:
        if dataset_name not in splits["datasets"]:
            raise KeyError(f"Unknown eval dataset: {dataset_name}")

        dataset_split = splits["datasets"][dataset_name]
        eval_examples = (
            dataset_split["test_examples"]
            if is_nested_cv
            else dataset_split["examples"]
        )

        if len(eval_examples) == 0:
            raise ValueError(f"No examples available for eval dataset: {dataset_name}")

        eval_example_specs.append((dataset_name, eval_examples))
        final_eval_names.append(dataset_name)

    feature_cache = None

    if cache_frozen_features:
        if feature_cache_base_model is None or feature_cache_energy_model is None:
            raise ValueError(
                "cache_frozen_features=True requires feature_cache_base_model "
                "and feature_cache_energy_model."
            )

        cache_texts = []
        for ex in splits["examples"]:
            cache_texts.append(
                format_question_claim(
                    ex["question"],
                    ex["claim"],
                    short_answer=ex.get("short_answer", ""),
                    use_short_answer=use_short_answer,
                )
            )

        feature_cache = build_or_load_frozen_feature_cache(
            texts=cache_texts,
            tokenizer=tokenizer,
            base_model=feature_cache_base_model,
            energy_model=feature_cache_energy_model,
            device=feature_cache_device or neighbour_llm_device or "cpu",
            max_length=max_length,
            batch_size=feature_cache_batch_size,
            cache_path=feature_cache_path,
            dtype=torch.float16,
        )

    pair_collate_fn = make_pair_collate_fn(tokenizer, max_length)
    claim_collate_fn = make_claim_collate_fn(tokenizer, max_length)

    train_ds = PairClaimDataset(
        train_rows,
        use_short_answer=use_short_answer,
        k_neighbours=k_neighbours,
        neighbour_backend=neighbour_backend,
        llm_tokenizer=tokenizer,
        llm_base_model=neighbour_llm_base_model,
        llm_device=neighbour_llm_device,
        llm_hidden_batch_size=neighbour_llm_batch_size,
        llm_hidden_max_length=max_length,
        feature_cache=feature_cache,
    )

    eval_loaders = {}

    for eval_name, eval_examples in eval_example_specs:
        eval_ds = ClaimDataset(
            eval_examples,
            use_short_answer=use_short_answer,
            feature_cache=feature_cache,
        )

        eval_loaders[eval_name] = DataLoader(
            eval_ds,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=claim_collate_fn,
            num_workers=num_workers,
        )

    return {
        "train_dataset": train_dataset,
        "eval_datasets": final_eval_names,
        "monitor_datasets": [validation_eval_name],
        "train": DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=pair_collate_fn,
            num_workers=num_workers,
        ),
        "eval": eval_loaders,
    }
