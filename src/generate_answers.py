import argparse
import ast
import os
import re
import random
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
TRIVIAQA_DATASET_ID = "mandarjoshi/trivia_qa"
TRUTHFULQA_DATASET_ID = "domenicrosati/TruthfulQA"

DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

MAX_SAMPLES_PER_DATASET = 10000
MAX_NEW_TOKENS_FACTUAL = 48
MAX_NEW_TOKENS_HALLUCINATED = 64

OUTPUT_PATH = "processed_qa_10000hallucination_dataset.csv"
CHECKPOINT_PATH = "processed_qa_hallucination_dataset_checkpoint.csv"

SEED = 42


def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


tokenizer = None
model = None


def load_generator():
    global tokenizer, model
    set_seed(SEED)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision=MODEL_REVISION, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.float16 if DEVICE in ["cuda", "mps"] else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, revision=MODEL_REVISION, torch_dtype=dtype, trust_remote_code=True,
    ).to(DEVICE)
    model.eval()


def load_dataset(*args, **kwargs):
    from datasets import load_dataset as load
    return load(*args, **kwargs)


def clean_text(text):
    if text is None:
        return None

    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)

    if text == "":
        return None

    return text


def ensure_sentence(text):
    text = clean_text(text)

    if text is None:
        return None

    if text[-1] not in [".", "!", "?"]:
        text += "."

    return text


def remove_prompt_echo(text):
    """Removes prompt echo if the model repeats the instruction."""
    text = clean_text(text)

    if text is None:
        return None

    markers = [
        "Factual sentence:",
        "Full sentence answer:",
        "Hallucinated answer:",
        "Answer:",
    ]

    for marker in markers:
        if marker in text:
            text = text.split(marker)[-1].strip()

    return clean_text(text)


def keep_first_sentence(text):
    """Keep only the first sentence to avoid explanations, notes, citations,"""
    text = clean_text(text)

    if text is None:
        return None

    bad_starts = [
        "To determine",
        "The correct answer",
        "Therefore",
        "Note:",
        "This hallucinated answer",
        "Explanation:",
        "Here is",
        "Sure",
    ]

    for bad in bad_starts:
        if text.lower().startswith(bad.lower()):
            return None

    stop_markers = [
        " Note:",
        " Explanation:",
        " Source:",
        " [source]",
        " [cite]",
        " Hallucinated answer:",
        " Factual sentence:",
        "\n",
    ]

    for marker in stop_markers:
        if marker in text:
            text = text.split(marker)[0].strip()

    match = re.search(r"(.+?[.!?])(\s|$)", text)

    if match:
        return clean_text(match.group(1))

    return ensure_sentence(text)


def clean_generated_answer(text):
    text = remove_prompt_echo(text)
    text = keep_first_sentence(text)
    text = ensure_sentence(text)
    return text


def is_valid_answer(text):
    text = clean_text(text)

    if text is None:
        return False

    lowered = text.lower()

    bad_phrases = [
        "you are converting",
        "you are generating",
        "requirements:",
        "question:",
        "correct answer:",
        "factual sentence:",
        "hallucinated answer:",
        "do not explain",
        "training data",
        "dataset",
        "note:",
        "source",
        "cite",
        "to determine",
        "therefore",
        "explanation",
    ]

    if any(p in lowered for p in bad_phrases):
        return False

    bad_exact = {
        "",
        "none",
        "unknown",
        "i don't know",
        "i do not know",
        "cannot answer",
        "not enough information",
    }

    if lowered in bad_exact:
        return False

    word_count = len(text.split())

    if word_count < 3:
        return False

    if word_count > 45:
        return False

    return True


def run_llm(
    prompt,
    max_new_tokens=64,
    do_sample=False,
    temperature=0.8,
    top_p=0.9,
):
    messages = [
        {
            "role": "system",
            "content": (
                "You are a careful data generation assistant. "
                "Return only one requested sentence. "
                "Do not repeat the prompt. "
                "Do not explain anything."
            ),
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    chat_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        chat_text,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
    ).to(DEVICE)

    input_length = inputs["input_ids"].shape[1]

    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    if do_sample:
        generation_kwargs.update({
            "do_sample": True,
            "temperature": temperature,
            "top_p": top_p,
        })
    else:
        generation_kwargs.update({
            "do_sample": False,
        })

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            **generation_kwargs,
        )

    generated_ids = output_ids[0, input_length:]

    raw_output = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    )

    return clean_generated_answer(raw_output)


def build_full_sentence_prompt(question, short_answer):
    return f"""
You are converting short-answer QA data into full factual claim sentences for hallucination detection research.

Given a question and the correct short answer, write one complete factual sentence that directly answers the question.

Requirements:
1. The sentence must be factually consistent with the given correct answer.
2. The sentence must be fluent and natural.
3. The sentence must be a complete sentence, not just a noun phrase.
4. The sentence should preserve the meaning of the original question.
5. Do not add extra facts that are not required by the question.
6. Do not explain anything.
7. Do not mention that this is a dataset or training example.
8. Return only the factual sentence.

Question: {question}
Correct answer: {short_answer}

Factual sentence:
""".strip()


def convert_answer_to_full_sentence(question, short_answer):
    prompt = build_full_sentence_prompt(
        question=question,
        short_answer=short_answer,
    )

    return run_llm(
        prompt=prompt,
        max_new_tokens=MAX_NEW_TOKENS_FACTUAL,
        do_sample=False,
    )


def build_hallucinated_answer_prompt(question, correct_answer):
    return f"""
You are generating training data for hallucination detection.

Given a question and its correct answer, write one hallucinated answer.

Requirements:
1. The hallucinated answer must be factually incorrect.
2. It must be fluent and plausible.
3. It must be a complete sentence.
4. It should have similar length and style to the truthful answer.
5. It should not say "I don't know" or express uncertainty.
6. It should not be obviously absurd.
7. It should change only the key factual entity or fact, while keeping the rest of the sentence structure similar.
8. Do not explain why it is wrong.
9. Return only the hallucinated answer.

Question: {question}
Correct answer: {correct_answer}

Hallucinated answer:
""".strip()


def generate_negative_answer(question, correct_answer):
    prompt = build_hallucinated_answer_prompt(
        question=question,
        correct_answer=correct_answer,
    )

    return run_llm(
        prompt=prompt,
        max_new_tokens=MAX_NEW_TOKENS_HALLUCINATED,
        do_sample=True,
        temperature=0.8,
        top_p=0.9,
    )


def safe_select_dataset(ds, max_samples):
    n = min(max_samples, len(ds))
    return ds.select(range(n))


def make_example(dataset_name, question, positive, negatives, short_answer=None):
    return {
        "dataset": dataset_name,
        "question": clean_text(question),
        "short_answer": clean_text(short_answer) if short_answer is not None else None,
        "positive": clean_text(positive),
        "negatives": negatives,
    }


def save_checkpoint(examples):
    if len(examples) == 0:
        return

    df = pd.DataFrame(examples)
    df.to_csv(CHECKPOINT_PATH, index=False)


def parse_checkpoint_negatives(value):
    if isinstance(value, list):
        return [
            clean_text(item)
            for item in value
            if clean_text(item) is not None
        ]

    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []

    try:
        parsed = ast.literal_eval(str(value))
    except (SyntaxError, ValueError):
        parsed = [value]

    if not isinstance(parsed, list):
        parsed = [parsed]

    return [
        clean_text(item)
        for item in parsed
        if clean_text(item) is not None
    ]


def load_checkpoint_examples():
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"No checkpoint found at {CHECKPOINT_PATH}; starting from scratch.")
        return []

    df = pd.read_csv(CHECKPOINT_PATH)
    examples = []

    for row in df.to_dict(orient="records"):
        dataset_name = clean_text(row.get("dataset"))
        question = clean_text(row.get("question"))
        positive = clean_text(row.get("positive"))
        negatives = parse_checkpoint_negatives(row.get("negatives"))
        short_answer = row.get("short_answer")

        if isinstance(short_answer, float) and pd.isna(short_answer):
            short_answer = None
        else:
            short_answer = clean_text(short_answer)

        if (
            dataset_name is None
            or question is None
            or positive is None
            or len(negatives) == 0
        ):
            continue

        examples.append(
            {
                "dataset": dataset_name,
                "question": question,
                "short_answer": short_answer,
                "positive": positive,
                "negatives": negatives,
            }
        )

    print(
        f"Loaded {len(examples)} grouped examples from checkpoint: "
        f"{CHECKPOINT_PATH}"
    )
    return examples


def checkpoint_with_prefix(prefix_examples, current_examples):
    save_checkpoint([
        *(prefix_examples or []),
        *current_examples,
    ])


def existing_questions(examples, dataset_names):
    dataset_names = set(dataset_names)
    return {
        ex["question"]
        for ex in examples
        if ex.get("dataset") in dataset_names
    }


def process_hotpotqa(
    split="train",
    max_samples=1000,
    checkpoint_prefix=None,
    skip_questions=None,
):
    print("\nLoading HotpotQA...")

    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split=split)
    ds = safe_select_dataset(ds, max_samples)

    examples = []
    skip_questions = set(skip_questions or [])

    for ex in tqdm(ds, desc="Processing HotpotQA"):
        question = clean_text(ex.get("question"))
        short_answer = clean_text(ex.get("answer"))

        if question is None or short_answer is None:
            continue

        if question in skip_questions:
            continue

        positive = convert_answer_to_full_sentence(
            question=question,
            short_answer=short_answer,
        )

        if not is_valid_answer(positive):
            print("Bad positive:", positive)
            continue

        negative = generate_negative_answer(
            question=question,
            correct_answer=positive,
        )

        if not is_valid_answer(negative):
            print("Bad negative:", negative)
            continue

        if positive.lower() == negative.lower():
            print("Skipped identical positive/negative:", positive)
            continue

        examples.append(
            make_example(
                dataset_name="hotpotqa",
                question=question,
                short_answer=short_answer,
                positive=positive,
                negatives=[negative],
            )
        )

        if len(examples) % 100 == 0:
            checkpoint_with_prefix(checkpoint_prefix, examples)

    return examples


def extract_triviaqa_answer(ex):
    answer = ex.get("answer")

    if isinstance(answer, dict):
        if "value" in answer:
            return clean_text(answer["value"])

        if "normalized_value" in answer:
            return clean_text(answer["normalized_value"])

        if "aliases" in answer and len(answer["aliases"]) > 0:
            return clean_text(answer["aliases"][0])

    if isinstance(answer, str):
        return clean_text(answer)

    return None


def process_triviaqa_wiki(
    split="train",
    max_samples=1000,
    checkpoint_prefix=None,
    skip_questions=None,
):
    print("\nLoading TriviaQA Wiki...")

    ds = load_dataset(
        TRIVIAQA_DATASET_ID,
        "rc.wikipedia",
        split=split,
    )
    ds = safe_select_dataset(ds, max_samples)

    examples = []
    skip_questions = set(skip_questions or [])

    for ex in tqdm(ds, desc="Processing TriviaQA Wiki"):
        question = clean_text(ex.get("question"))
        short_answer = extract_triviaqa_answer(ex)

        if question is None or short_answer is None:
            continue

        if question in skip_questions:
            continue

        positive = convert_answer_to_full_sentence(
            question=question,
            short_answer=short_answer,
        )

        if not is_valid_answer(positive):
            print("Bad positive:", positive)
            continue

        negative = generate_negative_answer(
            question=question,
            correct_answer=positive,
        )

        if not is_valid_answer(negative):
            print("Bad negative:", negative)
            continue

        if positive.lower() == negative.lower():
            print("Skipped identical positive/negative:", positive)
            continue

        examples.append(
            make_example(
                dataset_name="triviaqa_wiki",
                question=question,
                short_answer=short_answer,
                positive=positive,
                negatives=[negative],
            )
        )

        if len(examples) % 100 == 0:
            checkpoint_with_prefix(checkpoint_prefix, examples)

    return examples


def process_truthfulqa(
    split="train",
    max_samples=1000,
    checkpoint_prefix=None,
    skip_questions=None,
):
    print("\nLoading TruthfulQA...")

    ds = load_dataset(TRUTHFULQA_DATASET_ID, split=split)
    ds = safe_select_dataset(ds, max_samples)

    examples = []
    skip_questions = set(skip_questions or [])

    for ex in tqdm(ds, desc="Processing TruthfulQA"):
        question = ex.get("Question")
        positive = ex.get("Best Answer")
        incorrect_str = ex.get("Incorrect Answers", "")

        if not question or not positive or not incorrect_str:
            continue

        question = question.strip()

        if question in skip_questions:
            continue

        negatives = [
            x.strip()
            for x in incorrect_str.split(";")
            if x.strip()
        ]

        if negatives:
            examples.append({
                "dataset": "truthfulqa",
                "question": question,
                "short_answer": positive.strip(),
                "positive": positive.strip(),
                "negatives": negatives,
            })

        if len(examples) % 100 == 0:
            checkpoint_with_prefix(checkpoint_prefix, examples)

    return examples


def extract_squad_answer(ex):
    answers = ex.get("answers")

    if not isinstance(answers, dict):
        return None

    answer_texts = answers.get("text", [])

    if isinstance(answer_texts, str):
        answer_texts = [answer_texts]

    for answer_text in answer_texts:
        answer = clean_text(answer_text)

        if answer is not None:
            return answer

    return None


def process_squadqa(
    split="train",
    max_samples=1000,
    checkpoint_prefix=None,
    skip_questions=None,
):
    print("\nLoading SQuAD...")

    ds = load_dataset("rajpurkar/squad", split=split)
    ds = safe_select_dataset(ds, max_samples)

    examples = []
    skip_questions = set(skip_questions or [])

    for ex in tqdm(ds, desc="Processing SQuAD"):
        question = clean_text(ex.get("question"))
        short_answer = extract_squad_answer(ex)

        if question is None or short_answer is None:
            continue

        if question in skip_questions:
            continue

        positive = convert_answer_to_full_sentence(
            question=question,
            short_answer=short_answer,
        )

        if not is_valid_answer(positive):
            print("Bad positive:", positive)
            continue

        negative = generate_negative_answer(
            question=question,
            correct_answer=positive,
        )

        if not is_valid_answer(negative):
            print("Bad negative:", negative)
            continue

        if positive.lower() == negative.lower():
            print("Skipped identical positive/negative:", positive)
            continue

        examples.append(
            make_example(
                dataset_name="squadqa",
                question=question,
                short_answer=short_answer,
                positive=positive,
                negatives=[negative],
            )
        )

        if len(examples) % 100 == 0:
            checkpoint_with_prefix(checkpoint_prefix, examples)

    return examples


def flatten_examples(examples):
    rows = []

    for ex in examples:
        question = ex["question"]
        positive = ex["positive"]
        negatives = ex["negatives"]

        for negative in negatives:
            rows.append({
                "dataset": ex["dataset"],
                "question": question,
                "short_answer": ex.get("short_answer"),
                "positive": positive,
                "negative": negative,
            })

    return rows


def build_dataset(
    save_flattened=True,
    dataset_names=None,
    resume=False,
):
    if dataset_names is None:
        dataset_names = [
            "hotpotqa",
            "triviaqa",
            "truthfulqa",
            "squadqa",
        ]

    requested_datasets = set(dataset_names)
    supported_datasets = {
        "hotpotqa",
        "triviaqa",
        "truthfulqa",
        "squadqa",
    }
    unknown_datasets = requested_datasets - supported_datasets

    if unknown_datasets:
        raise ValueError(
            "Unknown dataset names: "
            f"{sorted(unknown_datasets)}. "
            f"Choose from: {sorted(supported_datasets)}."
        )

    all_examples = load_checkpoint_examples() if resume else []

    processors = [
        (
            "hotpotqa",
            {"hotpotqa"},
            process_hotpotqa,
        ),
        (
            "triviaqa",
            {"triviaqa", "triviaqa_wiki"},
            process_triviaqa_wiki,
        ),
        (
            "truthfulqa",
            {"truthfulqa"},
            process_truthfulqa,
        ),
        (
            "squadqa",
            {"squadqa"},
            process_squadqa,
        ),
    ]

    for dataset_key, checkpoint_names, processor in processors:
        if dataset_key not in requested_datasets:
            continue

        skip_questions = existing_questions(
            all_examples,
            checkpoint_names,
        )

        if skip_questions:
            print(
                f"Resuming {dataset_key}: skipping "
                f"{len(skip_questions)} checkpointed questions."
            )

        new_examples = processor(
            split="train",
            max_samples=MAX_SAMPLES_PER_DATASET,
            checkpoint_prefix=all_examples,
            skip_questions=skip_questions,
        )
        all_examples.extend(new_examples)
        save_checkpoint(all_examples)

    if save_flattened:
        rows = flatten_examples(all_examples)
        df = pd.DataFrame(rows)

        df = df.dropna(subset=["question", "positive", "negative"])
        df = df.drop_duplicates(
            subset=["dataset", "question", "positive", "negative"]
        )
        df = df.reset_index(drop=True)

        df.to_csv(OUTPUT_PATH, index=False)

        print("\nDone.")
        print(f"Saved flattened pairwise dataset to: {OUTPUT_PATH}")
        print(f"Number of pairwise examples: {len(df)}")
        print(df.head())

        return df

    df = pd.DataFrame(all_examples)

    df = df.dropna(subset=["question", "positive", "negatives"])
    df = df.drop_duplicates(
        subset=["dataset", "question", "positive"]
    )
    df = df.reset_index(drop=True)

    df.to_csv(OUTPUT_PATH, index=False)

    print("\nDone.")
    print(f"Saved grouped dataset to: {OUTPUT_PATH}")
    print(f"Number of grouped examples: {len(df)}")
    print(df.head())

    return df


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate paired truthful and hallucinated QA claims."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=[
            "hotpotqa",
            "triviaqa",
            "truthfulqa",
            "squadqa",
        ],
        default=[
            "hotpotqa",
            "triviaqa",
            "truthfulqa",
            "squadqa",
        ],
        help="Datasets to process, in canonical pipeline order.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Load the grouped checkpoint, preserve existing examples, "
            "and skip already checkpointed questions."
        ),
    )
    parser.add_argument(
        "--grouped-output",
        action="store_true",
        help="Write grouped negatives instead of flattened pairwise rows.",
    )
    parser.add_argument("--from-checkpoint", action="store_true",
                        help="Flatten the saved generation checkpoint without loading a model.")
    parser.add_argument("--output-path", default="inputs/processed_qa_hallucination_dataset.csv")
    parser.add_argument("--checkpoint-path", default="inputs/generation_checkpoint.csv")
    parser.add_argument("--max-samples", type=int, default=10000)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    OUTPUT_PATH = args.output_path
    CHECKPOINT_PATH = args.checkpoint_path
    MAX_SAMPLES_PER_DATASET = args.max_samples
    if os.path.exists(OUTPUT_PATH) and not args.resume:
        raise SystemExit("Output exists; use --resume or a new --output-path.")
    for path in [OUTPUT_PATH, CHECKPOINT_PATH]:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if args.from_checkpoint:
        if not os.path.isfile(CHECKPOINT_PATH):
            raise SystemExit(f"Missing checkpoint: {CHECKPOINT_PATH}")
        build_dataset(dataset_names=[], resume=True)
        raise SystemExit(0)
    load_generator()
    df = build_dataset(
        save_flattened=not args.grouped_output,
        dataset_names=args.datasets,
        resume=args.resume,
    )
