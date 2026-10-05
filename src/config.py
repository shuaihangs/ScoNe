import torch

MODEL_NAMES = [
    "Qwen/Qwen2.5-3B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "microsoft/Phi-3.5-mini-instruct",
]
DATASET_NAMES = ["hotpotqa", "triviaqa", "truthfulqa", "squadqa"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CSV_PATH = "inputs/processed_qa_hallucination_dataset.csv"
OUTPUT_DIR = "outputs/scone"
CHECKPOINT_DIR = OUTPUT_DIR + "/checkpoints"
HISTORY_DIR = OUTPUT_DIR + "/histories"
FEATURE_CACHE_DIR = OUTPUT_DIR + "/feature_cache"
MAX_LENGTH = 128
BATCH_SIZE = 16
LR = 2e-4
MAX_EPOCHS = 25
SEED = 42
VALIDATION_RATIO = 0.2
PROJ_DIM = 64
DROPOUT = 0.4
WEIGHT_DECAY = 3e-3
CACHE_FROZEN_LLM_FEATURES = True
FEATURE_CACHE_BATCH_SIZE = 16
NEIGHBOUR_LLM_BATCH_SIZE = 8
USE_SHORT_ANSWER_IN_TEXT = False
NORMALIZE_PROJECTED_STATES = False
USE_FEATURE_STANDARDIZATION = False

MODEL_REVISIONS = {'Qwen/Qwen2.5-3B-Instruct': 'aa8e72537993ba99e69dfaafa59ed015b17504d1', 'meta-llama/Llama-3.2-3B-Instruct': '0cb88a4f764b7a12671c53f0838cd831a0843b95', 'microsoft/Phi-3.5-mini-instruct': '2fe192450127e6a83f7441aef6e3ca586c338b77'}
