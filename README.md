# SCoNE: question-conditioned hallucination detection

SCoNE freezes an instruction-tuned language model, mean-pools claim-token states at five transformer depths, applies a shared 64-dimensional projection, and concatenates the projections into a 320-dimensional representation. A jointly trained linear head produces the hallucination logit (truthful = 0, hallucinated = 1).

The reported objective is:

```text
loss = alpha * BCE + (1 - alpha) * soft_neighbour_loss / 6
```

Training uses question-only semantic retrieval. Positive neighbours have the same factuality label; negative candidates include opposite-label neighbours and opposite-label claims in the batch. Similarity-based positive weights are detached. Inference uses the retained projection and linear classifier, without retrieval or classifier refitting.

## Setup

Run all commands from the repository root. The tested environment is Linux, Python 3.10.12, PyTorch 2.5.1 + CUDA 12.1, and an NVIDIA RTX A5000 with 24 GB memory. Training dependencies are pinned in `requirements.txt`, and backbone revisions are pinned in `src/config.py`.

Create and activate a virtual environment:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
hf auth login
```

The Llama model requires access through your Hugging Face account. Downloads require network access; subsequent runs can use the local model cache. Each experiment keeps its own feature cache and checkpoints, so reserve disk space for the requested runs.

## Dataset generation

The generation model is Qwen2.5-1.5B-Instruct. The pipeline reads the training splits of HotpotQA (`distractor`), TriviaQA (`rc.wikipedia`), TruthfulQA, and SQuAD. It converts short reference answers to sentences and generates incorrect alternatives for HotpotQA, TriviaQA, and SQuAD. TruthfulQA uses its original question, best answer, and provided incorrect answers. Each incorrect answer becomes a paired row with the factual candidate.

Install the generation dependencies and build a paired corpus:

```bash
python -m pip install -r requirements-generation.txt
python -m src.generate_answers \
  --datasets hotpotqa triviaqa truthfulqa squadqa \
  --max-samples 10000 \
  --output-path inputs/generated_qa.csv \
  --checkpoint-path inputs/generated_qa_checkpoint.csv
```

Add `--resume` with the same paths to continue interrupted generation. The checkpoint contains grouped answers; the output CSV contains `dataset,question,short_answer,positive,negative`. Filtering means fewer than 10,000 retained questions per dataset. TruthfulQA can have multiple paired rows per question. The `short_answer` field is not included in detector inputs.

A saved grouped checkpoint can be flattened without loading the generator:

```bash
python -m src.generate_answers --from-checkpoint \
  --checkpoint-path inputs/generated_qa_checkpoint.csv \
  --output-path inputs/rebuilt_qa.csv
```

### Data for exact reproduction

Exact reproduction of the reported scores requires the fixed processed CSV identified below. **This file is not included in the Git checkout.** If you have the reference corpus, place it at this path and verify its checksum:

The reference corpus contains 32,376 paired rows: HotpotQA 9,516; TriviaQA 9,788; TruthfulQA 3,369; SQuAD 9,703. All rows for a question are kept together after whitespace and case normalisation.

If the reference corpus is unavailable, use the generation commands above. To train on the generated corpus, append `--csv-path inputs/generated_qa.csv` to any training or launcher command below. For example:

```bash
bash scripts/run_reported_experiments.sh final --csv-path inputs/generated_qa.csv
```

Generation uses sampling; source-data revisions, generation environments, and interrupted sampling can change the corpus. Training on a newly generated or re-flattened CSV is therefore not expected to reproduce the reference scores exactly. The commands below default to the reference-corpus path.

## Reproduce the ablations

The complete grid is three backbones × four source datasets × five alpha values × five folds for each K in `{10,30,50}`. Each K has 300 training runs. The outer 80/20 question-grouped split is fixed; the five inner folds partition only the outer training set.

```bash
bash scripts/run_reported_experiments.sh ablation --dry-run
bash scripts/run_reported_experiments.sh ablation
```

The launcher runs the three K sweeps sequentially. Add `--resume` to restart completed/partial experiments safely. A directory lock prevents concurrent writers to one experiment. Use separate output directories for distinct K values or configurations.

To run one K sweep:

```bash
python -u run_model.py --k-neighbours 30 \
  --alpha-values 0 0.25 0.5 0.75 1 --cv-folds 5 \
  --output-dir outputs/ablation_k30
```

For a single configuration and fold:

```bash
python -u run_model.py \
  --models Qwen/Qwen2.5-3B-Instruct --train-datasets truthfulqa \
  --alpha-values 0.5 --k-neighbours 30 --cv-folds 5 --folds 1 \
  --output-dir outputs/qwen_truthful_k30_fold1
```

`--scaled-alpha-values` is an alias of `--alpha-values`. Alpha always weights BCE: alpha=1 is BCE-only and sets effective K=0; alpha=0 leaves the retained classifier without BCE supervision and is only a diagnostic endpoint.

## Reproduce the reported 80/20 results

The final results in `reproducibility/reference_results/final.csv` use **K=10**, with **Qwen alpha=0.5, Llama alpha=0.75, and Phi alpha=0.75**. The final-training launcher uses these settings for every source dataset.

```bash
bash scripts/run_reported_experiments.sh final --dry-run
bash scripts/run_reported_experiments.sh final
```

This retrains a new detector for each backbone/source combination on the full 80% training partition, with no inner CV or second training stage. Results are written under `outputs/final/{qwen,llama,phi}` and combined in `outputs/final/experiment_summary.csv` (12 training runs, 60 evaluation rows including training diagnostics).

To evaluate another alpha/K configuration, use `run_model.py --cv-folds 1` with the desired settings and a separate output directory.

## Fixed training settings

| Setting | Reported value |
|---|---|
| Epochs / checkpoint | 25; retain the final epoch, no early stopping |
| Optimiser | AdamW, learning rate 0.0002, weight decay 0.003 |
| Batch size | 16 truthful–hallucinated pairs |
| Projection | Linear → LayerNorm → GELU → Dropout(0.4) → Linear; 64 dimensions per depth |
| Classifier | Linear(320, 1), jointly trained and retained |
| Temperatures | T = Tp = 0.2 |
| Contrastive divisor | 6 |
| Input | `Question: {q}\nClaim: {a}`, maximum 128 tokens; preserve claim tokens preferentially |
| Depths | Qwen [0,9,18,27,35]; Llama [0,7,14,21,27]; Phi [0,8,16,24,31] |
| Split seed | 42 plus canonical dataset index |
| Training seed | 42 + fold index − 1; 42 for final 80/20 runs |
| Backbone / cached states | Frozen FP16 on CUDA; five pooled depth vectors cached in FP16 |
| Feature-cache / question-embedding batches | 16 / 8 |

Indices refer to transformer-block outputs, excluding the embedding state. The classifier receives unnormalised projected features; contrastive similarities use L2 normalisation. Neighbours come only from the current training partition, exclude the anchor question, and use one representative paired row per distinct neighbouring question.

## Outputs and evaluation

Each run directory contains:

- `experiment_summary.csv`: per-fold metrics for each source/evaluation dataset.
- `experiment_summary_cv_averaged.csv`: averages and sample standard deviations for completed five-fold configurations.
- `checkpoints/`: final projection and retained classifier, optimiser state, and settings.
- `feature_cache/`: fixed, untrained backbone features.
- `run_manifest.json`: input hash, settings, package versions, and source hashes for each training run.

For coefficient/K selection, use **only** rows whose `eval_dataset` is `<source>_inner_val`. Average the five folds for each source, then macro-average the four sources if selecting one configuration per backbone. Candidate mixed-loss coefficients are `{0.25,0.5,0.75}`; keep alpha=1 as the BCE baseline and alpha=0 as diagnostic.

Cross-validation also records source outer-test and other-dataset outer-test metrics after training. These are not inputs to training or hyperparameter selection. Final 80/20 runs use `<source>_val` to name the held-out 20% in-domain partition, and evaluate on the complete other datasets. Do not mix these final OOD scores with CV OOD scores, which use different evaluation subsets.

Load a saved detector with `src.model.load_detector_checkpoint`. This function also supports the reference checkpoint format. The implementation preserves the initialisation and batch-shuffle sequence used for the reference experiments.


## Code layout

```text
run_model.py                       training CLI
src/config.py                      fixed settings and backbone revisions
src/generate_answers.py            dataset-generation pipeline
src/data.py                        grouped splits, tokenisation, retrieval, caches
src/model.py                       projection, linear energy, checkpoint loading
src/soft_neighbour.py              contrastive objective
src/training.py                    joint training and final evaluation
src/evaluation.py                  AUROC and other metrics
src/experiment.py                  experiment paths and split audits
scripts/run_reported_experiments.sh ablation and final launch commands
```

