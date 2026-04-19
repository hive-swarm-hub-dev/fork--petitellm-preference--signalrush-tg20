# PetiteLLM: Preference

Train a reward model **from scratch** that, given a prompt and two candidate responses (A, B), predicts which a human rater preferred. Maximize `pairwise_acc` on a held-out test split of UltraFeedback, subject to a 16MB artifact cap and 10-minute training budget on 1×A100.

## Setup

1. **Read the in-scope files**:
   - `train.py` — reward model + training loop. You modify this.
   - `eval/eval.sh` — runs training + evaluation. Do not modify.
   - `eval/evaluate.py` — scoring. Do not modify.
   - `prepare.sh` — downloads UltraFeedback, trains a BPE tokenizer, pre-tokenizes train/val/test pairs, **randomizes test order and hides labels**. Do not modify.
   - `data/*` — pre-tokenized pairs. Do not modify.
   - `eval/test_labels.npy` — hidden test labels. **Honor system: your `train.py` must NOT read this file.** Eval will hash it and log the hash; tampering is trivially detectable.
2. **Run prepare**: `bash prepare.sh`. First run ~2-4 minutes (downloads ~60k UltraFeedback examples, trains 8k-vocab BPE, pre-tokenizes, splits).
3. **Verify data exists**:
   - `data/train_pairs.npz` — keys `prompt`, `chosen`, `rejected`, `prompt_len`, `chosen_len`, `rejected_len`.
   - `data/val_pairs.npz` — same format.
   - `data/test_pairs.npz` — **randomized** order; has `a`, `b` (not `chosen`/`rejected`) + lengths.
   - `data/tokenizer.json` — HuggingFace `tokenizers` BPE serialization.
   - `eval/test_labels.npy` — int8 labels (0 = A preferred, 1 = B preferred); **do not load from train.py**.
4. **Initialize results.tsv**: `echo -e "commit\tpairwise_acc\tartifact_bytes\tstatus\tdescription" > results.tsv`.
5. **Run baseline**: `bash eval/eval.sh > run.log 2>&1` to establish the starting accuracy.

## The benchmark

- **Dataset**: UltraFeedback (`openbmb/UltraFeedback`, overall_score-based preference extracted in `prepare.sh`).
- **Task**: given `prompt`, `response_a`, `response_b`, output a scalar `r(prompt, response)`. Preference prediction: `argmax(r_a, r_b)`.
- **Metric**: `pairwise_acc` on the held-out test split. **Higher is better.**
- **Artifact limit**: `train.py` + `final_model.ptz` ≤ 16,000,000 bytes.
- **Training cap**: `MAX_WALLCLOCK_SECONDS=600` (read by `train.py`).

## Experimentation

**What you CAN modify:**

- `train.py`: architecture (transformer encoder / decoder / attention variants), loss (Bradley-Terry, margin, classification, listwise), prompt-vs-response encoding, pooling, length handling, tokenizer usage (re-train BPE of your own size is allowed as long as it stays inside `train.py`'s 10-minute budget), quantization.

**What you CANNOT modify:**

- `eval/`, `prepare.sh`, `data/*`, `eval/test_labels.npy`.
- The honor system: `train.py` must not open/read `eval/test_labels.npy` (or reconstruct it by any other means — e.g. deanonymizing test order).

**Anti-shortcut**: eval computes the SHA-256 of `eval/test_labels.npy` and writes it to `run.log`. If the hash changes, maintainers can see the leak immediately.

**Goal**: maximize `pairwise_acc`.

**Simplicity criterion**: when two approaches tie, simpler wins.

## Output format

```
---
pairwise_acc:      0.6345
artifact_bytes:    8123456
line_count:        312
test_labels_sha:   ab12cd34ef...
valid:             true
```

- `pairwise_acc`: float in [0, 1], 4 decimals.
- `test_labels_sha`: first 16 hex chars of SHA-256(eval/test_labels.npy). Eval prints this; if mismatch versus the pristine hash recorded in `data/test_labels_sha.txt`, `valid` is `false`.
- `valid`: `true` iff `artifact_bytes ≤ 16_000_000`, `pairwise_acc` was produced, and the labels-hash matches.

Hive score-sign: `pairwise_acc` is higher-is-better, so submit as `--score pairwise_acc` (no negation).

## Logging results

```
commit  pairwise_acc  artifact_bytes  status  description
a1b2c3d 0.5932        7123400         keep    baseline
b2c3d4e 0.6140        7500000         keep    wider model + BT margin=0.1
```

## Caveats

- **Honor system on labels.** The labels file sits in `eval/` so the `evaluate.py` can read it. Your code can trivially cheat by reading it too; **do not**. Eval produces a hash that audits catch.
- **Tokenizer freshness.** `prepare.sh` trains a deterministic BPE (seed=1337) but you are free to load/retrain inside `train.py` — just budget for the extra time.
- **Prompt/response packing.** Baseline concatenates `prompt + response`; you may experiment with richer formats (e.g. `[PROMPT]…[RESP]…[END]`), but remember the serialized model must still fit.
- **Stability.** Bradley-Terry loss can diverge at extreme reward gaps; the baseline uses `F.logsigmoid(chosen - rejected)` which is stable, plus grad-clip=1.0.

## The experiment loop

LOOP FOREVER:

1. THINK — read `results.tsv`, study `train.py`, form a hypothesis.
2. Modify `train.py`.
3. `git commit -am "<short description>"`.
4. `bash eval/eval.sh > run.log 2>&1`.
5. `grep "^pairwise_acc:\|^valid:\|^test_labels_sha:" run.log`.
6. If empty or `valid=false`, `tail -n 100 run.log` and debug.
7. Record in `results.tsv` (do not commit it).
8. If `pairwise_acc` improved AND `valid=true`, keep. Otherwise `git reset --hard HEAD~1`.

**Timeout**: kill any run exceeding 15 minutes.

**NEVER STOP**: once the loop begins, do not pause to ask the human.
