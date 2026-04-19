#!/usr/bin/env bash
# One-time setup for petitellm-preference.
# - Install deps
# - Download UltraFeedback
# - Train a BPE tokenizer (8k, seed=1337)
# - Pre-tokenize train/val/test pairs
# - Randomize test order and emit hidden labels + pristine sha256
set -euo pipefail
cd "$(dirname "$0")"

PY="${PY:-python3}"

echo "[1/2] Installing requirements ..."
if command -v uv >/dev/null 2>&1; then
    uv pip install -r requirements.txt
else
    "$PY" -m pip install --upgrade pip
    "$PY" -m pip install -r requirements.txt
fi

mkdir -p data eval

if [ -f data/train_pairs.npz ] && [ -f data/val_pairs.npz ] && [ -f data/test_pairs.npz ] \
   && [ -f data/tokenizer.json ] && [ -f eval/test_labels.npy ] && [ -f data/test_labels_sha.txt ]; then
    echo "[2/2] Data already prepared, skipping."
    echo "      test_labels_sha: $(cat data/test_labels_sha.txt)"
    exit 0
fi

echo "[2/2] Building dataset + tokenizer ..."
"$PY" - <<'PY_SCRIPT'
import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

SEED = 1337
random.seed(SEED); np.random.seed(SEED)

DATA = Path("data"); DATA.mkdir(exist_ok=True)
EVAL = Path("eval"); EVAL.mkdir(exist_ok=True)

print("loading openbmb/UltraFeedback ...")
ds = load_dataset("openbmb/UltraFeedback", split="train")
print(f"  {len(ds)} raw examples")

# Build pairs. Use aspect 'overall_score' on `completions[i].annotations.{aspect}.Rating`.
def score_of(comp):
    try:
        ann = comp.get("annotations") or {}
        overall = ann.get("overall_score")
        if isinstance(overall, (int, float)):
            return float(overall)
        if isinstance(overall, dict):
            r = overall.get("Rating") or overall.get("rating")
            if r is not None:
                try: return float(r)
                except Exception: return None
        # Fallback: average of helpfulness/honesty ratings if available.
        ratings = []
        for k in ("helpfulness", "honesty", "instruction_following", "truthfulness"):
            sub = ann.get(k)
            if isinstance(sub, dict):
                r = sub.get("Rating") or sub.get("rating")
                if r is not None:
                    try: ratings.append(float(r))
                    except Exception: pass
        return sum(ratings)/len(ratings) if ratings else None
    except Exception:
        return None

pairs = []
for ex in ds:
    prompt = ex.get("instruction") or ex.get("prompt") or ""
    comps = ex.get("completions") or []
    if len(comps) < 2:
        continue
    scored = []
    for c in comps:
        s = score_of(c)
        resp = c.get("response") or ""
        if s is None or not resp or not prompt:
            continue
        scored.append((s, resp))
    if len(scored) < 2:
        continue
    scored.sort(key=lambda x: x[0], reverse=True)
    s_hi, resp_hi = scored[0]
    s_lo, resp_lo = scored[-1]
    if s_hi <= s_lo:  # no real preference signal
        continue
    pairs.append({"prompt": prompt, "chosen": resp_hi, "rejected": resp_lo})
print(f"  built {len(pairs)} valid preference pairs")

# Cap for speed: use first ~60k (or all, if fewer).
MAX_PAIRS = 60000
random.Random(SEED).shuffle(pairs)
pairs = pairs[:MAX_PAIRS]
print(f"  using {len(pairs)} pairs after cap")

# Train/val/test split.
n = len(pairs)
n_test = min(3000, n // 10)
n_val = min(2000, n // 10)
n_train = n - n_test - n_val
train_pairs = pairs[:n_train]
val_pairs = pairs[n_train:n_train + n_val]
test_pairs = pairs[n_train + n_val:]
print(f"  split: train={len(train_pairs)} val={len(val_pairs)} test={len(test_pairs)}")

# Train a BPE tokenizer on train prompts + chosen + rejected.
print("training BPE tokenizer ...")
tok = Tokenizer(BPE(unk_token="<unk>"))
tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
tok.decoder = ByteLevelDecoder()
trainer = BpeTrainer(
    vocab_size=8192,
    special_tokens=["<pad>", "<unk>", "<s>", "</s>", "<prompt>", "<resp>", "<end>"],
    min_frequency=2,
    show_progress=False,
)
def texts_iter():
    for p in train_pairs:
        yield p["prompt"]
        yield p["chosen"]
        yield p["rejected"]
tok.train_from_iterator(texts_iter(), trainer=trainer)
tok.save(str(DATA / "tokenizer.json"))
print(f"  vocab_size = {tok.get_vocab_size()}")

PAD = tok.token_to_id("<pad>")
PROMPT_SEP = tok.token_to_id("<prompt>")
RESP_SEP = tok.token_to_id("<resp>")
END = tok.token_to_id("<end>")

MAX_PROMPT = 256
MAX_RESP = 384
def enc(text, max_len):
    ids = tok.encode(text).ids
    return ids[:max_len]

def pack_split(split_pairs, keys):
    prompts, fields_a, fields_b = [], [], []
    p_lens, a_lens, b_lens = [], [], []
    for p in split_pairs:
        pi = enc(p["prompt"], MAX_PROMPT)
        a = enc(p[keys[0]], MAX_RESP)
        b = enc(p[keys[1]], MAX_RESP)
        prompts.append(pi); fields_a.append(a); fields_b.append(b)
        p_lens.append(len(pi)); a_lens.append(len(a)); b_lens.append(len(b))
    def to_arr(lst, max_len):
        out = np.full((len(lst), max_len), PAD, dtype=np.int32)
        for i, ids in enumerate(lst):
            out[i, :len(ids)] = ids
        return out
    return {
        "prompt": to_arr(prompts, MAX_PROMPT),
        keys[0]: to_arr(fields_a, MAX_RESP),
        keys[1]: to_arr(fields_b, MAX_RESP),
        "prompt_len": np.asarray(p_lens, dtype=np.int32),
        f"{keys[0]}_len": np.asarray(a_lens, dtype=np.int32),
        f"{keys[1]}_len": np.asarray(b_lens, dtype=np.int32),
    }

print("pre-tokenizing train ...")
np.savez_compressed(DATA / "train_pairs.npz", **pack_split(train_pairs, ("chosen", "rejected")))
print("pre-tokenizing val ...")
np.savez_compressed(DATA / "val_pairs.npz", **pack_split(val_pairs, ("chosen", "rejected")))

# Test: randomize A/B order and hide labels.
print("pre-tokenizing + randomizing test ...")
rng = random.Random(SEED + 1)
test_keys = ("a", "b")
test_randomized = []
labels = []  # 0 = A is chosen, 1 = B is chosen
for p in test_pairs:
    if rng.random() < 0.5:
        test_randomized.append({"prompt": p["prompt"], "a": p["chosen"], "b": p["rejected"]})
        labels.append(0)
    else:
        test_randomized.append({"prompt": p["prompt"], "a": p["rejected"], "b": p["chosen"]})
        labels.append(1)
np.savez_compressed(DATA / "test_pairs.npz", **pack_split(test_randomized, test_keys))

labels_arr = np.asarray(labels, dtype=np.int8)
np.save(EVAL / "test_labels.npy", labels_arr)

# Pristine hash (read-only reference) so eval can audit if train.py tampered.
sha = hashlib.sha256(open(EVAL / "test_labels.npy", "rb").read()).hexdigest()
(DATA / "test_labels_sha.txt").write_text(sha + "\n")
print(f"  labels SHA-256: {sha}")
print("done.")
PY_SCRIPT

echo "Done."
