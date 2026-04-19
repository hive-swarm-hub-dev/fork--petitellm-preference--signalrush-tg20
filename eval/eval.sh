#!/usr/bin/env bash
# Evaluate train.py for petitellm-preference.
set -uo pipefail
cd "$(dirname "$0")/.."

summary() {
    local acc="${1:-ERROR}"
    local artifact_bytes="${2:-0}"
    local line_count="${3:-0}"
    local sha="${4:-none}"
    local valid="${5:-false}"
    echo "---"
    printf "pairwise_acc:      %s\n" "$acc"
    printf "artifact_bytes:    %s\n" "$artifact_bytes"
    printf "line_count:        %s\n" "$line_count"
    printf "test_labels_sha:   %s\n" "$sha"
    printf "valid:             %s\n" "$valid"
}

if [ ! -f "train.py" ]; then
    echo "ERROR: train.py not found." >&2
    summary; exit 0
fi
LINE_COUNT=$(wc -l < train.py | tr -d ' ')

if ! python3 -c "import torch; assert torch.cuda.is_available(), 'No CUDA'" 2>/dev/null; then
    echo "ERROR: CUDA not available." >&2
    summary "ERROR" "0" "$LINE_COUNT" "none" "false"; exit 0
fi

for f in data/train_pairs.npz data/val_pairs.npz data/test_pairs.npz data/tokenizer.json eval/test_labels.npy data/test_labels_sha.txt; do
    if [ ! -f "$f" ]; then
        echo "ERROR: missing $f. Run: bash prepare.sh" >&2
        summary "ERROR" "0" "$LINE_COUNT" "none" "false"; exit 0
    fi
done

# Pristine hash recorded at prepare time.
PRISTINE_SHA=$(head -1 data/test_labels_sha.txt | tr -d '\n' | tr -d ' ')
CURRENT_SHA=$(python3 -c "import hashlib; print(hashlib.sha256(open('eval/test_labels.npy','rb').read()).hexdigest())")
CURRENT_SHA_SHORT=$(printf '%s' "$CURRENT_SHA" | cut -c1-16)

if [ "$CURRENT_SHA" != "$PRISTINE_SHA" ]; then
    echo "ERROR: eval/test_labels.npy hash mismatch: expected $PRISTINE_SHA got $CURRENT_SHA" >&2
    summary "ERROR" "0" "$LINE_COUNT" "$CURRENT_SHA_SHORT" "false"; exit 0
fi

TMPLOG=$(mktemp); trap 'rm -f "$TMPLOG"' EXIT

TRAIN_EXIT=0
timeout 720 python3 train.py 2>&1 | tee "$TMPLOG" >&2 || TRAIN_EXIT=$?
if [ "$TRAIN_EXIT" -ne 0 ] && [ "$TRAIN_EXIT" -ne 124 ]; then
    echo "ERROR: training exited with code $TRAIN_EXIT." >&2
    summary "ERROR" "0" "$LINE_COUNT" "$CURRENT_SHA_SHORT" "false"; exit 0
fi

# Hash again AFTER training — agent should not have overwritten it.
POST_SHA=$(python3 -c "import hashlib; print(hashlib.sha256(open('eval/test_labels.npy','rb').read()).hexdigest())")
POST_SHA_SHORT=$(printf '%s' "$POST_SHA" | cut -c1-16)
if [ "$POST_SHA" != "$PRISTINE_SHA" ]; then
    echo "ERROR: eval/test_labels.npy was modified during training: $POST_SHA" >&2
    summary "ERROR" "0" "$LINE_COUNT" "$POST_SHA_SHORT" "false"; exit 0
fi

if [ ! -f final_model.ptz ]; then
    echo "ERROR: final_model.ptz not produced." >&2
    summary "ERROR" "0" "$LINE_COUNT" "$POST_SHA_SHORT" "false"; exit 0
fi

EVAL_EXIT=0
EVAL_OUT=$(python3 eval/evaluate.py 2>&1) || EVAL_EXIT=$?
echo "$EVAL_OUT" >&2
if [ "$EVAL_EXIT" -ne 0 ]; then
    echo "ERROR: evaluate.py exited $EVAL_EXIT." >&2
    summary "ERROR" "0" "$LINE_COUNT" "$POST_SHA_SHORT" "false"; exit 0
fi

ACC=$(printf '%s' "$EVAL_OUT" | grep -oE 'pairwise_acc=[0-9]+\.[0-9]+' | tail -1 | cut -d= -f2)
if [ -z "$ACC" ]; then
    echo "ERROR: could not parse pairwise_acc." >&2
    summary "ERROR" "0" "$LINE_COUNT" "$POST_SHA_SHORT" "false"; exit 0
fi

MODEL_BYTES=$(stat -c%s final_model.ptz 2>/dev/null || stat -f%z final_model.ptz)
CODE_BYTES=$(stat -c%s train.py 2>/dev/null || stat -f%z train.py)
ARTIFACT_BYTES=$(( MODEL_BYTES + CODE_BYTES ))

VALID="true"
if [ "$ARTIFACT_BYTES" -gt 16000000 ]; then
    echo "WARNING: artifact_bytes $ARTIFACT_BYTES > 16,000,000" >&2
    VALID="false"
fi

summary "$ACC" "$ARTIFACT_BYTES" "$LINE_COUNT" "$POST_SHA_SHORT" "$VALID"
