"""Evaluate a trained reward model on the hidden test split.

Reads:
    data/test_pairs.npz          (prompt, a, b + lengths)
    eval/test_labels.npy         (0 = A is chosen, 1 = B is chosen)
    final_model.ptz              zlib-compressed model produced by train.py
    data/tokenizer.json          (for completeness; model is expected to be
                                  self-sufficient but tokenizer shape info
                                  lives here if needed).

train.py MUST expose a `load_for_eval(model_path: str) -> callable` entry
point inside `final_model.ptz`-adjacent code: since we ship a single `.ptz`
of weights, we also require `train.py` to define a module-level function
`score_batch(prompt_ids, response_ids, prompt_lens, response_lens, model) -> Tensor`
and a `build_and_load(model_path) -> model` function. See the baseline
train.py for the interface.

Prints a single line:
    pairwise_acc=<float>
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import torch


def load_train_module():
    spec = importlib.util.spec_from_file_location("task_train", "train.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    return mod


@torch.inference_mode()
def main():
    if not torch.cuda.is_available():
        print("ERROR: CUDA required", file=sys.stderr)
        sys.exit(1)
    device = torch.device("cuda")

    task = load_train_module()
    for attr in ("build_and_load", "score_batch"):
        if not hasattr(task, attr):
            print(f"ERROR: train.py missing required `{attr}`", file=sys.stderr)
            sys.exit(2)

    model = task.build_and_load("final_model.ptz")
    model = model.to(device).eval()

    test = np.load("data/test_pairs.npz")
    labels = np.load("eval/test_labels.npy")
    prompt = torch.from_numpy(test["prompt"]).long()
    a = torch.from_numpy(test["a"]).long()
    b = torch.from_numpy(test["b"]).long()
    p_len = torch.from_numpy(test["prompt_len"]).long()
    a_len = torch.from_numpy(test["a_len"]).long()
    b_len = torch.from_numpy(test["b_len"]).long()

    N = prompt.size(0)
    B = 64
    correct = 0
    for i in range(0, N, B):
        j = min(i + B, N)
        pr = prompt[i:j].to(device, non_blocking=True)
        ar = a[i:j].to(device, non_blocking=True)
        br = b[i:j].to(device, non_blocking=True)
        prl = p_len[i:j].to(device, non_blocking=True)
        arl = a_len[i:j].to(device, non_blocking=True)
        brl = b_len[i:j].to(device, non_blocking=True)
        ra = task.score_batch(pr, ar, prl, arl, model)
        rb = task.score_batch(pr, br, prl, brl, model)
        pred = (rb > ra).long().cpu().numpy().astype(np.int8)
        correct += int((pred == labels[i:j]).sum())
    acc = correct / N
    print(f"pairwise_acc={acc:.4f}")


if __name__ == "__main__":
    main()
