"""
PetiteLLM: Preference — reward model training script.

Baseline: small transformer encoder takes [prompt <resp> response <end>] tokens,
mean-pools, feeds through a scalar head to produce r(prompt, response). Trained
with pairwise Bradley-Terry loss on UltraFeedback pairs. Saves fp16 + zlib
compressed artifact. Provides two entry points required by eval/evaluate.py:

    build_and_load(model_path: str) -> nn.Module
    score_batch(prompt_ids, response_ids, prompt_lens, response_lens, model) -> Tensor

Agents should iterate on ARCHITECTURE, LOSS, TOKENIZATION, OPTIMIZER, QUANTIZATION.
"""
from __future__ import annotations

import io
import json
import math
import os
import random
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------- hyperparameters -----------------------------


class HP:
    seed = int(os.environ.get("SEED", 1337))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))

    # Model arch
    vocab_size = int(os.environ.get("VOCAB_SIZE", 8192))
    model_dim = int(os.environ.get("MODEL_DIM", 256))
    num_layers = int(os.environ.get("NUM_LAYERS", 4))
    num_heads = int(os.environ.get("NUM_HEADS", 4))
    mlp_mult = int(os.environ.get("MLP_MULT", 4))
    max_prompt_len = 256
    max_resp_len = 384
    max_seq_len = max_prompt_len + max_resp_len + 2  # +<resp>,<end>

    # Optim
    lr = float(os.environ.get("LR", 3e-4))
    batch_size = int(os.environ.get("BATCH_SIZE", 16))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 100))
    weight_decay = float(os.environ.get("WEIGHT_DECAY", 0.01))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    grad_clip = float(os.environ.get("GRAD_CLIP", 1.0))

    # Token ids populated from tokenizer.json at runtime.
    pad_id = 0
    resp_sep_id = 5
    end_id = 6


# ----------------------------- model -----------------------------


class RewardModel(nn.Module):
    def __init__(self, vocab, dim, num_layers, num_heads, mlp_mult, max_seq_len, pad_id):
        super().__init__()
        self.pad_id = pad_id
        self.embed = nn.Embedding(vocab, dim, padding_idx=pad_id)
        self.pos = nn.Embedding(max_seq_len, dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=dim * mlp_mult,
            dropout=0.0,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.ln = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, 1)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """ids: (B, L) int64.  Returns (B,) scalar reward."""
        B, L = ids.shape
        pad_mask = ids.eq(self.pad_id)
        pos = torch.arange(L, device=ids.device).unsqueeze(0).expand(B, L)
        h = self.embed(ids) + self.pos(pos)
        h = self.encoder(h, src_key_padding_mask=pad_mask)
        h = self.ln(h)
        # Mean-pool over non-pad tokens (avoid divide-by-zero).
        mask = (~pad_mask).unsqueeze(-1).to(h.dtype)
        denom = mask.sum(dim=1).clamp_min(1.0)
        pooled = (h * mask).sum(dim=1) / denom
        return self.head(pooled).squeeze(-1)


# ----------------------------- packing -----------------------------


def pack_prompt_response(
    prompt: torch.Tensor, resp: torch.Tensor, prompt_len: torch.Tensor, resp_len: torch.Tensor,
    resp_sep: int, end_id: int, pad_id: int, max_total: int,
) -> torch.Tensor:
    """Builds [prompt[:pl], resp_sep, resp[:rl], end_id, pad...] as (B, L)."""
    B = prompt.size(0)
    out = torch.full((B, max_total), pad_id, dtype=torch.long, device=prompt.device)
    for i in range(B):
        pl = int(prompt_len[i].item())
        rl = int(resp_len[i].item())
        total = pl + 1 + rl + 1
        if total > max_total:
            # Truncate response head first to fit.
            rl = max(0, max_total - pl - 2)
            total = pl + 1 + rl + 1
        out[i, :pl] = prompt[i, :pl]
        out[i, pl] = resp_sep
        out[i, pl + 1:pl + 1 + rl] = resp[i, :rl]
        out[i, pl + 1 + rl] = end_id
    return out


# ----------------------------- eval entry points -----------------------------


def score_batch(prompt_ids, response_ids, prompt_lens, response_lens, model):
    packed = pack_prompt_response(
        prompt_ids, response_ids, prompt_lens, response_lens,
        resp_sep=HP.resp_sep_id, end_id=HP.end_id, pad_id=HP.pad_id,
        max_total=HP.max_seq_len,
    )
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        return model(packed).float()


def build_and_load(model_path: str):
    """Instantiate architecture, decompress + load fp16 state dict."""
    tok_meta = json.load(open("data/tokenizer.json"))
    vocab_sz = max(HP.vocab_size, len(tok_meta.get("model", {}).get("vocab", {})) or HP.vocab_size)
    # Populate special ids from tokenizer.json (added_tokens section).
    try:
        for t in tok_meta.get("added_tokens", []):
            content = t.get("content")
            tid = t.get("id")
            if content == "<pad>" and isinstance(tid, int): HP.pad_id = tid
            elif content == "<resp>" and isinstance(tid, int): HP.resp_sep_id = tid
            elif content == "<end>" and isinstance(tid, int): HP.end_id = tid
    except Exception:
        pass

    model = RewardModel(
        vocab=vocab_sz, dim=HP.model_dim, num_layers=HP.num_layers,
        num_heads=HP.num_heads, mlp_mult=HP.mlp_mult,
        max_seq_len=HP.max_seq_len, pad_id=HP.pad_id,
    )

    with open(model_path, "rb") as f:
        blob = f.read()
    buf = io.BytesIO(zlib.decompress(blob))
    sd = torch.load(buf, map_location="cpu")
    out = {}
    for k, v in model.state_dict().items():
        out[k] = sd[k].to(dtype=v.dtype) if k in sd else v
    model.load_state_dict(out, strict=True)
    return model


# ----------------------------- data loader -----------------------------


class PairDataset:
    def __init__(self, path: str):
        z = np.load(path)
        self.prompt = torch.from_numpy(z["prompt"]).long()
        self.chosen = torch.from_numpy(z["chosen"]).long()
        self.rejected = torch.from_numpy(z["rejected"]).long()
        self.p_len = torch.from_numpy(z["prompt_len"]).long()
        self.c_len = torch.from_numpy(z["chosen_len"]).long()
        self.r_len = torch.from_numpy(z["rejected_len"]).long()
        self.n = self.prompt.size(0)

    def batch(self, idxs, device):
        return (
            self.prompt[idxs].to(device, non_blocking=True),
            self.chosen[idxs].to(device, non_blocking=True),
            self.rejected[idxs].to(device, non_blocking=True),
            self.p_len[idxs].to(device, non_blocking=True),
            self.c_len[idxs].to(device, non_blocking=True),
            self.r_len[idxs].to(device, non_blocking=True),
        )


# ----------------------------- train -----------------------------


def bradley_terry_loss(r_chosen, r_rejected, margin: float = 0.0):
    return -F.logsigmoid(r_chosen - r_rejected - margin).mean()


def save_model_compressed(model: nn.Module, path: str) -> int:
    sd = {k: v.detach().to(torch.float16).cpu() for k, v in model.state_dict().items()}
    buf = io.BytesIO()
    torch.save(sd, buf)
    blob = zlib.compress(buf.getvalue(), level=9)
    with open(path, "wb") as f:
        f.write(blob)
    return os.path.getsize(path)


def main():
    random.seed(HP.seed); np.random.seed(HP.seed); torch.manual_seed(HP.seed)
    if not torch.cuda.is_available():
        print("ERROR: CUDA required", file=sys.stderr); sys.exit(1)
    device = torch.device("cuda")

    # Populate special ids from tokenizer.
    tok_meta = json.load(open("data/tokenizer.json"))
    for t in tok_meta.get("added_tokens", []):
        c = t.get("content"); tid = t.get("id")
        if c == "<pad>" and isinstance(tid, int): HP.pad_id = tid
        elif c == "<resp>" and isinstance(tid, int): HP.resp_sep_id = tid
        elif c == "<end>" and isinstance(tid, int): HP.end_id = tid
    vocab_sz = max(HP.vocab_size, len(tok_meta.get("model", {}).get("vocab", {})) or HP.vocab_size)
    print(f"vocab_size={vocab_sz} pad={HP.pad_id} resp={HP.resp_sep_id} end={HP.end_id}")

    train_ds = PairDataset("data/train_pairs.npz")
    val_ds = PairDataset("data/val_pairs.npz")
    print(f"train n={train_ds.n}  val n={val_ds.n}")

    model = RewardModel(
        vocab=vocab_sz, dim=HP.model_dim, num_layers=HP.num_layers,
        num_heads=HP.num_heads, mlp_mult=HP.mlp_mult,
        max_seq_len=HP.max_seq_len, pad_id=HP.pad_id,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"reward model params: {n_params/1e6:.2f}M")

    opt = torch.optim.AdamW(
        model.parameters(), lr=HP.lr, betas=(HP.beta1, HP.beta2),
        weight_decay=HP.weight_decay,
    )

    def lr_at(step):
        if step < HP.warmup_steps:
            return HP.lr * (step + 1) / HP.warmup_steps
        return HP.lr

    start = time.time()
    step = 0
    model.train()
    rng = np.random.default_rng(HP.seed)
    while True:
        elapsed = time.time() - start
        if elapsed >= HP.max_wallclock_seconds:
            break
        for g in opt.param_groups: g["lr"] = lr_at(step)
        idxs = rng.integers(0, train_ds.n, size=HP.batch_size)
        idxs_t = torch.from_numpy(idxs).long()
        pr, ch, rj, pl, cl, rl = train_ds.batch(idxs_t, device)
        rc = score_batch(pr, ch, pl, cl, model)
        rr = score_batch(pr, rj, pl, rl, model)
        loss = bradley_terry_loss(rc, rr)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if HP.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), HP.grad_clip)
        opt.step()
        step += 1
        if step % 50 == 0:
            acc = (rc > rr).float().mean().item()
            print(f"step={step} t={elapsed:.0f}s loss={loss.item():.4f} train_batch_acc={acc:.3f}", flush=True)

    # Quick val
    model.eval()
    with torch.inference_mode():
        total = 0; correct = 0
        B = 64
        for i in range(0, val_ds.n, B):
            j = min(i + B, val_ds.n)
            idxs_t = torch.arange(i, j).long()
            pr, ch, rj, pl, cl, rl = val_ds.batch(idxs_t, device)
            rc = score_batch(pr, ch, pl, cl, model)
            rr = score_batch(pr, rj, pl, rl, model)
            correct += int((rc > rr).sum().item()); total += (j - i)
        val_acc = correct / max(total, 1)
    print(f"val_acc={val_acc:.4f}  steps={step}  secs={time.time()-start:.0f}")

    # Save
    model_bytes = save_model_compressed(model, "final_model.ptz")
    code_bytes = os.path.getsize(__file__)
    print(f"final_model.ptz: {model_bytes} bytes  train.py: {code_bytes} bytes  total: {model_bytes + code_bytes} bytes")


if __name__ == "__main__":
    main()
