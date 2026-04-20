"""
PetiteLLM: Preference — reward model training script.

Small transformer encoder scores (prompt, response) pairs. Trained with pairwise
Bradley-Terry loss on UltraFeedback pairs. Saves fp16 + zlib compressed artifact.

Required eval entry points:
    build_and_load(model_path: str) -> nn.Module
    score_batch(prompt_ids, response_ids, prompt_lens, response_lens, model) -> Tensor
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
    model_dim = int(os.environ.get("MODEL_DIM", 288))
    num_layers = int(os.environ.get("NUM_LAYERS", 6))
    num_heads = int(os.environ.get("NUM_HEADS", 4))
    mlp_mult = int(os.environ.get("MLP_MULT", 4))
    dropout = float(os.environ.get("DROPOUT", 0.0))
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
    train_frac = float(os.environ.get("TRAIN_FRAC", 0.99))  # reserve ~1% for save/val
    bt_margin = float(os.environ.get("BT_MARGIN", 0.0))
    anchor_w = float(os.environ.get("ANCHOR_W", 0.0))

    # Token ids populated from tokenizer.json at runtime.
    pad_id = 0
    resp_sep_id = 5
    end_id = 6


# ----------------------------- model -----------------------------


class RewardModel(nn.Module):
    def __init__(self, vocab, dim, num_layers, num_heads, mlp_mult, max_seq_len, pad_id, dropout=0.0):
        super().__init__()
        self.pad_id = pad_id
        self.embed = nn.Embedding(vocab, dim, padding_idx=pad_id)
        self.pos = nn.Embedding(max_seq_len, dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=dim * mlp_mult,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.ln = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, 1)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """ids: (B, L) int64. Returns (B,) scalar reward."""
        B, L = ids.shape
        pad_mask = ids.eq(self.pad_id)
        pos = torch.arange(L, device=ids.device).unsqueeze(0).expand(B, L)
        h = self.embed(ids) + self.pos(pos)
        h = self.encoder(h, src_key_padding_mask=pad_mask)
        h = self.ln(h)
        # Mean-pool over non-pad tokens.
        mask = (~pad_mask).unsqueeze(-1).to(h.dtype)
        denom = mask.sum(dim=1).clamp_min(1.0)
        pooled = (h * mask).sum(dim=1) / denom
        return self.head(pooled).squeeze(-1)


# ----------------------------- packing (vectorized) -----------------------------


def pack_prompt_response(
    prompt: torch.Tensor, resp: torch.Tensor, prompt_len: torch.Tensor, resp_len: torch.Tensor,
    resp_sep: int, end_id: int, pad_id: int, max_total: int,
) -> torch.Tensor:
    """Builds [prompt[:pl], resp_sep, resp[:rl], end_id, pad...] as (B, max_total).

    Fully vectorized — no Python loop. Uses torch.where + gather.
    """
    B, P = prompt.shape
    _, R = resp.shape
    device = prompt.device

    # Truncate response to fit within max_total after prompt + 2 specials.
    r_eff = torch.minimum(resp_len, (max_total - prompt_len - 2).clamp_min(0))

    pos = torch.arange(max_total, device=device, dtype=prompt.dtype).unsqueeze(0).expand(B, -1)
    p_len_b = prompt_len.unsqueeze(1)
    r_eff_b = r_eff.unsqueeze(1)

    is_prompt = pos < p_len_b
    is_sep = pos == p_len_b
    resp_offset = pos - p_len_b - 1  # may be negative
    is_resp = (resp_offset >= 0) & (resp_offset < r_eff_b)
    is_end = pos == (p_len_b + 1 + r_eff_b)

    prompt_gather = prompt.gather(1, pos.clamp(0, P - 1))
    resp_gather = resp.gather(1, resp_offset.clamp(0, R - 1))

    out = torch.full((B, max_total), pad_id, dtype=prompt.dtype, device=device)
    out = torch.where(is_prompt, prompt_gather, out)
    out = torch.where(is_sep, torch.full_like(out, resp_sep), out)
    out = torch.where(is_resp, resp_gather, out)
    out = torch.where(is_end, torch.full_like(out, end_id), out)
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


def _populate_special_ids():
    try:
        tok_meta = json.load(open("data/tokenizer.json"))
        for t in tok_meta.get("added_tokens", []):
            content = t.get("content")
            tid = t.get("id")
            if content == "<pad>" and isinstance(tid, int): HP.pad_id = tid
            elif content == "<resp>" and isinstance(tid, int): HP.resp_sep_id = tid
            elif content == "<end>" and isinstance(tid, int): HP.end_id = tid
        vocab_sz = max(HP.vocab_size, len(tok_meta.get("model", {}).get("vocab", {})) or HP.vocab_size)
        return vocab_sz
    except Exception:
        return HP.vocab_size


def build_and_load(model_path: str):
    """Instantiate architecture, decompress + load fp16 state dict."""
    vocab_sz = _populate_special_ids()

    model = RewardModel(
        vocab=vocab_sz, dim=HP.model_dim, num_layers=HP.num_layers,
        num_heads=HP.num_heads, mlp_mult=HP.mlp_mult,
        max_seq_len=HP.max_seq_len, pad_id=HP.pad_id,
        dropout=0.0,  # no dropout at eval
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


def pairwise_loss(r_chosen, r_rejected, margin: float = 0.0, aux_weight: float = 0.0):
    """Bradley-Terry + optional small anchor term that keeps rewards bounded."""
    loss = bradley_terry_loss(r_chosen, r_rejected, margin=margin)
    if aux_weight > 0:
        # Light L2 anchor on reward magnitudes — prevents runaway scores.
        loss = loss + aux_weight * 0.5 * (r_chosen.pow(2).mean() + r_rejected.pow(2).mean())
    return loss


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
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    vocab_sz = _populate_special_ids()
    print(f"vocab_size={vocab_sz} pad={HP.pad_id} resp={HP.resp_sep_id} end={HP.end_id}")

    train_ds = PairDataset("data/train_pairs.npz")
    val_ds = PairDataset("data/val_pairs.npz")
    print(f"train n={train_ds.n}  val n={val_ds.n}")

    model = RewardModel(
        vocab=vocab_sz, dim=HP.model_dim, num_layers=HP.num_layers,
        num_heads=HP.num_heads, mlp_mult=HP.mlp_mult,
        max_seq_len=HP.max_seq_len, pad_id=HP.pad_id,
        dropout=HP.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"reward model params: {n_params/1e6:.2f}M  bs={HP.batch_size} lr={HP.lr} dropout={HP.dropout}")

    opt = torch.optim.AdamW(
        model.parameters(), lr=HP.lr, betas=(HP.beta1, HP.beta2),
        weight_decay=HP.weight_decay,
    )

    train_seconds = HP.max_wallclock_seconds * HP.train_frac

    # Linear warmup by step count, then constant lr (matches baseline recipe).
    def lr_at(step, elapsed):
        if step < HP.warmup_steps:
            return HP.lr * (step + 1) / HP.warmup_steps
        return HP.lr

    start = time.time()
    step = 0
    last_log = start
    model.train()
    rng = np.random.default_rng(HP.seed)
    B = HP.batch_size

    while True:
        elapsed = time.time() - start
        if elapsed >= train_seconds:
            break
        cur_lr = lr_at(step, elapsed)
        for g in opt.param_groups: g["lr"] = cur_lr

        idxs = rng.integers(0, train_ds.n, size=B)
        idxs_t = torch.from_numpy(idxs).long()
        pr, ch, rj, pl, cl, rl = train_ds.batch(idxs_t, device)

        packed_c = pack_prompt_response(
            pr, ch, pl, cl, HP.resp_sep_id, HP.end_id, HP.pad_id, HP.max_seq_len
        )
        packed_r = pack_prompt_response(
            pr, rj, pl, rl, HP.resp_sep_id, HP.end_id, HP.pad_id, HP.max_seq_len
        )
        both = torch.cat([packed_c, packed_r], dim=0)  # (2B, L)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            r = model(both).float()
        rc, rr = r[:B], r[B:]

        loss = pairwise_loss(rc, rr, margin=HP.bt_margin, aux_weight=HP.anchor_w)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if HP.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), HP.grad_clip)
        opt.step()
        step += 1

        now = time.time()
        if now - last_log >= 10.0:
            acc = (rc > rr).float().mean().item()
            print(f"step={step} t={elapsed:.0f}s lr={cur_lr:.1e} loss={loss.item():.4f} train_batch_acc={acc:.3f}", flush=True)
            last_log = now

    print(f"training done: step={step} elapsed={time.time()-start:.1f}s")

    # Quick val
    model.eval()
    with torch.inference_mode():
        total = 0; correct = 0
        B_eval = 128
        for i in range(0, val_ds.n, B_eval):
            j = min(i + B_eval, val_ds.n)
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
