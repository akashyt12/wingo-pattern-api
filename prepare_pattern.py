#!/usr/bin/env python3
"""Pattern-matching features: exact B/S pattern history (w=3,7,9) + next-10 outcomes.

For every row i we look at the last w size-bits (B=digit>=5, S=digit<5), find all
historical occurrences of that exact pattern before i, and summarise what happened
in the following 1..10 draws. Everything is strictly causal (t+k <= i), no leakage.
"""
from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

WARMUP = 250
WS = (3, 7, 9)
KS = tuple(range(1, 11))

CANDIDATES = [
    os.environ.get("WINGO_SRC", ""),
    "/kaggle/input/wingo-raw-draws/big_dataset.json",
    r"C:\Users\Welcome\Downloads\Telegram Desktop\big_dataset (2).json",
]
SRC = next((Path(p) for p in CANDIDATES if p and Path(p).exists()), Path(CANDIDATES[-1]))
OUT = Path(os.environ.get("WINGO_OUT", "/kaggle/working/data_pattern"))
if not OUT.drive and os.name == "nt":
    OUT = Path(r"C:\Users\Welcome\ai-research\wingo_ai\data_pattern")


def load_records():
    d = json.load(open(SRC, encoding="utf-8"))
    recs = list(reversed(d["records"]))
    n = len(recs)
    digits = np.array([int(r["number"]) for r in recs], dtype=np.int16)
    is_big = (digits >= 5).astype(np.uint8)
    red = np.zeros(n, dtype=np.float32)
    green = np.zeros(n, dtype=np.float32)
    violet = np.zeros(n, dtype=np.float32)
    for i, r in enumerate(recs):
        c = r["colour"]
        red[i] = 1.0 if "RED" in c else 0.0
        green[i] = 1.0 if "GREEN" in c else 0.0
        violet[i] = 1.0 if "VIOLET" in c else 0.0
    return digits, is_big, red, green, violet


def _build_stats(bits, w):
    ts_by_pat = defaultdict(list)
    for t in range(w, len(bits)):
        ts_by_pat[bytes(bits[t - w:t])].append(t)
    stats = {}
    for pat, ts in ts_by_pat.items():
        ts = np.asarray(ts, dtype=np.int32)
        cums = []
        for k in KS:
            valid = ts + k - 1 < len(bits)
            tk = ts[valid]
            vals = bits[tk + k - 1].astype(np.float64)
            cums.append((tk, np.concatenate([[0.0], np.cumsum(vals)])))
        stats[pat] = cums
    return stats


def make_features(is_big):
    bits = np.asarray(is_big, dtype=np.uint8)
    n = len(bits)
    names = []
    onehots = {}
    for w in WS:
        onehots[w] = np.zeros((n, 1 << w), dtype=np.float32)
        for b in range(1 << w):
            names.append(f"pat{w}_onehot_{b:0{w}b}")
    stat_names = []
    for w in WS:
        for k in KS:
            stat_names.append(f"p{w}_next{k}_big")
        stat_names += [
            f"p{w}_cnt", f"p{w}_cnt10", f"p{w}_big10_rate",
            f"p{w}_major10", f"p{w}_conf10", f"p{w}_vote", f"p{w}_vote_conf",
        ]
    names += stat_names
    names += ["cross_vote_big", "cross_vote_conf", "cross_vote10_big", "cross_vote10_conf", "cross_all_agree"]
    X = np.zeros((n, len(names)), dtype=np.float32)

    all_stats = {w: _build_stats(bits, w) for w in WS}
    col = {name: j for j, name in enumerate(names)}

    for w in WS:
        stats = all_stats[w]
        for i in range(n):
            key = bytes(bits[i - w:i]) if i >= w else None
            if key is None:
                continue
            pidx = 0
            for bit in key:
                pidx = (pidx << 1) | int(bit)
            onehots[w][i, pidx] = 1.0
            cums = stats.get(key)
            if not cums:
                continue
            rates = np.empty(len(KS), dtype=np.float64)
            for k_idx, (tk, cum) in enumerate(cums):
                pos = int(np.searchsorted(tk, i - (k_idx + 1), side="right"))
                rates[k_idx] = (cum[pos] / pos) if pos > 0 else 0.5
                X[i, col[f"p{w}_next{k_idx + 1}_big"]] = rates[k_idx]
            cnt1 = int(np.searchsorted(cums[0][0], i - 1, side="right"))
            cnt10 = int(np.searchsorted(cums[9][0], i - 10, side="right"))
            big10 = float(rates[9])
            X[i, col[f"p{w}_cnt"]] = math.log1p(cnt1)
            X[i, col[f"p{w}_cnt10"]] = math.log1p(cnt10)
            X[i, col[f"p{w}_big10_rate"]] = big10
            X[i, col[f"p{w}_major10"]] = 1.0 if big10 > 0.5 else 0.0
            X[i, col[f"p{w}_conf10"]] = abs(big10 - 0.5) * 2.0
            X[i, col[f"p{w}_vote"]] = 1.0 if rates[0] > 0.5 else 0.0
            X[i, col[f"p{w}_vote_conf"]] = abs(rates[0] - 0.5) * 2.0

    for i in range(n):
        votes = [X[i, col[f"p{w}_vote"]] for w in WS]
        confs = [X[i, col[f"p{w}_vote_conf"]] for w in WS]
        votes10 = [X[i, col[f"p{w}_major10"]] for w in WS]
        confs10 = [X[i, col[f"p{w}_conf10"]] for w in WS]
        X[i, col["cross_vote_big"]] = sum(votes)
        X[i, col["cross_vote_conf"]] = sum(confs) / 3.0
        X[i, col["cross_vote10_big"]] = sum(votes10)
        X[i, col["cross_vote10_conf"]] = sum(confs10) / 3.0
        X[i, col["cross_all_agree"]] = 1.0 if len(set(votes)) == 1 else 0.0

    off = 0
    for w in WS:
        dim = 1 << w
        X[:, off:off + dim] = onehots[w]
        off += dim
    if WARMUP:
        X[:WARMUP] = 0.0
    return X, names


def build(records=None, save=True):
    if records is None:
        digits, is_big, red, green, violet = load_records()
    else:
        digits, is_big, red, green, violet = records
    X, names = make_features(is_big)
    if not save:
        return X, names
    y_size = (digits >= 5).astype(np.int64)
    y_colour = np.where(violet > 0, 2, np.where(red > 0, 0, 1)).astype(np.int64)
    n = len(X)
    ntr = int(n * 0.8)
    nva = int(n * 0.1)
    idx = {
        "train": np.arange(WARMUP, ntr),
        "val": np.arange(ntr, ntr + nva),
        "test": np.arange(ntr + nva, n),
    }
    mu = X[idx["train"]].mean(axis=0).astype(np.float32)
    sd = X[idx["train"]].std(axis=0).astype(np.float32)
    sd[sd < 1e-6] = 1.0
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT / "data.npz",
        X=X, y_size=y_size, y_colour=y_colour,
        tr=idx["train"], va=idx["val"], te=idx["test"],
        mu=mu, sd=sd,
    )
    meta = {
        "n_features": int(X.shape[1]),
        "features": names,
        "warmup": WARMUP,
        "windows": list(WS),
        "next_k": list(KS),
        "rows": int(n),
        "train": int(len(idx["train"])),
        "val": int(len(idx["val"])),
        "test": int(len(idx["test"])),
    }
    with open(OUT / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[pattern] rows={n} feats={X.shape[1]} train={len(idx['train'])} val={len(idx['val'])} test={len(idx['test'])}")
    return X, y_size, y_colour, names


if __name__ == "__main__":
    build()
