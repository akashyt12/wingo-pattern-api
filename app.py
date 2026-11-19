#!/usr/bin/env python3
"""Stylo AI Pattern-5M - live predictor API (NumPy inference, no torch)."""
from __future__ import annotations

import json
import math
import os
import threading
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from fastapi import FastAPI
from fastapi.responses import JSONResponse

import prepare_pattern as P
from model_pattern import PatternNetNP, softmax

BASE = Path(__file__).resolve().parent
UPSTREAM = os.environ.get("PREDICT_UPSTREAM", "https://wingo-api-two.vercel.app").rstrip("/")
TTL = float(os.environ.get("CACHE_TTL", "3"))
HISTORY_N = int(os.environ.get("HISTORY_N", "500"))
MODEL = PatternNetNP()
META = json.load(open(BASE / "feature_meta.json"))
MU = np.load(BASE / "feat_mu.npy")
SD = np.load(BASE / "feat_sd.npy")
app = FastAPI(title="Stylo AI Pattern-5M", docs_url=None, redoc_url=None)

_lock = threading.Lock()
_cache = {"ts": 0.0, "draws": [], "pred": None}


def fetch_json(url, timeout=8):
    req = urllib.request.Request(url, headers={"User-Agent": "stylo-pattern-api/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def get_draws(force=False):
    now = time.time()
    if not force and _cache["draws"] and now - _cache["ts"] < TTL:
        return _cache["draws"]
    data = fetch_json(f"{UPSTREAM}/history?limit={HISTORY_N}")
    draws = data.get("draws", []) if isinstance(data, dict) else []
    if not draws:
        raise RuntimeError("upstream history empty")
    with _lock:
        _cache["draws"] = draws
        _cache["ts"] = time.time()
        _cache["pred"] = None
    return draws


def records_from_draws(draws):
    n = len(draws)
    digits = np.zeros(n + 1, dtype=np.int16)
    is_big = np.zeros(n + 1, dtype=np.uint8)
    red = np.zeros(n + 1, dtype=np.float32)
    green = np.zeros(n + 1, dtype=np.float32)
    violet = np.zeros(n + 1, dtype=np.float32)
    for i, d in enumerate(draws):
        num = int(d["number"])
        digits[i] = num
        is_big[i] = 1 if num >= 5 else 0
        cols = d.get("colours") or []
        red[i] = 1.0 if "RED" in cols else 0.0
        green[i] = 1.0 if "GREEN" in cols else 0.0
        violet[i] = 1.0 if "VIOLET" in cols else 0.0
    digits[n] = 0
    return digits, is_big, red, green, violet


def run_model(draws, force=False):
    now = time.time()
    if not force and _cache["pred"] and now - _cache["ts"] < TTL:
        return _cache["pred"]
    recs = records_from_draws(draws)
    X, _ = P.build(records=recs, save=False)
    x = ((X[-1:] - MU) / SD).astype(np.float32)
    size_p, col_p = MODEL.predict_proba(x)
    p_big = float(size_p[0][1])
    p_red, p_green, p_violet = [float(v) for v in col_p[0]]

    primary = ["RED", "GREEN", "VIOLET"][int(np.argmax(col_p[0]))]
    colours = [c for c, p in (("RED", p_red), ("GREEN", p_green), ("VIOLET", p_violet)) if p > 0.5]
    if not colours:
        colours = [primary]

    last = draws[-1]
    out = {
        "status": "ok",
        "period": str(int(last["period"]) + 1),
        "prediction": {
            "size": "BIG" if p_big > 0.5 else "SMALL",
            "big_small": "BIG" if p_big > 0.5 else "SMALL",
            "colour": primary,
            "colours": colours,
            "violet_alert": p_violet > 0.5,
        },
        "last_pick": {
            "period": str(last["period"]),
            "number": int(last["number"]),
            "size": str(last.get("size", "")),
            "colour": ",".join(last.get("colours") or []),
            "actual_number": int(last["number"]),
            "actual_size": str(last.get("size", "")),
            "actual_colour": ",".join(last.get("colours") or []),
        },
        "session": {
            "total": 0, "size_wins": 0, "size_losses": 0,
            "col_wins": 0, "col_losses": 0, "size_acc": None, "col_acc": None, "recent": [],
        },
        "probabilities": {
            "big": round(p_big, 6),
            "small": round(1.0 - p_big, 6),
            "red": round(p_red, 6),
            "green": round(p_green, 6),
            "violet": round(p_violet, 6),
            "size_z": round((p_big - 0.5) / 0.007963, 4),
        },
        "latest_result": {
            "period": str(last["period"]),
            "number": int(last["number"]),
            "size": str(last.get("size", "")),
            "colour": ",".join(last.get("colours") or []),
        },
        "rule_engine": {"pred": None, "confidence": 0.0, "votes": []},
        "rule_engine_status": "PatternNet-5M (w=3/7/9 exact match, next-10 majority)",
        "model": {
            "backend": "numpy-native (no torch required)",
            "features": int(META["n_features"]),
            "params": int(META["params"]),
            "version": "pattern-5m",
        },
        "upstream_records": len(draws),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    with _lock:
        _cache["pred"] = out
        _cache["ts"] = time.time()
    return out


@app.get("/")
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_backend": "numpy-native (no torch required)",
        "features": int(META["n_features"]),
        "params": int(META["params"]),
        "version": "pattern-5m",
        "upstream": UPSTREAM,
        "endpoints": ["/health", "/predict", "/predict?fresh=true", "/history?limit=50"],
    }


@app.get("/predict")
def predict(fresh: bool = False):
    try:
        return run_model(get_draws(force=fresh), force=fresh)
    except Exception as e:
        return JSONResponse(status_code=502, content={"error": "prediction service unavailable", "detail": str(e)[:200]})


@app.get("/history")
def history(limit: int = 50):
    try:
        draws = get_draws()
        return {"count": min(max(1, limit), len(draws)), "draws": draws[-max(1, limit):]}
    except Exception as e:
        return JSONResponse(status_code=502, content={"error": str(e)[:200]})
