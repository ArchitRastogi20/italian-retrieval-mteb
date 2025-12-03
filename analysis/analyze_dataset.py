#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Ultra-fast EDA for ArchitRastogi/it-retrieval-triplets-mc4 (v2.1)

Fixes:
- Polars fast-path: use positive infer_schema_length, safer casting.
- tqdm responsiveness: smaller adaptive batches (32â€“128 MB).
- Sensible worker default + env override EDA_WORKERS.
- Clear runtime banner with workers & target batch size.

Outputs:
- figs/*.png
- EDA_report.md
"""

from __future__ import annotations
import os, re, time, json, psutil, subprocess
from pathlib import Path
from typing import Dict, List
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

# ----------------------------
# Config
# ----------------------------
DATASET_REPO = "ArchitRastogi/it-retrieval-triplets-mc4"
FILES = ["train.jsonl", "train_random_neg.jsonl"]
BASE_HF_URL = f"https://huggingface.co/datasets/{DATASET_REPO}/resolve/main"

OUT_DIR = Path("./it_mc4_eda_fast")
DATA_DIR = OUT_DIR / "data"
FIG_DIR = OUT_DIR / "figs"
REPORT_MD = OUT_DIR / "EDA_report.md"

CHAR_CAP = 1024
TOKEN_CAP = 256
BINS_CHAR = np.arange(0, CHAR_CAP + 1, dtype=np.int64)
BINS_TOKEN = np.arange(0, TOKEN_CAP + 1, dtype=np.int64)

CPU_COUNT = os.cpu_count() or 32
# Default: leave a couple for OS, cap to avoid oversubscription; env override allowed.
WORKERS = int(os.environ.get("EDA_WORKERS", str(max(4, min(40, CPU_COUNT - 2)))))


VM = psutil.virtual_memory()
TOTAL_RAM = VM.total
TARGET_RAM_FRAC = 0.95
SAFETY_BYTES = 1_000_000_000  # ~1 GB
TARGET_BYTES = int(TOTAL_RAM * TARGET_RAM_FRAC)

WS_RE = re.compile(r"\s+")

# JSON loader
try:
    import orjson as fastjson
    def loads(b: bytes | str): return fastjson.loads(b)
    JSON_IS_BYTES = True
except Exception:
    def loads(s: str): return json.loads(s)
    JSON_IS_BYTES = False

# Optional Polars
try:
    import polars as pl
    HAS_POLARS = True
except Exception:
    HAS_POLARS = False

# ----------------------------
# Download
# ----------------------------
def download_with_wget(url: str, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["wget", "-q", "--show-progress", "-c", "-O", str(out_path), url]
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)

# ----------------------------
# Histogram helpers
# ----------------------------
def update_histogram(vec: np.ndarray, value: int, cap: int):
    if value < 0: value = 0
    elif value > cap: value = cap
    vec[value] += 1

def percentile_from_hist(hist: np.ndarray, bins: np.ndarray, q: float) -> float:
    cdf = np.cumsum(hist)
    if cdf[-1] == 0: return 0.0
    target = q * cdf[-1]
    idx = int(np.searchsorted(cdf, target, side="left"))
    return float(bins[min(idx, len(bins)-1)])

def mean_from_hist(hist: np.ndarray, bins: np.ndarray) -> float:
    tot = hist.sum()
    return 0.0 if tot == 0 else float((hist * bins).sum() / tot)

# ----------------------------
# Worker
# ----------------------------
def process_lines(lines: List[bytes] | List[str]) -> Dict:
    char_hist = {k: np.zeros_like(BINS_CHAR, dtype=np.int64) for k in ["query","positive","negative"]}
    tok_hist  = {k: np.zeros_like(BINS_TOKEN, dtype=np.int64) for k in ["query","positive","negative"]}
    missing = {"query": 0, "positive": 0, "negative": 0}
    bad_json = 0
    total = 0

    for raw in lines:
        if not raw: continue
        try:
            obj = loads(raw)
        except Exception:
            bad_json += 1; continue

        total += 1
        q, p, n = obj.get("query"), obj.get("positive"), obj.get("negative")
        for k, v in (("query", q), ("positive", p), ("negative", n)):
            if not isinstance(v, str) or not v: missing[k] += 1

        if isinstance(q, str) and isinstance(p, str) and isinstance(n, str):
            update_histogram(char_hist["query"], len(q), CHAR_CAP)
            update_histogram(char_hist["positive"], len(p), CHAR_CAP)
            update_histogram(char_hist["negative"], len(n), CHAR_CAP)
            q,p,n = q.strip(), p.strip(), n.strip()
            update_histogram(tok_hist["query"], len(WS_RE.split(q)) if q else 0, TOKEN_CAP)
            update_histogram(tok_hist["positive"], len(WS_RE.split(p)) if p else 0, TOKEN_CAP)
            update_histogram(tok_hist["negative"], len(WS_RE.split(n)) if n else 0, TOKEN_CAP)

    return {"total": total, "bad_json": bad_json, "missing": missing,
            "char_hist": char_hist, "tok_hist": tok_hist}

# ----------------------------
# Adaptive reader (32â€“128MB batches) with byte progress
# ----------------------------
MIN_BATCH = 32 * 1024 * 1024     # 32 MB
MAX_BATCH = 128 * 1024 * 1024    # 128 MB

def blocks_from_file(path: Path):
    """
    Yields (batch, bytes_in_batch). Smaller batches improve responsiveness.
    """
    bytes_accum = 0
    batch: List[bytes] | List[str] = []
    bufsize = 64 * 1024 * 1024    # buffered read

    mode = "rb" if JSON_IS_BYTES else "r"
    with open(path, mode, buffering=bufsize) as f:
        for line in f:
            sz = len(line)
            batch.append(line)
            bytes_accum += sz

            vm = psutil.virtual_memory()
            free_headroom = TARGET_BYTES - vm.used - SAFETY_BYTES
            # aim for ~half of safe headroom but clamp to [MIN_BATCH, MAX_BATCH]
            bytes_target = max(MIN_BATCH, min(MAX_BATCH, int(free_headroom * 0.5)))
            if bytes_accum >= bytes_target:
                yield batch, bytes_accum
                batch, bytes_accum = [], 0
        if batch:
            yield batch, bytes_accum

# ----------------------------
# Aggregate
# ----------------------------
def aggregate_results(parts: List[Dict]) -> Dict:
    char_hist = {k: np.zeros_like(BINS_CHAR, dtype=np.int64) for k in ["query","positive","negative"]}
    tok_hist  = {k: np.zeros_like(BINS_TOKEN, dtype=np.int64) for k in ["query","positive","negative"]}
    total = bad_json = 0
    missing = {"query": 0, "positive": 0, "negative": 0}
    for r in parts:
        total += r["total"]; bad_json += r["bad_json"]
        for k in missing: missing[k] += r["missing"][k]
        for fld in char_hist:
            char_hist[fld] += r["char_hist"][fld]; tok_hist[fld] += r["tok_hist"][fld]
    return {"total": total, "bad_json": bad_json, "missing": missing,
            "char_hist": char_hist, "tok_hist": tok_hist}

# ----------------------------
# Plotting
# ----------------------------
def plot_overlaid(hist_map: Dict[str, np.ndarray], bins: np.ndarray, title: str, out_png: Path, xlabel: str):
    plt.figure(figsize=(10, 5.2))
    for fld in ["query", "positive", "negative"]:
        counts = hist_map[fld].astype(np.float64)
        factor = 4
        L = (len(counts) // factor) * factor
        if L == 0: continue
        ds = counts[:L].reshape(-1, factor).sum(axis=1)
        xb = bins[:L].reshape(-1, factor).mean(axis=1)
        plt.plot(xb, ds, label=fld, linewidth=1.6)
    plt.title(title); plt.xlabel(xlabel); plt.ylabel("frequency"); plt.legend()
    plt.tight_layout(); plt.savefig(out_png, dpi=160); plt.close()

# ----------------------------
# Reporting
# ----------------------------
def stats_from_hists(char_hist, tok_hist) -> Dict[str, Dict[str, float]]:
    out = {}
    for fld in ["query","positive","negative"]:
        ch, th = char_hist[fld], tok_hist[fld]
        out[fld] = {
            "count": int(ch.sum()),
            "char_mean": mean_from_hist(ch, BINS_CHAR),
            "char_p50": percentile_from_hist(ch, BINS_CHAR, 0.50),
            "char_p90": percentile_from_hist(ch, BINS_CHAR, 0.90),
            "char_p95": percentile_from_hist(ch, BINS_CHAR, 0.95),
            "char_p99": percentile_from_hist(ch, BINS_CHAR, 0.99),
            "tok_mean": mean_from_hist(th, BINS_TOKEN),
            "tok_p50": percentile_from_hist(th, BINS_TOKEN, 0.50),
            "tok_p90": percentile_from_hist(th, BINS_TOKEN, 0.90),
            "tok_p95": percentile_from_hist(th, BINS_TOKEN, 0.95),
            "tok_p99": percentile_from_hist(th, BINS_TOKEN, 0.99),
        }
    return out

def write_report(entries: List[Dict]):
    lines = [f"# EDA: {DATASET_REPO}\n"]
    for e in entries:
        lines.append(f"## File: {e['label']}")
        lines.append(f"- Path: `{e['file']}`")
        lines.append(f"- Total lines: **{e['total']:,}** | Bad JSON: **{e['bad_json']:,}**")
        lines.append(f"- Missing fields: {e['missing']}\n")
        for fld in ["query","positive","negative"]:
            s = e["stats"][fld]
            lines.append(f"**{fld}**")
            lines.append(f"- chars: mean={s['char_mean']:.1f}, p50={s['char_p50']:.0f}, "
                         f"p90={s['char_p90']:.0f}, p95={s['char_p95']:.0f}, p99={s['char_p99']:.0f}")
            lines.append(f"- tokens: mean={s['tok_mean']:.1f}, p50={s['tok_p50']:.0f}, "
                         f"p90={s['tok_p90']:.0f}, p95={s['tok_p95']:.0f}, p99={s['tok_p99']:.0f}\n")

    lines.append("\n---\n## Takeaway: Are queries shorter than passages?\n")
    for e in entries:
        q50 = e["stats"]["query"]["tok_p50"]; p50 = e["stats"]["positive"]["tok_p50"]
        basis = "tokens"
        if q50 == p50:
            q50 = e["stats"]["query"]["char_p50"]; p50 = e["stats"]["positive"]["char_p50"]; basis = "characters"
        verdict = "shorter" if q50 < p50 else ("longer" if q50 > p50 else "about the same length")
        lines.append(f"- **{e['label']}**: queries are **{verdict}** than positives "
                     f"(median {basis}: query={q50:.0f}, positive={p50:.0f}).")

    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] Report â†’ {REPORT_MD}")

# ----------------------------
# Polars fast-path (fixed)
# ----------------------------
def polars_fast_hist(path: Path, label: str) -> Dict:
    assert HAS_POLARS, "polars not available"
    t0 = time.time()
    scan = (
        pl.scan_ndjson(str(path), infer_schema_length=1024)  # must be positive
        .select([
            pl.col("query").cast(pl.Utf8, strict=False),
            pl.col("positive").cast(pl.Utf8, strict=False),
            pl.col("negative").cast(pl.Utf8, strict=False),
        ])
    )

    def mk(col):
        df = scan.select([
            pl.col(col).alias("txt"),
            pl.col(col).str.len_chars().clip_max(CHAR_CAP).fill_null(0).alias("len_char"),
            pl.when(pl.col(col).str.strip().eq(""))
              .then(0)
              .otherwise(pl.col(col).str.count_matches(r"\s+") + 1)
              .clip_max(TOKEN_CAP)
              .fill_null(0)
              .alias("len_tok"),
        ])
        char_counts = (df.group_by("len_char").len()
                         .select([pl.col("len_char").alias("k"), pl.col("len").alias("v")])
                         .collect(streaming=True))
        tok_counts  = (df.group_by("len_tok").len()
                         .select([pl.col("len_tok").alias("k"), pl.col("len").alias("v")])
                         .collect(streaming=True))
        ch = np.zeros_like(BINS_CHAR, dtype=np.int64); th = np.zeros_like(BINS_TOKEN, dtype=np.int64)
        k, v = np.clip(char_counts["k"].to_numpy(), 0, CHAR_CAP), char_counts["v"].to_numpy(); ch[k] += v
        k, v = np.clip(tok_counts["k"].to_numpy(), 0, TOKEN_CAP), tok_counts["v"].to_numpy(); th[k] += v
        return ch, th

    ch_q, th_q = mk("query"); ch_p, th_p = mk("positive"); ch_n, th_n = mk("negative")
    total = int(max(ch_q.sum(), ch_p.sum(), ch_n.sum()))
    agg = {"total": total, "bad_json": 0, "missing": {"query": 0, "positive": 0, "negative": 0},
           "char_hist": {"query": ch_q, "positive": ch_p, "negative": ch_n},
           "tok_hist": {"query": th_q, "positive": th_p, "negative": th_n}}

    print(f"[EDA|polars] {label}: done in {(time.time()-t0)/60:.1f} min â€” linesâ‰ˆ{total:,}")
    stats = stats_from_hists(agg["char_hist"], agg["tok_hist"])
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plot_overlaid(agg["char_hist"], BINS_CHAR, f"[{label}] Char-length distribution",
                  FIG_DIR / f"{label}_chars.png", "characters (clipped at 1024)")
    plot_overlaid(agg["tok_hist"], BINS_TOKEN, f"[{label}] Token-length distribution",
                  FIG_DIR / f"{label}_tokens.png", "whitespace tokens (clipped at 256)")
    return {"label": label, "file": str(path), "total": total, "bad_json": 0,
            "missing": {"query": 0, "positive": 0, "negative": 0}, "stats": stats}

# ----------------------------
# MP path with responsive tqdm
# ----------------------------
def run_eda_for_file(path: Path, label: str) -> Dict:
    if HAS_POLARS:
        try:
            return polars_fast_hist(path, label)
        except Exception as e:
            print(f"[WARN] Polars fast-path failed ({e}); falling back to multiprocessing.")

    file_size = path.stat().st_size
    # Small batches + moderate inflight â†’ early progress and steady throughput
    max_inflight = WORKERS * 2
    inflight = []
    parts: List[Dict] = []
    t0 = time.time()

    print(f"[EDA] {label}: workers={WORKERS}, batchâ‰ˆ{MIN_BATCH//(1024*1024)}â€“{MAX_BATCH//(1024*1024)}MB, "
          f"RAM target â‰ˆ {int(TARGET_RAM_FRAC*100)}%")
    pbar = tqdm(total=file_size, unit="B", unit_scale=True, desc=f"Processing {label}")

    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        for block, bytes_in_batch in blocks_from_file(path):
            fut = pool.submit(process_lines, block)
            inflight.append((fut, bytes_in_batch))

            # drain as they complete or when queue is full
            i = 0
            while i < len(inflight):
                fut_i, bytes_i = inflight[i]
                if fut_i.done():
                    parts.append(fut_i.result())
                    pbar.update(bytes_i)
                    inflight.pop(i)
                else:
                    i += 1
            while len(inflight) >= max_inflight:
                time.sleep(0.02)
                i = 0
                while i < len(inflight):
                    fut_i, bytes_i = inflight[i]
                    if fut_i.done():
                        parts.append(fut_i.result()); pbar.update(bytes_i); inflight.pop(i)
                    else:
                        i += 1

        # drain remaining
        for fut, bytes_i in inflight:
            parts.append(fut.result()); pbar.update(bytes_i)

    pbar.close()
    agg = aggregate_results(parts)
    print(f"[EDA] {label}: done in {(time.time()-t0)/60:.1f} min â€” "
          f"lines={agg['total']:,}, bad_json={agg['bad_json']:,}")

    stats = stats_from_hists(agg["char_hist"], agg["tok_hist"])
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plot_overlaid(agg["char_hist"], BINS_CHAR, f"[{label}] Char-length distribution",
                  FIG_DIR / f"{label}_chars.png", "characters (clipped at 1024)")
    plot_overlaid(agg["tok_hist"], BINS_TOKEN, f"[{label}] Token-length distribution",
                  FIG_DIR / f"{label}_tokens.png", "whitespace tokens (clipped at 256)")

    return {"label": label, "file": str(path), "total": agg["total"], "bad_json": agg["bad_json"],
            "missing": agg["missing"], "stats": stats}

# ----------------------------
# Entrypoint
# ----------------------------
def main():
    print(f"[SETUP] OUT_DIR: {OUT_DIR.resolve()}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for fname in FILES:
        url = f"{BASE_HF_URL}/{fname}?download=true"
        dest = DATA_DIR / fname
        if not dest.exists():
            download_with_wget(url, dest)
        else:
            print(f"[SKIP] {dest} exists")

    results = []
    for fname in FILES:
        path = DATA_DIR / fname; label = Path(fname).stem
        results.append(run_eda_for_file(path, label))

    write_report(results)
    for r in results:
        q50 = r["stats"]["query"]["tok_p50"]; p50 = r["stats"]["positive"]["tok_p50"]
        print(f"[SUMMARY] {r['label']}: median tokens â†’ query={q50:.0f}, positive={p50:.0f}")

if __name__ == "__main__":
    main()
