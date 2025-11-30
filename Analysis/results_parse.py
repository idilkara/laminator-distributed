#!/usr/bin/env python3
"""
Parse "CENSUS-S Scalability Tests.txt" and compute means/stds for each
(epochs, workers, mode ∈ {baseline, attested}).

Attested total = handshake + preprocess + total_training + report_generation.
Validation is parsed separately in the input but NOT included in that total.

Outputs:
  - aggregated CSVs and JSON
  - console summaries
"""

import re, json
from pathlib import Path
from collections import defaultdict
import pandas as pd

LOG_PATH = Path("/home/gavin/laminator-distributed/Analysis/CENSUS-S Scalability Tests.txt")  # adjust if needed

# ---------- Regexes ----------
SEC_HDR = re.compile(
    r"(?P<label>(Baseline|With Attestation))\s*\(CENSUS-S,\s*(?P<epochs>\d+)\s*epochs,\s*(?P<workers>\d+)\s*workers\)",
    re.IGNORECASE,
)

TIMING_BASELINE = re.compile(
    r"Timing summary:\s*preprocess=(?P<preprocess>[0-9.]+)s\s*\|\s*total_training=(?P<train>[0-9.]+)s\s*\|\s*avg_epoch=(?P<avg_epoch>[0-9.]+)s",
    re.IGNORECASE,
)

TIMING_ATTEST = re.compile(
    r"Timing summary:\s*handshake=(?P<handshake>[0-9.]+)s\s*\|\s*preprocess=(?P<preprocess>[0-9.]+)s\s*\|\s*total_training=(?P<train>[0-9.]+)s\s*\|\s*avg_epoch=(?P<avg_epoch>[0-9.]+)s\s*\|\s*report_generation=(?P<report>[0-9.]+)s",
    re.IGNORECASE,
)

SIZES_TRIPLE = re.compile(
    r"hash_report\.txt size \(bytes\):\s*(?P<report>\d+).*?"
    r"hash_report\.txt\.sig size \(bytes\):\s*(?P<sig>\d+).*?"
    r"final_weights\.json size \(bytes\):\s*(?P<weights>\d+)",
    re.IGNORECASE | re.DOTALL,
    )

# ---------- Parse ----------
text = LOG_PATH.read_text(encoding="utf-8", errors="replace")
lines = text.splitlines()

rows = []
memory_overhead = {}  # (epochs, workers) -> dict
current = None
last_section_key = None

i = 0
while i < len(lines):
    line = lines[i]

    # Section headers
    m = SEC_HDR.search(line)
    if m:
        mode = "baseline" if m.group("label").lower().startswith("baseline") else "attested"
        epochs = int(m.group("epochs"))
        workers = int(m.group("workers"))
        last_section_key = (epochs, workers)
        current = {"mode": mode, "epochs": epochs, "workers": workers, "run_index": 0}
        i += 1
        continue

    # Timing lines (per run)
    if current:
        if current["mode"] == "baseline":
            m = TIMING_BASELINE.search(line)
            if m:
                current["run_index"] += 1
                rows.append({
                    "epochs": current["epochs"],
                    "workers": current["workers"],
                    "mode": "baseline",
                    "run": current["run_index"],
                    "handshake_s": 0.0,
                    "preprocess_s": float(m["preprocess"]),
                    "training_s": float(m["train"]),
                    "report_generation_s": 0.0,
                    "avg_epoch_s": float(m["avg_epoch"]),
                })
        else:
            m = TIMING_ATTEST.search(line)
            if m:
                current["run_index"] += 1
                rows.append({
                    "epochs": current["epochs"],
                    "workers": current["workers"],
                    "mode": "attested",
                    "run": current["run_index"],
                    "handshake_s": float(m["handshake"]),
                    "preprocess_s": float(m["preprocess"]),
                    "training_s": float(m["train"]),
                    "report_generation_s": float(m["report"]),
                    "avg_epoch_s": float(m["avg_epoch"]),
                })

    # Memory sizes block (multi-line)
    if line.strip().lower().startswith("for all runs:"):
        # Grab the next few lines to ensure we capture the 3 sizes even if they wrap
        block = "\n".join(lines[i : min(i + 8, len(lines))])
        sm = SIZES_TRIPLE.search(block)
        if sm and last_section_key is not None:
            e, w = last_section_key
            report = int(sm["report"])
            sig = int(sm["sig"])
            weights = int(sm["weights"])
            memory_overhead[(e, w)] = {
                "hash_report_bytes": report,
                "report_sig_bytes": sig,
                "final_weights_bytes": weights,
                "memory_overhead_bytes": report + sig - weights,
            }

    i += 1

if not rows:
    raise SystemExit("No runs parsed. Check file path/format.")

# ---------- DataFrames ----------
df = pd.DataFrame(rows)
df["total_pipeline_train_s"] = (
        df["handshake_s"] + df["preprocess_s"] + df["training_s"] + df["report_generation_s"]
)
df["baseline_like_total_s"] = df["preprocess_s"] + df["training_s"]

agg = df.groupby(["epochs", "workers", "mode"]).agg(
    runs=("run", "count"),
    handshake_mean_s=("handshake_s", "mean"),
    handshake_std_s=("handshake_s", "std"),
    preprocess_mean_s=("preprocess_s", "mean"),
    preprocess_std_s=("preprocess_s", "std"),
    training_mean_s=("training_s", "mean"),
    training_std_s=("training_s", "std"),
    report_mean_s=("report_generation_s", "mean"),
    report_std_s=("report_generation_s", "std"),
    avg_epoch_mean_s=("avg_epoch_s", "mean"),
    avg_epoch_std_s=("avg_epoch_s", "std"),
    total_pipeline_mean_s=("total_pipeline_train_s", "mean"),
    total_pipeline_std_s=("total_pipeline_train_s", "std"),
    baseline_like_total_mean_s=("baseline_like_total_s", "mean"),
    baseline_like_total_std_s=("baseline_like_total_s", "std"),
).reset_index()

# Pivot to compare attested vs baseline
pivot = agg.pivot(index=["epochs", "workers"], columns="mode",
                  values=["total_pipeline_mean_s", "total_pipeline_std_s",
                          "baseline_like_total_mean_s", "baseline_like_total_std_s"])
pivot.columns = ['_'.join(col).strip() for col in pivot.columns.values]
pivot = pivot.reset_index()

# Locate the baseline mean col robustly
baseline_candidates = [c for c in pivot.columns if c.startswith("baseline_like_total_mean_s_") and c.endswith("baseline")]
if baseline_candidates:
    baseline_mean_col = baseline_candidates[0]
else:
    # Fallback: any baseline_like_total_mean_s_* column (should still be baseline)
    baseline_candidates = [c for c in pivot.columns if c.startswith("baseline_like_total_mean_s_")]
    baseline_mean_col = baseline_candidates[0]

attested_mean_col = "total_pipeline_mean_s_attested"

pivot["overhead_abs_s"] = pivot[attested_mean_col] - pivot[baseline_mean_col]
pivot["overhead_pct"] = 100.0 * pivot["overhead_abs_s"] / pivot[baseline_mean_col]

# Memory overhead DF (ensure it always has the join keys)
mem_rows = [{"epochs": e, "workers": w, **d} for (e, w), d in memory_overhead.items()]
mem_df = pd.DataFrame(mem_rows, columns=[
    "epochs","workers","hash_report_bytes","report_sig_bytes","final_weights_bytes","memory_overhead_bytes"
])

# Merge if we have memory rows; otherwise leave pivot as-is
if not mem_df.empty:
    summary = pivot.merge(mem_df, on=["epochs", "workers"], how="left")
else:
    summary = pivot.copy()
    # Create empty columns to keep downstream code simple
    for col in ["hash_report_bytes","report_sig_bytes","final_weights_bytes","memory_overhead_bytes"]:
        summary[col] = pd.NA

# ---------- Output ----------
pd.set_option("display.max_columns", None)

print("\n=== Aggregated (mean/std) per (epochs, workers, mode) ===")
print(agg.sort_values(["epochs", "workers", "mode"]).to_string(index=False))

print("\n=== Attestation overhead vs baseline (by epochs, workers) ===")
cols = ["epochs","workers", baseline_mean_col, attested_mean_col, "overhead_abs_s","overhead_pct"]
print(summary[cols].sort_values(["epochs","workers"]).to_string(index=False))

print("\n=== Memory overhead per (epochs, workers) ===")
mcols = ["epochs","workers","hash_report_bytes","report_sig_bytes","final_weights_bytes","memory_overhead_bytes"]
print(summary[mcols].drop_duplicates().sort_values(["epochs","workers"]).to_string(index=False))

# Save artifacts
summary.to_csv("census_s_scalability_summary.csv", index=False)
agg.to_csv("census_s_scalability_agg.csv", index=False)
df.to_csv("census_s_scalability_per_run.csv", index=False)

with open("census_s_scalability_summary.json","w") as f:
    json.dump(summary.to_dict(orient="records"), f, indent=2)

print("\nWrote: census_s_scalability_[per_run|agg|summary].csv and census_s_scalability_summary.json")

# ---------- Plots: one figure per (epochs, workers) ----------
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
os.makedirs("figs", exist_ok=True)

# Colors (baseline & attested share preprocess/training colors)
C_PRE = "#4e79a7"   # preprocess
C_TRAIN = "#59a14f" # training
C_HS = "#f28e2b"    # handshake
C_REP = "#edc949"   # report gen

# Build quick lookups from agg & summary
def get_group_mean_std(epochs, workers, mode):
    g = agg[(agg["epochs"] == epochs) & (agg["workers"] == workers) & (agg["mode"] == mode)]
    if g.empty:
        return None
    row = g.iloc[0]
    return {
        "handshake_mean": float(row["handshake_mean_s"]),
        "preprocess_mean": float(row["preprocess_mean_s"]),
        "training_mean": float(row["training_mean_s"]),
        "report_mean": float(row["report_mean_s"]),
        "total_mean": float(row["total_pipeline_mean_s"]),
        "total_std": float(row["total_pipeline_std_s"]),
        "baseline_like_total_mean": float(row["baseline_like_total_mean_s"]),
        "baseline_like_total_std": float(row["baseline_like_total_std_s"]),
    }

# Overhead lookup
summary_lut = {
    (int(r["epochs"]), int(r["workers"])): {
        "baseline_total": float(r[[c for c in summary.columns if c.startswith("baseline_like_total_mean_s_")][0]]),
        "attested_total": float(r["total_pipeline_mean_s_attested"]),
        "overhead_abs": float(r["overhead_abs_s"]),
        "overhead_pct": float(r["overhead_pct"]),
    }
    for _, r in summary.iterrows()
}

pairs = sorted({(int(e), int(w)) for e, w in zip(agg["epochs"], agg["workers"])})

for (e, w) in pairs:
    b = get_group_mean_std(e, w, "baseline")
    a = get_group_mean_std(e, w, "attested")
    if b is None or a is None:
        # skip incomplete pairs
        continue

    # Component means
    b_pre, b_train = b["preprocess_mean"], b["training_mean"]
    a_hs, a_pre, a_train, a_rep = a["handshake_mean"], a["preprocess_mean"], a["training_mean"], a["report_mean"]

    # Totals & 1σ (baseline uses baseline_like_total)
    b_total, b_std = b["baseline_like_total_mean"], b["baseline_like_total_std"]
    a_total, a_std = a["total_mean"], a["total_std"]

    # X positions: left=baseline, right=attested
    x_baseline, x_attest = 0.0, 1.0

    fig, ax = plt.subplots(figsize=(9, 6))

    # --- Baseline stacked bar (preprocess + training) ---
    b1 = ax.bar(x_baseline, b_pre, width=0.6, color=C_PRE, label="Preprocess (baseline)")
    b2 = ax.bar(x_baseline, b_train, width=0.6, bottom=b_pre, color=C_TRAIN, label="Training (baseline)")
    # Error bar on total height
    ax.errorbar([x_baseline], [b_total], yerr=[b_std], fmt="none", capsize=6, elinewidth=2, ecolor="black")

    # --- Attested stacked bar (handshake + preprocess + training + report) ---
    a_stack1 = ax.bar(x_attest, a_hs, width=0.6, color=C_HS, label="Handshake")
    a_stack2 = ax.bar(x_attest, a_pre, width=0.6, bottom=a_hs, color=C_PRE,
                      hatch="//", edgecolor="black", label="Preprocess (attested)")
    a_stack3 = ax.bar(x_attest, a_train, width=0.6, bottom=a_hs + a_pre, color=C_TRAIN,
                      hatch="\\\\", edgecolor="black", label="Training (attested)")
    a_stack4 = ax.bar(x_attest, a_rep, width=0.6, bottom=a_hs + a_pre + a_train, color=C_REP, label="Report gen")
    # Error bar on total height
    ax.errorbar([x_attest], [a_total], yerr=[a_std], fmt="none", capsize=6, elinewidth=2, ecolor="black")

    # Axis/labels
    ax.set_ylabel("Total training pipeline time (s)")
    ax.set_xticks([x_baseline, x_attest], labels=["Baseline", "Attested"])
    ax.set_title(f"CENSUS-S: {e} epochs, {w} workers — Baseline vs Attested")

    # Grid behind bars
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)

    # Force all graphs to share the same Y-axis range
    ax.set_ylim(0, 80)

    # Legend (clean + non-duplicative)
    legend_handles = [
        Patch(facecolor=C_PRE, label="Preprocess (baseline)"),
        Patch(facecolor=C_PRE, hatch="//", edgecolor="black", label="Preprocess (attested)"),
        Patch(facecolor=C_TRAIN, label="Training (baseline)"),
        Patch(facecolor=C_TRAIN, hatch="\\\\", edgecolor="black", label="Training (attested)"),
        Patch(facecolor=C_HS, label="Handshake"),
        Patch(facecolor=C_REP, label="Report gen"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", frameon=True)

    # --- Overhead sticker (placed in axes coordinates to avoid title overlap) ---
    oh = summary_lut.get((e, w))
    if oh:
        sticker = f"Overhead: +{oh['overhead_abs']:.2f}s (+{oh['overhead_pct']:.1f}%)"

        # Place the text at a fixed location in the axes coordinate system.
        # (0.5, 0.88) means center horizontally, 88% up vertically.
        ax.text(
            0.5, 0.88, sticker,
            transform=ax.transAxes,
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="black", alpha=0.9),
            fontsize=12
        )


    # Save
    out_path = f"figs/census_s_{e}e_{w}w.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

print("Wrote per-config figures to: ./figs/ (e.g., figs/census_s_10e_2w.png)")
