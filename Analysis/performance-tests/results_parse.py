#!/usr/bin/env python3
"""
Parse performance_tests.txt and summarize timing breakdowns for:
  - Baseline vs attested pipelines
  - CENSUS-S (1 hidden layer 128) and CENSUS-L (128/256/512/256)

Outputs (written next to this script):
  - performance_breakdown_per_run.csv
  - performance_breakdown_agg.csv
  - figs/ subcomponent bar charts for each (model, mode)
"""

import re
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

BASE_DIR = Path(__file__).resolve().parent
LOG_PATH = BASE_DIR / "performance_tests.txt"

# ---------- Regexes ----------
SEC_HDR = re.compile(
    r"(?P<label>(Baseline|With Attestation))\s*\((?P<model>CENSUS-[SL]),\s*(?P<epochs>\d+)\s*epochs,\s*(?P<workers>\d+)\s*workers\)",
    re.IGNORECASE,
)

TIMING = re.compile(
    r"Timing summary:\s*handshake=(?P<handshake>[0-9.]+)s\s*\|\s*preprocess=(?P<preprocess>[0-9.]+)s\s*\|\s*total_training=(?P<train>[0-9.]+)s\s*\|\s*avg_epoch=(?P<avg_epoch>[0-9.]+)s(?:\s*\|\s*report_generation=(?P<report>[0-9.]+)s)?",
    re.IGNORECASE,
)

COMPONENT = re.compile(
    r"Coordinator:\s*total and average (?P<name>hashing|envelope verification|hash verification|envelope signing) time per task:\s*(?P<total>[0-9.]+)",
    re.IGNORECASE,
)

NAME_MAP = {
    "hashing": "hashing_total_s",
    "envelope verification": "envelope_verify_total_s",
    "hash verification": "hash_verify_total_s",
    "envelope signing": "envelope_sign_total_s",
}

# ---------- Parse ----------
text = LOG_PATH.read_text(encoding="utf-8", errors="replace").splitlines()

rows = []
current_section = None
run_counter = 0

for line in text:
    sec = SEC_HDR.search(line)
    if sec:
        mode = "attested" if sec["label"].lower().startswith("with attestation") else "baseline"
        current_section = {
            "mode": mode,
            "model": sec["model"].upper(),
            "epochs": int(sec["epochs"]),
            "workers": int(sec["workers"]),
        }
        run_counter = 0
        continue

    # Timing summary starts a new run row
    t = TIMING.search(line)
    if current_section and t:
        run_counter += 1
        row = {
            "model": current_section["model"],
            "mode": current_section["mode"],
            "epochs": current_section["epochs"],
            "workers": current_section["workers"],
            "run": run_counter,
            "handshake_s": float(t["handshake"]),
            "preprocess_s": float(t["preprocess"]),
            "training_s": float(t["train"]),
            "avg_epoch_s": float(t["avg_epoch"]),
            "report_generation_s": float(t["report"] or 0.0),
            # crypto components default to 0 unless populated below
            "hashing_total_s": 0.0,
            "envelope_verify_total_s": 0.0,
            "hash_verify_total_s": 0.0,
            "envelope_sign_total_s": 0.0,
        }
        rows.append(row)
        continue

    # Component totals (only meaningful for attested runs)
    c = COMPONENT.search(line)
    if c and rows and current_section and current_section["mode"] == "attested":
        key = NAME_MAP[c["name"].lower()]
        rows[-1][key] = float(c["total"])

if not rows:
    raise SystemExit("No runs parsed. Check performance_tests.txt formatting.")

df = pd.DataFrame(rows)

# Derived columns
df["crypto_total_s"] = (
    df["hashing_total_s"]
    + df["envelope_verify_total_s"]
    + df["hash_verify_total_s"]
    + df["envelope_sign_total_s"]
)
# Approximate gradient/descent time by subtracting crypto work inside total_training
df["gradient_s"] = (df["training_s"] - df["crypto_total_s"]).clip(lower=0.0)
df["pipeline_total_s"] = df["handshake_s"] + df["preprocess_s"] + df["training_s"] + df["report_generation_s"]

# Save per-run
per_run_path = BASE_DIR / "performance_breakdown_per_run.csv"
df.to_csv(per_run_path, index=False)

# Aggregated means/std per (model, mode)
agg = df.groupby(["model", "mode"]).agg(
    runs=("run", "count"),
    handshake_mean_s=("handshake_s", "mean"),
    preprocess_mean_s=("preprocess_s", "mean"),
    gradient_mean_s=("gradient_s", "mean"),
    crypto_mean_s=("crypto_total_s", "mean"),
    hash_mean_s=("hashing_total_s", "mean"),
    envelope_verify_mean_s=("envelope_verify_total_s", "mean"),
    hash_verify_mean_s=("hash_verify_total_s", "mean"),
    envelope_sign_mean_s=("envelope_sign_total_s", "mean"),
    report_mean_s=("report_generation_s", "mean"),
    pipeline_total_mean_s=("pipeline_total_s", "mean"),
).reset_index()

agg_path = BASE_DIR / "performance_breakdown_agg.csv"
agg.to_csv(agg_path, index=False)

print("\n=== Aggregated mean times by model/mode ===")
print(agg.to_string(index=False))
print(f"\nWrote per-run CSV: {per_run_path}")
print(f"Wrote aggregated CSV: {agg_path}")

# ---------- Plots ----------
fig_dir = BASE_DIR / "figs"
fig_dir.mkdir(exist_ok=True)

C_HANDSHAKE = "#f28e2b"
C_PRE = "#4e79a7"
C_GRAD = "#59a14f"
C_SIGN = "#af7aa1"  # coordinator signing
C_VERIFY = "#e15759"  # coordinator verifying worker envelopes
C_HASH = "#76b7b2"
C_HASH_VERIFY = "#ff9da7"
C_REPORT = "#edc949"


def plot_bar(model, mode, subset):
    """Stacked bar per component (means only)."""
    means = subset.iloc[0]
    x = 0.0
    bottoms = 0.0

    plt.figure(figsize=(7, 6))

    # Components are stacked to reflect contribution to pipeline total.
    parts = [
        ("Handshake", means["handshake_mean_s"], C_HANDSHAKE),
        ("Preprocess", means["preprocess_mean_s"], C_PRE),
        ("Gradient (approx)", means["gradient_mean_s"], C_GRAD),
        ("Coordinator signing", means["envelope_sign_mean_s"], C_SIGN),
        ("Envelope verify", means["envelope_verify_mean_s"], C_VERIFY),
        ("Hashing", means["hash_mean_s"], C_HASH),
        ("Hash verify", means["hash_verify_mean_s"], C_HASH_VERIFY),
        ("Report gen", means["report_mean_s"], C_REPORT),
    ]

    handles = []
    for label, height, color in parts:
        if height <= 0:
            continue
        bar = plt.bar(x, height, bottom=bottoms, color=color, width=0.6, label=label)
        bottoms += height
        handles.append(Patch(facecolor=color, label=label))

    plt.ylabel("Time (s)")
    plt.xticks([x], [f"{model} / {mode}"])
    plt.title(f"{model} ({mode}) — subcomponent breakdown")
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.legend(handles=handles, loc="upper left", frameon=True)
    plt.tight_layout()
    out_path = fig_dir / f"{model.lower()}_{mode}_breakdown.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Wrote figure: {out_path}")

# One bar per (model, mode)
for (model, mode), group in agg.groupby(["model", "mode"]):
    plot_bar(model, mode, group)


def plot_grid(all_rows: pd.DataFrame):
    """Combined 2x2 grid with shared Y-axis for easy comparison."""
    # Fixed ordering to keep layout predictable
    order = [
        ("CENSUS-S", "baseline"),
        ("CENSUS-S", "attested"),
        ("CENSUS-L", "baseline"),
        ("CENSUS-L", "attested"),
    ]
    parts = [
        ("Handshake", "handshake_mean_s", C_HANDSHAKE),
        ("Preprocess", "preprocess_mean_s", C_PRE),
        ("Gradient (approx)", "gradient_mean_s", C_GRAD),
        ("Coordinator signing", "envelope_sign_mean_s", C_SIGN),
        ("Envelope verify", "envelope_verify_mean_s", C_VERIFY),
        ("Hashing", "hash_mean_s", C_HASH),
        ("Hash verify", "hash_verify_mean_s", C_HASH_VERIFY),
        ("Report gen", "report_mean_s", C_REPORT),
    ]

    ymax = all_rows["pipeline_total_mean_s"].max() * 1.10

    # Pre-build legend handles so we don't miss components that are zero in the first panel
    legend_handles = [Patch(facecolor=color, label=label) for label, _, color in parts]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharey=True)
    axes = axes.flatten()

    for idx, (model, mode) in enumerate(order):
        ax = axes[idx]
        row = all_rows[(all_rows["model"] == model) & (all_rows["mode"] == mode)]
        if row.empty:
            ax.axis("off")
            continue
        r = row.iloc[0]
        bottom = 0.0
        for label, col, color in parts:
            height = float(r[col])
            if height <= 0:
                continue
            ax.bar(0.0, height, bottom=bottom, width=0.6, color=color)
            bottom += height

        ax.set_title(f"{model} — {mode}")
        ax.set_xticks([0.0], [f"{model}\n{mode}"])
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.set_ylim(0, ymax)

    fig.legend(handles=legend_handles, loc="upper center", ncol=4, frameon=True, bbox_to_anchor=(0.5, 1.00))
    fig.suptitle("Subcomponent breakdown (mean) — shared Y-axis", y=1.05)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    out_path = fig_dir / "combined_breakdown_grid.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Wrote combined grid: {out_path}")


plot_grid(agg)
