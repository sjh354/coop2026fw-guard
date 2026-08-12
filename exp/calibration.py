# EXP-3: Confidence calibration (docs/NEW_PLAN.md 참고)
# GPU 재실행 없음 — eval.py --tag _conf로 이미 저장된 예측(PGPrompts en/ko 전량 + ko_probe 150행)을
# 모델별로 합쳐 10-bin reliability diagram + ECE를 계산한다.
# 세 파일 모두 컬럼명이 id/pred_label/true_label/confidence로 동일해서 그대로 합칠 수 있다.

import json
import os
import subprocess
from datetime import datetime, timezone

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MODELS = ["llamaguard", "polyguard", "sguard"]
N_BINS = 10
BIN_EDGES = np.linspace(0.5, 1.0, N_BINS + 1)
MIN_BIN_N = 30

SOURCE_PATHS = {
    "en": "results/{model}_en_conf.csv",
    "ko": "results/{model}_ko_conf.csv",
    "ko_probe": "results/ko_probe_{model}_conf.csv",
}

CONFIDENCE_DEFINITION = {
    "llamaguard": "safe/unsafe 결정 토큰 2개 후보 softmax max",
    "polyguard": "yes/no 결정 토큰 2개 후보 softmax max",
    "sguard": "5카테고리 네이티브 확률 — triggered 카테고리 중 max, 전부 safe면 min (타 모델과 정의 다름)",
}


def load_model_predictions(model_name):
    frames = []
    for source, path_tpl in SOURCE_PATHS.items():
        df = pd.read_csv(path_tpl.format(model=model_name))
        sub = df[["id", "pred_label", "true_label", "confidence"]].copy()
        sub.columns = ["id", "pred", "label", "confidence"]
        sub["source"] = source
        frames.append(sub)
    combined = pd.concat(frames, ignore_index=True)
    n_before = len(combined)
    combined = combined.dropna(subset=["pred", "confidence"])
    n_dropped = n_before - len(combined)
    combined["correct"] = (combined["pred"] == combined["label"]).astype(int)
    return combined, n_dropped


def compute_bins(df):
    bin_idx = np.clip(np.digitize(df["confidence"], BIN_EDGES[1:-1], right=True), 0, N_BINS - 1)
    df = df.assign(bin=bin_idx)
    rows = []
    for b in range(N_BINS):
        sub = df[df["bin"] == b]
        n = len(sub)
        rows.append(
            {
                "bin": b,
                "bin_start": round(float(BIN_EDGES[b]), 3),
                "bin_end": round(float(BIN_EDGES[b + 1]), 3),
                "n": n,
                "mean_confidence": round(sub["confidence"].mean(), 4) if n else None,
                "accuracy": round(sub["correct"].mean(), 4) if n else None,
                "low_sample": n < MIN_BIN_N,
            }
        )
    return pd.DataFrame(rows)


def compute_ece(bins_df, n_total):
    valid = bins_df.dropna(subset=["mean_confidence", "accuracy"])
    weighted_gap = (valid["n"] / n_total) * (valid["mean_confidence"] - valid["accuracy"]).abs()
    return round(float(weighted_gap.sum()), 4)


def plot_reliability(model_name, bins_df, df, ece, path):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5.5, 7), gridspec_kw={"height_ratios": [3, 1]}, sharex=True)

    ax1.plot([0.5, 1.0], [0.5, 1.0], linestyle="--", color="gray", label="perfect calibration")
    plotted = bins_df.dropna(subset=["mean_confidence", "accuracy"])
    colors = ["#999999" if row.low_sample else "#1f77b4" for row in plotted.itertuples()]
    ax1.bar(
        plotted["bin_start"],
        plotted["accuracy"],
        width=0.045,
        align="edge",
        color=colors,
        edgecolor="black",
        alpha=0.85,
        label="accuracy per bin",
    )
    ax1.set_ylim(0.0, 1.02)
    ax1.set_xlim(0.5, 1.0)
    ax1.set_ylabel("accuracy")
    ax1.set_title(f"{model_name} reliability diagram (ECE={ece:.4f}, n={len(df)})")
    ax1.legend(loc="upper left", fontsize=8)

    # 서버에 Hangul 폰트가 없어(fc-list 확인) matplotlib DejaVu Sans가 한글 glyph를 못 그린다.
    # 캡션만 영어로 둔다 — CSV/문서는 그대로 한국어.
    caption = "Gray bins = n < 30 (low-confidence, avoid over-interpreting)."
    if model_name == "sguard":
        caption += (
            "\nSGuard confidence differs from LG3/PolyGuard (2-way token softmax):\n"
            "native 5-category probability (max over triggered, else min)."
        )
    ax1.text(0.02, 0.02, caption, transform=ax1.transAxes, fontsize=7, va="bottom", color="#444444")

    ax2.hist(df["confidence"], bins=BIN_EDGES, color="#1f77b4", edgecolor="black", alpha=0.7)
    ax2.set_xlabel("confidence")
    ax2.set_ylabel("count")
    ax2.set_xlim(0.5, 1.0)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs("results/final", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    summary_rows = []
    for model_name in MODELS:
        df, n_dropped = load_model_predictions(model_name)
        df[["id", "pred", "label", "confidence", "source"]].to_csv(
            f"results/final/calibration_{model_name}.csv", index=False
        )

        bins_df = compute_bins(df)
        ece = compute_ece(bins_df, len(df))
        n_low = int(bins_df["low_sample"].sum())
        summary_rows.append(
            {
                "model": model_name,
                "ece": ece,
                "n_total": len(df),
                "n_dropped_parse_fail": int(n_dropped),
                "n_bins_low_sample": n_low,
            }
        )
        print(f"{model_name}: ECE={ece} n={len(df)} dropped={n_dropped} low_sample_bins={n_low}")

        plot_reliability(model_name, bins_df, df, ece, f"figures/reliability_{model_name}.png")

    pd.DataFrame(summary_rows).to_csv("results/final/calibration_summary.csv", index=False)

    meta = {
        "n_bins": N_BINS,
        "bin_range": [0.5, 1.0],
        "min_bin_n": MIN_BIN_N,
        "sources": list(SOURCE_PATHS.keys()),
        "confidence_definition": CONFIDENCE_DEFINITION,
        "date": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip(),
    }
    with open("results/final/calibration_summary.meta.json", "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("saved: results/final/calibration_{model}.csv, calibration_summary.csv, figures/reliability_{model}.png")
