# 사용자 질문 4 — confidence 분포 + threshold sweep. GPU 재실행 없음, results/*_conf.csv +
# results/multilingual_{model}.csv(confidence 컬럼 이미 존재)만 사용.
#
# confidence는 EXP-3(calibration)에서 결정 토큰 softmax로 이미 추출된 값 — pred_label이 그
# confidence로 나왔다고 가정하고, threshold 미만이면 "확신 없음"으로 보류(reject)했을 때
# 나머지(accepted) 표본의 precision/recall/F1이 어떻게 바뀌는지 sweep한다.

import json
import subprocess
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

MODELS = ["llamaguard", "polyguard", "sguard"]
THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]

DATASETS = {
    "en": "results/{model}_en_conf.csv",
    "ko": "results/{model}_ko_conf.csv",
}


def normalize(df, label_col_true="true_label", label_col_pred="pred_label", pos="harmful"):
    df = df.dropna(subset=[label_col_pred, label_col_true, "confidence"]).copy()
    return df


def sweep(df, pos_label):
    rows = []
    n_total = len(df)
    for t in THRESHOLDS:
        kept = df[df["confidence"] >= t]
        coverage = len(kept) / n_total if n_total else float("nan")
        if len(kept) == 0 or kept[df.columns[0]].nunique() == 0:
            rows.append({"threshold": t, "coverage": coverage, "n": len(kept), "precision": None, "recall": None, "f1": None})
            continue
        p = precision_score(kept["true_label"], kept["pred_label"], pos_label=pos_label, zero_division=0)
        r = recall_score(kept["true_label"], kept["pred_label"], pos_label=pos_label, zero_division=0)
        f1 = f1_score(kept["true_label"], kept["pred_label"], pos_label=pos_label, zero_division=0)
        rows.append({"threshold": t, "coverage": round(coverage, 4), "n": len(kept), "precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4)})
    return rows


def confidence_distribution(df):
    correct = df[df.pred_label == df.true_label]["confidence"]
    wrong = df[df.pred_label != df.true_label]["confidence"]
    return {
        "n": len(df),
        "mean_confidence_correct": round(correct.mean(), 4) if len(correct) else None,
        "mean_confidence_wrong": round(wrong.mean(), 4) if len(wrong) else None,
        "median_confidence_correct": round(correct.median(), 4) if len(correct) else None,
        "median_confidence_wrong": round(wrong.median(), 4) if len(wrong) else None,
        "pct_wrong_conf_ge_0.9": round((wrong >= 0.9).mean(), 4) if len(wrong) else None,
    }


def main():
    sweep_rows = []
    dist_rows = []

    for model in MODELS:
        for lang, path_tpl in DATASETS.items():
            df = pd.read_csv(path_tpl.format(model=model))
            df = normalize(df)
            for t_row in sweep(df, pos_label="harmful"):
                t_row.update({"model": model, "dataset": f"pgprompts_{lang}"})
                sweep_rows.append(t_row)
            d = confidence_distribution(df)
            d.update({"model": model, "dataset": f"pgprompts_{lang}"})
            dist_rows.append(d)

        # multilingual Track A/B (confidence 컬럼 이미 존재, pred_label은 safe/unsafe)
        mdf = pd.read_csv(f"results/multilingual_{model}.csv")
        mdf = normalize(mdf)
        for lang in sorted(mdf["lang"].unique()):
            sub = mdf[mdf.lang == lang]
            for t_row in sweep(sub, pos_label="unsafe"):
                t_row.update({"model": model, "dataset": f"multilingual_{lang}"})
                sweep_rows.append(t_row)
            d = confidence_distribution(sub)
            d.update({"model": model, "dataset": f"multilingual_{lang}"})
            dist_rows.append(d)

    sweep_df = pd.DataFrame(sweep_rows)[["model", "dataset", "threshold", "coverage", "n", "precision", "recall", "f1"]]
    dist_df = pd.DataFrame(dist_rows)[
        ["model", "dataset", "n", "mean_confidence_correct", "mean_confidence_wrong",
         "median_confidence_correct", "median_confidence_wrong", "pct_wrong_conf_ge_0.9"]
    ]

    sweep_df.to_csv("results/final/confidence_threshold_sweep.csv", index=False)
    dist_df.to_csv("results/final/confidence_distribution.csv", index=False)

    pd.set_option("display.max_rows", 300)
    print("=== confidence distribution (correct vs wrong) ===")
    print(dist_df.to_string(index=False))
    print("\n=== threshold sweep, pgprompts en/ko only ===")
    print(sweep_df[sweep_df.dataset.str.startswith("pgprompts")].to_string(index=False))

    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    meta = {
        "script": "exp/confidence_threshold_sweep.py",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit,
        "note": (
            "confidence = EXP-3에서 추출한 결정 토큰 softmax 값(0~1). threshold sweep은 "
            "'confidence < threshold면 사람 검토로 보류'를 가정하고 남은 표본(coverage)의 "
            "precision/recall/F1을 재계산한 것 — 실제 2차 검사 파이프라인을 구현/검증한 것은 아님."
        ),
    }
    with open("results/final/confidence_threshold_sweep.meta.json", "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print("\nsaved: results/final/confidence_threshold_sweep.csv, results/final/confidence_distribution.csv")


if __name__ == "__main__":
    main()
