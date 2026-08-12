# EXP-4: McNemar 유의성 검정 (docs/NEW_PLAN.md 참고)
# GPU 재실행 없음 — 기존 예측 CSV(results/*.csv)만 사용해 모델 쌍의 정오답 불일치를 검정한다.
#
# 비교쌍:
#   1. PGPrompts en: PolyGuard vs LG3
#   2. PGPrompts ko: PolyGuard vs LG3
#   3. ko probe(150행): 3모델 pairwise 3쌍 (Bonferroni 보정, alpha=0.05/3)
#
# 2x2 분할표: a=둘다 정답, b=A만 정답, c=B만 정답, d=둘다 오답.
# mcnemar는 b+c(불일치 셀 합)가 25 미만이면 exact binomial, 그 이상이면 continuity-corrected chi-square.

import json
import os
import subprocess
from datetime import datetime, timezone

import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar

MODEL_LABELS = {"llamaguard": "LG3-1B", "polyguard": "PolyGuard", "sguard": "SGuard-v1"}

PGPROMPTS_PATHS = {
    "en": "results/{model}_en.csv",
    "ko": "results/{model}_ko.csv",
}
KO_PROBE_PATH = "results/ko_probe_{model}.csv"

EXACT_THRESHOLD = 25  # b+c < 25 -> exact; else chi-square with continuity correction


def load(path_tpl, model):
    df = pd.read_csv(path_tpl.format(model=model))
    df["correct"] = df["pred_label"] == df["true_label"]
    return df[["id", "correct"]].rename(columns={"correct": model})


def cohens_kappa(pred_a, pred_b):
    labels = sorted(set(pred_a) | set(pred_b))
    n = len(pred_a)
    po = (pred_a.values == pred_b.values).sum() / n
    pe = sum(
        (pred_a == lab).sum() / n * (pred_b == lab).sum() / n
        for lab in labels
    )
    if pe == 1.0:
        return float("nan")
    return (po - pe) / (1 - pe)


def run_pair(dataset, model_a, model_b, pred_paths):
    df_a = load(pred_paths, model_a)
    df_b = load(pred_paths, model_b)
    merged = df_a.merge(df_b, on="id", how="inner")
    n = len(merged)

    a_correct = merged[model_a]
    b_correct = merged[model_b]
    both = ((a_correct) & (b_correct)).sum()
    only_a = ((a_correct) & (~b_correct)).sum()
    only_b = ((~a_correct) & (b_correct)).sum()
    neither = ((~a_correct) & (~b_correct)).sum()
    assert both + only_a + only_b + neither == n

    table = [[both, only_a], [only_b, neither]]
    b, c = only_a, only_b
    exact = (b + c) < EXACT_THRESHOLD
    result = mcnemar(table, exact=exact, correction=not exact)

    raw_a = pd.read_csv(pred_paths.format(model=model_a))[["id", "pred_label"]].set_index("id")
    raw_b = pd.read_csv(pred_paths.format(model=model_b))[["id", "pred_label"]].set_index("id")
    common_ids = merged["id"]
    kappa = cohens_kappa(raw_a.loc[common_ids, "pred_label"], raw_b.loc[common_ids, "pred_label"])

    if b > c:
        direction = f"{MODEL_LABELS[model_a]} 우세 (b={b} > c={c})"
    elif c > b:
        direction = f"{MODEL_LABELS[model_b]} 우세 (c={c} > b={b})"
    else:
        direction = "동률 (b=c)"

    return {
        "pair": f"{MODEL_LABELS[model_a]} vs {MODEL_LABELS[model_b]}",
        "dataset": dataset,
        "n": n,
        "b": int(b),
        "c": int(c),
        "test_type": "exact" if exact else "chi2_corrected",
        "p_raw": result.pvalue,
        "p_adjusted": None,  # filled by caller for ko_probe pairs
        "kappa": kappa,
        "direction": direction,
    }


def main():
    rows = []

    for lang, path_tpl in PGPROMPTS_PATHS.items():
        rows.append(run_pair(f"PGPrompts {lang}", "polyguard", "llamaguard", path_tpl))

    ko_probe_pairs = [
        ("polyguard", "llamaguard"),
        ("polyguard", "sguard"),
        ("llamaguard", "sguard"),
    ]
    ko_probe_rows = [run_pair("ko_probe", a, b, KO_PROBE_PATH) for a, b in ko_probe_pairs]
    n_comparisons = len(ko_probe_rows)
    for r in ko_probe_rows:
        r["p_adjusted"] = min(r["p_raw"] * n_comparisons, 1.0)
    rows.extend(ko_probe_rows)

    # PGPrompts 쌍은 단일 비교라 보정 없음(p_adjusted = p_raw)
    for r in rows:
        if r["p_adjusted"] is None:
            r["p_adjusted"] = r["p_raw"]

    out_df = pd.DataFrame(rows)[
        ["pair", "dataset", "n", "b", "c", "test_type", "p_raw", "p_adjusted", "kappa", "direction"]
    ]

    os.makedirs("results/final", exist_ok=True)
    out_path = "results/final/mcnemar.csv"
    out_df.to_csv(out_path, index=False)

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    meta = {
        "script": "exp/mcnemar.py",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit,
        "exact_threshold": EXACT_THRESHOLD,
        "bonferroni_n": n_comparisons,
        "bonferroni_scope": "ko_probe pairwise 3쌍만 보정 (alpha=0.05/3)",
        "sources": {
            "PGPrompts en/ko": "results/{model}_{lang}.csv",
            "ko_probe": "results/ko_probe_{model}.csv",
        },
    }
    with open("results/final/mcnemar.meta.json", "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(out_df.to_string(index=False))
    print(f"\nsaved: {out_path}")


if __name__ == "__main__":
    main()
