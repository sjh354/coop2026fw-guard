"""D4 확장: results/multilingual_variants_{model}.csv(variant 예측)를
results/multilingual_{model}.csv의 track B base 예측과 base_id+lang으로 조인해
flip rate(원문 대비 예측 반전 비율)를 계산한다.

출력: results/final/flip_rate_multilingual.csv, results/final/flip_rate_multilingual.meta.json
"""
import json
import subprocess
from datetime import datetime, timezone

import pandas as pd

MODELS = ["llamaguard", "polyguard", "sguard"]


def main():
    rows = []
    for model in MODELS:
        base = pd.read_csv(f"results/multilingual_{model}.csv")
        base = base[base.track == "B"][["base_id", "lang", "pred_label"]].rename(
            columns={"pred_label": "baseline_pred"}
        )
        variants = pd.read_csv(f"results/multilingual_variants_{model}.csv")

        merged = variants.merge(base, on=["base_id", "lang"], how="left")
        merged = merged.dropna(subset=["pred_label", "baseline_pred"])
        merged["flip"] = merged["pred_label"] != merged["baseline_pred"]

        grouped = merged.groupby(["lang", "variant_type"]).agg(
            flip_rate=("flip", "mean"), n=("flip", "size")
        ).reset_index()
        grouped["model"] = model
        rows.append(grouped)

    out_df = pd.concat(rows, ignore_index=True)[["model", "lang", "variant_type", "flip_rate", "n"]]
    out_df["flip_rate"] = out_df["flip_rate"].round(4)
    out_df = out_df.sort_values(["model", "lang", "variant_type"])

    out_path = "results/final/flip_rate_multilingual.csv"
    out_df.to_csv(out_path, index=False)

    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    meta = {
        "script": "exp/multilingual_variants.py + eval.py --multilingual-variants + exp/flip_compare_multilingual.py",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit,
        "note": (
            "7개 언어(es/hi/ar/ru/th/vi/id) variant는 원어민 검증 없이 규칙기반+기계번역"
            "(Google Translate)으로 생성됨(exp/multilingual_variants.py 참고) — ko_probe/en_probe"
            "(사람이 직접 작성/검토)와 생성 방법론이 달라 flip rate 절대값을 직접 비교하지 말 것."
            " th(태국어) code-switching은 word segmentation 부재로 대부분 no-op(원문과 동일)."
        ),
        "baseline_source": "results/multilingual_{model}.csv (track B, EXP-6 기존 예측 재사용, 재실행 안 함)",
    }
    with open("results/final/flip_rate_multilingual.meta.json", "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(out_df.to_string(index=False))
    print(f"\nsaved: {out_path}")


if __name__ == "__main__":
    main()
