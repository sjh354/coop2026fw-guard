# 사용자 질문 1 — 7개 언어 실패 사례 후보 추출. GPU 재실행 없음, 기존
# results/multilingual_{model}.csv(Track A/B baseline) + results/multilingual_variants_{model}.csv만 사용.
# 사람이 스코어링해서 12~18건을 손으로 고르기 위한 원재료(candidate pool)를 만드는 스크립트 —
# 최종 문서화(docs/failure_cases_multilingual.md)는 이 출력에서 수작업으로 큐레이션한다.

import subprocess
from datetime import datetime, timezone

import pandas as pd

from exp.category_fp_fn import parse_pred_categories  # noqa: E402 (재사용, 중복 방지)

MODELS = ["llamaguard", "polyguard", "sguard"]


def main():
    rows = []
    for model in MODELS:
        base = pd.read_csv(f"results/multilingual_{model}.csv")
        base = base.dropna(subset=["pred_label", "true_label"])
        base["pred_categories"] = base["raw_output"].apply(lambda r: parse_pred_categories(model, r))
        wrong = base[base.pred_label != base.true_label].copy()
        wrong["model"] = model
        wrong["source"] = "baseline"
        rows.append(wrong[["model", "source", "lang", "base_id", "track", "text", "true_label", "pred_label", "pred_categories", "confidence"]])

        var = pd.read_csv(f"results/multilingual_variants_{model}.csv")
        var = var.dropna(subset=["pred_label", "true_label"])
        var["pred_categories"] = var["raw_output"].apply(lambda r: parse_pred_categories(model, r))
        wrong_v = var[var.pred_label != var.true_label].copy()
        wrong_v["model"] = model
        wrong_v["source"] = "variant:" + wrong_v["variant_type"]
        wrong_v["track"] = "B"
        rows.append(wrong_v[["model", "source", "lang", "base_id", "track", "text", "true_label", "pred_label", "pred_categories", "confidence"]])

    all_df = pd.concat(rows, ignore_index=True)
    all_df = all_df.sort_values(["lang", "model", "confidence"], ascending=[True, True, False])
    out_path = "results/final/failure_cases_multilingual_candidates.csv"
    all_df.to_csv(out_path, index=False)
    print(f"total candidates: {len(all_df)}")
    print(all_df["lang"].value_counts())
    print(f"\nsaved: {out_path}")


if __name__ == "__main__":
    main()
