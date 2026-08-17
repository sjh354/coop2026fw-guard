# 사용자 질문 2 — 모델 x 카테고리별 FP/FN 집계. GPU 재실행 없음, 기존 results/*.csv +
# 로컬 HF 캐시(ToxicityPrompts/PolyGuardPrompts parquet, id+language로 join)만 사용.
#
# 정답 카테고리(ground truth):
#   - PGPrompts en/ko(메인) + Track A(es/hi/th/ar/ru): parquet의 subcategory/prompt_categories(S-code)
#   - IndoSafety(id, Track B): data/multilingual_base.csv의 source_label(harmful (카테고리명))
#   - MultiJail/RTP-LX 소스: 세부 카테고리 없음 (분석 대상에서 제외, 이유 명시)
#
# 예측 카테고리: raw_output 파싱
#   - llamaguard: "unsafe\nS8" 형태 → 두 번째 줄
#   - polyguard: "Request safety violations: S2" 정규식
#   - sguard: "Crime:safe, Manipulation:safe, Privacy:unsafe, ..." → unsafe로 찍힌 항목 전부

import glob
import json
import re
import subprocess
from datetime import datetime, timezone

import pandas as pd
import pyarrow.parquet as pq

MODELS = ["llamaguard", "polyguard", "sguard"]


def _find_polyguardprompts_parquet():
    candidates = glob.glob(
        "/Users/sjh354/.cache/huggingface/hub/datasets--ToxicityPrompts--PolyGuardPrompts/blobs/*"
    )
    for p in candidates:
        try:
            if pq.ParquetFile(p).metadata.num_rows > 10000:
                return p
        except Exception:
            continue
    raise FileNotFoundError("PolyGuardPrompts parquet not found in local HF cache")


PARQUET_PATH = _find_polyguardprompts_parquet()

LANG_NAME = {"en": "English", "ko": "Korean", "es": "Spanish", "hi": "Hindi", "th": "Thai", "ar": "Arabic", "ru": "Russian"}


def load_polyguardprompts():
    df = pq.read_table(PARQUET_PATH).to_pandas()
    return df[["id", "language", "prompt_harm_label", "subcategory", "prompt_categories"]]


def parse_pred_categories(model_name, raw_output):
    """예측된 카테고리 목록 반환 (safe로 예측된 경우 빈 리스트)."""
    raw_output = str(raw_output)
    if model_name == "llamaguard":
        lines = [l.strip() for l in raw_output.strip().splitlines() if l.strip()]
        if not lines or lines[0].lower() != "unsafe":
            return []
        return lines[1].split() if len(lines) > 1 else []
    elif model_name == "polyguard":
        m = re.search(r"Request safety violations:\s*(\S.*)", raw_output)
        if not m or m.group(1).strip().lower() == "none":
            return []
        return m.group(1).strip().split()
    elif model_name == "sguard":
        cats = re.findall(r"(\w+):(safe|unsafe)", raw_output)
        return [name for name, v in cats if v == "unsafe"]
    else:
        raise ValueError(model_name)


def pgprompts_fp_fn(model, lang, pred_path, pgp_df):
    df = pd.read_csv(pred_path)
    df["pred_categories"] = df["raw_output"].apply(lambda r: parse_pred_categories(model, r))
    gt = pgp_df[pgp_df.language == LANG_NAME[lang]].set_index("id")
    df = df[df["id"].isin(gt.index)].copy()
    df["gt_subcategory"] = df["id"].map(gt["subcategory"])
    df["gt_categories"] = df["id"].map(gt["prompt_categories"]).fillna("")

    rows = []
    fp = df[(df.pred_label == "harmful") & (df.true_label == "unharmful")]
    for cats in fp["pred_categories"]:
        for c in cats:
            rows.append({"model": model, "lang": lang, "error_type": "FP_over_trigger_category", "category": c})
    fn = df[(df.pred_label == "unharmful") & (df.true_label == "harmful")]
    for cats in fn["gt_categories"]:
        for c in str(cats).split():
            rows.append({"model": model, "lang": lang, "error_type": "FN_missed_category", "category": c})
    for sub in fn["gt_subcategory"]:
        rows.append({"model": model, "lang": lang, "error_type": "FN_missed_subcategory", "category": sub})
    return rows


def track_a_fp_fn(model, pgp_df):
    base = pd.read_csv(f"results/multilingual_{model}.csv")
    base = base[base.track == "A"]
    mb = pd.read_csv("data/multilingual_base.csv")
    mb = mb[mb.source_dataset == "polyguard"][["base_id", "source_id"]]
    df = base.merge(mb, on="base_id", how="left")
    df["pred_categories"] = df["raw_output"].apply(lambda r: parse_pred_categories(model, r))

    rows = []
    for lang in df["lang"].unique():
        gt = pgp_df[pgp_df.language == LANG_NAME[lang]].set_index("id")
        sub = df[df.lang == lang].copy()
        sub = sub[sub["source_id"].isin(gt.index)]
        sub["gt_subcategory"] = sub["source_id"].map(gt["subcategory"])
        sub["gt_categories"] = sub["source_id"].map(gt["prompt_categories"]).fillna("")

        fp = sub[(sub.pred_label == "unsafe") & (sub.true_label == "safe")]
        for cats in fp["pred_categories"]:
            for c in cats:
                rows.append({"model": model, "lang": lang, "error_type": "FP_over_trigger_category", "category": c})
        fn = sub[(sub.pred_label == "safe") & (sub.true_label == "unsafe")]
        for cats in fn["gt_categories"]:
            for c in str(cats).split():
                rows.append({"model": model, "lang": lang, "error_type": "FN_missed_category", "category": c})
        for s in fn["gt_subcategory"]:
            rows.append({"model": model, "lang": lang, "error_type": "FN_missed_subcategory", "category": s})
    return rows


def indosafety_fp_fn(model):
    """Track B id(인도네시아어), source_label = 'harmful (카테고리명)' 형태. FP 카테고리는
    모델 예측(raw_output)에서, FN 카테고리는 IndoSafety source_label에서 뽑는다."""
    base = pd.read_csv(f"results/multilingual_{model}.csv")
    base = base[(base.track == "B") & (base.lang == "id")]
    mb = pd.read_csv("data/multilingual_base.csv")
    mb = mb[mb.source_dataset == "indosafety"][["base_id", "source_label"]]
    df = base.merge(mb, on="base_id", how="left")
    df["pred_categories"] = df["raw_output"].apply(lambda r: parse_pred_categories(model, r))

    rows = []
    fp = df[(df.pred_label == "unsafe") & (df.true_label == "safe")]
    for cats in fp["pred_categories"]:
        for c in cats:
            rows.append({"model": model, "lang": "id", "error_type": "FP_over_trigger_category", "category": c})
    fn = df[(df.pred_label == "safe") & (df.true_label == "unsafe")]
    for label in fn["source_label"].dropna():
        m = re.search(r"\((.+)\)", str(label))
        cat = m.group(1) if m else str(label)
        rows.append({"model": model, "lang": "id", "error_type": "FN_missed_subcategory", "category": cat})
    return rows


def main():
    pgp_df = load_polyguardprompts()
    all_rows = []
    for model in MODELS:
        all_rows += pgprompts_fp_fn(model, "en", f"results/{model}_en.csv", pgp_df)
        all_rows += pgprompts_fp_fn(model, "ko", f"results/{model}_ko.csv", pgp_df)
        all_rows += track_a_fp_fn(model, pgp_df)
        all_rows += indosafety_fp_fn(model)

    df = pd.DataFrame(all_rows)
    counts = df.groupby(["model", "error_type", "category"]).size().reset_index(name="n")
    counts = counts.sort_values(["model", "error_type", "n"], ascending=[True, True, False])

    out_path = "results/final/category_fp_fn.csv"
    counts.to_csv(out_path, index=False)

    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    meta = {
        "script": "exp/category_fp_fn.py",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit,
        "coverage": (
            "PGPrompts en/ko(메인 1699개), Track A(es/hi/th/ar/ru, polyguard 소스), "
            "IndoSafety(id, Track B) — 정답 카테고리가 존재하는 소스만 포함."
        ),
        "excluded": "MultiJail·RTP-LX 소스 샘플(en/ko probe 제외 다국어 일부) — 원본에 세부 카테고리 없음(harmful/harmless 또는 toxicity score만 존재).",
        "gt_source": "ToxicityPrompts/PolyGuardPrompts parquet (로컬 HF 캐시, id+language join), IndoSafety source_label(data/multilingual_base.csv)",
    }
    with open("results/final/category_fp_fn.meta.json", "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    pd.set_option("display.max_rows", 200)
    print(counts.to_string(index=False))
    print(f"\nsaved: {out_path}")


if __name__ == "__main__":
    main()
