# D9 확장: Track A(PolyGuardPrompts es/hi/th/ar/ru) response harmfulness.
# data/multilingual_track_a_response.csv(exp/extract_track_a_response.py 산출, 150행=5언어x30)를
# 3모델에 돌려 response F1을 prompt F1(EXP-6, results/multilingual_{model}.csv track A)과 병기.
# exp/response_harm.py(en/ko)와 같은 모델 로직(models.generate_batch target="response")을 쓰되
# 다국어 스코프라 별도 스크립트로 분리 — en/ko 파이프라인(동결 대상)은 건드리지 않는다.

import csv
import json
import os
import subprocess
from datetime import datetime, timezone

import pandas as pd
import torch
from sklearn.metrics import f1_score

from models import build_prompt_text, generate_batch, load

MODEL_IDS = {
    "polyguard": "ToxicityPrompts/PolyGuard-Qwen-Smol",
    "llamaguard": "meta-llama/Llama-Guard-3-1B",
    "sguard": "SamsungSDS-Research/SGuard-ContentFilter-2B-v1",
}
BATCH_SIZE = 16
MAX_NEW_TOKENS = 64
IN_PATH = "data/multilingual_track_a_response.csv"


def run_model(model_name, df):
    tok, model = load(MODEL_IDS[model_name])
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    rows = []
    for start in range(0, len(df), BATCH_SIZE):
        batch = df.iloc[start : start + BATCH_SIZE]
        texts = [build_prompt_text(tok, model_name, p, r) for p, r in zip(batch["prompt"], batch["response"])]
        pred_labels, raw_outputs, _ = generate_batch(model_name, tok, model, texts, MAX_NEW_TOKENS, target="response")

        for (_, r), pred, raw in zip(batch.iterrows(), pred_labels, raw_outputs):
            rows.append({
                "base_id": r["base_id"],
                "lang": r["lang"],
                "prompt": r["prompt"],
                "response": r["response"],
                "pred": pred,
                "label": r["response_harm_label"],
                "raw_output": raw,
            })
        print(f"{model_name} track_a response_harm: {min(start + BATCH_SIZE, len(df))}/{len(df)}")

    del model
    torch.cuda.empty_cache()
    return rows


def score(rows):
    valid = [r for r in rows if r["pred"] in ("harmful", "unharmful")]
    fail_rate = (len(rows) - len(valid)) / len(rows) if rows else 0.0
    f1 = f1_score([r["label"] for r in valid], [r["pred"] for r in valid], pos_label="harmful") if valid else None
    return f1, fail_rate, valid


def prompt_axis_f1_by_lang(model_name):
    path = f"results/multilingual_{model_name}.csv"
    df = pd.read_csv(path)
    df = df[df.track == "A"]
    out = {}
    for lang, g in df.groupby("lang"):
        y_true = (g.true_label == "unsafe").astype(int)
        y_pred = (g.pred_label == "unsafe").astype(int)
        tp = ((y_true == 1) & (y_pred == 1)).sum()
        fp = ((y_true == 0) & (y_pred == 1)).sum()
        fn = ((y_true == 1) & (y_pred == 0)).sum()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        out[lang] = round(2 * prec * rec / (prec + rec), 4) if (prec + rec) > 0 else 0.0
    return out


def main():
    os.makedirs("results/final", exist_ok=True)
    df_in = pd.read_csv(IN_PATH)

    summary_rows = []
    for model_name in MODEL_IDS:
        rows = run_model(model_name, df_in)
        fieldnames = ["base_id", "lang", "prompt", "response", "pred", "label", "raw_output"]
        with open(f"results/final/response_harm_track_a_{model_name}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)

        prompt_f1_by_lang = prompt_axis_f1_by_lang(model_name)
        rows_df = pd.DataFrame(rows)
        for lang, g in rows_df.groupby("lang"):
            f1, fail_rate, valid = score(g.to_dict("records"))
            print(f"{model_name}/{lang} (track A response): n={len(g)} parse_fail_rate={fail_rate:.2%} f1={f1}")
            summary_rows.append({
                "model": model_name, "lang": lang, "axis": "response",
                "f1": round(f1, 4) if f1 is not None else None,
                "n": len(valid), "parse_fail_rate": round(fail_rate, 4),
            })
            summary_rows.append({
                "model": model_name, "lang": lang, "axis": "prompt",
                "f1": prompt_f1_by_lang.get(lang), "n": None, "parse_fail_rate": None,
            })

    with open("results/final/harm_axis_summary_track_a.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "lang", "axis", "f1", "n", "parse_fail_rate"])
        w.writeheader()
        w.writerows(summary_rows)

    meta = {
        "models": MODEL_IDS,
        "dataset": "ToxicityPrompts/PolyGuardPrompts (Track A 5개 언어, exp/extract_track_a_response.py)",
        "label_field": "response_harm_label",
        "dtype": "bfloat16",
        "batch_size": BATCH_SIZE,
        "max_new_tokens": MAX_NEW_TOKENS,
        "note": "D9 확장 — Track B(7개 언어 중 id 등)는 response 텍스트가 있는 소스가 없어 구조적으로 불가, Track A만 가능해 진행. en/ko response_harm.py 파이프라인과 별도(그쪽은 동결 대상 아님, 이번에 en/ko도 함께 갱신).",
        "date": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip(),
    }
    with open("results/final/response_harm_track_a.meta.json", "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("saved: results/final/response_harm_track_a_{model}.csv, harm_axis_summary_track_a.csv")


if __name__ == "__main__":
    main()
