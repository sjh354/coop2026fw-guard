# PGPrompts 서브셋 평가 → CSV
# Phase 3 재현 (PLAN.md Phase 3 참고). 지표는 sklearn f1_score(pos_label='harmful') 하나.

import argparse

import pandas as pd
import torch
from datasets import load_dataset
from sklearn.metrics import f1_score

from models import build_prompt_text, generate_batch, load

MODEL_IDS = {
    "llamaguard": "meta-llama/Llama-Guard-3-1B",
    "polyguard": "ToxicityPrompts/PolyGuard-Qwen-Smol",
    "sguard": "SamsungSDS-Research/SGuard-ContentFilter-2B-v1",
}
LANG_NAMES = {"en": "English", "ko": "Korean"}


def run_eval_ko_probe(model_name, batch_size, max_new_tokens, tag=""):
    df_in = pd.read_csv("data/ko_probe.csv")

    tok, model = load(MODEL_IDS[model_name])
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    rows = []
    for start in range(0, len(df_in), batch_size):
        batch = df_in.iloc[start : start + batch_size]
        texts = [build_prompt_text(tok, model_name, p) for p in batch["text"]]
        pred_labels, raw_outputs, confidences = generate_batch(model_name, tok, model, texts, max_new_tokens)

        for (_, r), pred, raw, conf in zip(batch.iterrows(), pred_labels, raw_outputs, confidences):
            rows.append(
                {
                    "id": r["id"],
                    "base_id": r["base_id"],
                    "variant_type": r["variant_type"],
                    "text": r["text"],
                    "true_label": r["label"],
                    "pred_label": pred,
                    "raw_output": raw,
                    "confidence": conf,
                }
            )
        print(f"{model_name}/ko_probe: {min(start + batch_size, len(df_in))}/{len(df_in)}")

    del model
    torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    parse_fail_rate = df["pred_label"].isna().mean()
    print(f"parse failure rate: {parse_fail_rate:.2%}")

    baseline = df[df["variant_type"] == "원문"].set_index("base_id")["pred_label"]
    df["baseline_pred"] = df["base_id"].map(baseline)
    flippable = df.dropna(subset=["pred_label", "baseline_pred"])
    flippable = flippable[flippable["variant_type"] != "원문"]
    flip = flippable["pred_label"] != flippable["baseline_pred"]
    flip_rate = flippable.assign(flip=flip).groupby("variant_type")["flip"].mean()
    print("variant_type별 flip rate (원문 예측 대비):")
    print(flip_rate.to_string())

    out_path = f"results/ko_probe_{model_name}{tag}.csv"
    df.to_csv(out_path, index=False)
    print(f"saved: {out_path}")


def run_eval(model_name, lang, n_samples, batch_size, max_new_tokens, tag=""):
    ds = load_dataset("ToxicityPrompts/PolyGuardPrompts", split="test")
    ds = ds.filter(lambda x: x["language"] == LANG_NAMES[lang] and x["prompt_harm_label"] is not None)
    ds = ds.select(range(min(n_samples, len(ds))))

    tok, model = load(MODEL_IDS[model_name])
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    rows = []
    for start in range(0, len(ds), batch_size):
        batch = ds[start : start + batch_size]
        texts = [build_prompt_text(tok, model_name, p) for p in batch["prompt"]]
        pred_labels, raw_outputs, confidences = generate_batch(model_name, tok, model, texts, max_new_tokens)

        for i, (pred, raw, conf) in enumerate(zip(pred_labels, raw_outputs, confidences)):
            rows.append(
                {
                    "id": batch["id"][i],
                    "prompt": batch["prompt"][i],
                    "true_label": batch["prompt_harm_label"][i],
                    "pred_label": pred,
                    "raw_output": raw,
                    "adversarial": batch["adversarial"][i],
                    "confidence": conf,
                }
            )
        print(f"{model_name}/{lang}: {min(start + batch_size, len(ds))}/{len(ds)}")

    del model
    torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    parse_fail_rate = df["pred_label"].isna().mean()
    valid = df.dropna(subset=["pred_label"])
    f1 = f1_score(valid["true_label"], valid["pred_label"], pos_label="harmful")

    print(f"parse failure rate: {parse_fail_rate:.2%}")
    print(f"F1 (pos_label=harmful, parse 실패 제외 {len(valid)}/{len(df)}건): {f1:.4f}")

    out_path = f"results/{model_name}_{lang}{tag}.csv"
    df.to_csv(out_path, index=False)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODEL_IDS.keys(), required=True)
    parser.add_argument("--lang", choices=LANG_NAMES.keys())
    parser.add_argument("--ko-probe", action="store_true", help="data/ko_probe.csv로 flip rate 평가 (Phase 5)")
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--tag", default="", help="출력 파일명에 붙일 접미사, 예: _conf (기존 결과 덮어쓰지 않기 위함)")
    args = parser.parse_args()
    if args.ko_probe:
        run_eval_ko_probe(args.model, args.batch_size, args.max_new_tokens, args.tag)
    else:
        if not args.lang:
            parser.error("--lang은 --ko-probe가 아닐 때 필수")
        run_eval(args.model, args.lang, args.n, args.batch_size, args.max_new_tokens, args.tag)
