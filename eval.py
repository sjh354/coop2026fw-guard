# PGPrompts 서브셋 평가 → CSV
# Phase 3 재현 (PLAN.md Phase 3 참고). 지표는 sklearn f1_score(pos_label='harmful') 하나.

import argparse

import pandas as pd
import torch
from datasets import load_dataset
from sklearn.metrics import f1_score

from models import build_prompt_text, load, parse_prompt_harm

MODEL_IDS = {
    "llamaguard": "meta-llama/Llama-Guard-3-1B",
    "polyguard": "ToxicityPrompts/PolyGuard-Qwen-Smol",
}
LANG_NAMES = {"en": "English", "ko": "Korean"}


def run_eval(model_name, lang, n_samples, batch_size, max_new_tokens):
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
        inputs = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id
            )
        gen = out[:, inputs["input_ids"].shape[-1] :]
        raw_outputs = tok.batch_decode(gen, skip_special_tokens=True)

        for i, raw in enumerate(raw_outputs):
            pred = parse_prompt_harm(model_name, raw)
            rows.append(
                {
                    "id": batch["id"][i],
                    "prompt": batch["prompt"][i],
                    "true_label": batch["prompt_harm_label"][i],
                    "pred_label": pred,
                    "raw_output": raw,
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

    out_path = f"results/{model_name}_{lang}.csv"
    df.to_csv(out_path, index=False)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODEL_IDS.keys(), required=True)
    parser.add_argument("--lang", choices=LANG_NAMES.keys(), required=True)
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()
    run_eval(args.model, args.lang, args.n, args.batch_size, args.max_new_tokens)
