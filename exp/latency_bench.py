# EXP-1: 3모델 latency 벤치마크 (docs/NEW_PLAN.md 참고)
# 입력: ko_probe 원문 30문장 + PGPrompts en 20문장(기존 results/llamaguard_en.csv 순서 재사용) = 50건
# forward+generate 시간만 측정 (HTTP 경유 없음), 모델당 warmup 5회 제외 후 순차(batch=1) 측정.

import csv
import json
import os
import statistics as stats
import subprocess
import time
from datetime import datetime, timezone

import torch

from models import build_prompt_text, load

MODEL_IDS = {
    "llamaguard": "meta-llama/Llama-Guard-3-1B",
    "polyguard": "ToxicityPrompts/PolyGuard-Qwen-Smol",
    "sguard": "SamsungSDS-Research/SGuard-ContentFilter-2B-v1",
}
N_WARMUP = 5
MAX_NEW_TOKENS = 64
N_EN = 20


def build_inputs():
    inputs = []
    with open("data/ko_probe.csv", newline="") as f:
        for row in csv.DictReader(f):
            if row["variant_type"] == "원문":
                inputs.append({"id": row["id"], "text": row["text"], "lang": "ko"})

    with open("results/llamaguard_en.csv", newline="") as f:
        for i, row in enumerate(csv.DictReader(f)):
            if i >= N_EN:
                break
            inputs.append({"id": row["id"], "text": row["prompt"], "lang": "en"})

    assert len(inputs) == 50, f"expected 50 inputs, got {len(inputs)}"
    with open("exp/latency_inputs.json", "w") as f:
        json.dump(inputs, f, ensure_ascii=False, indent=2)
    return inputs


def generate_one(model_name, tok, model, text):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    inputs = tok(text, return_tensors="pt", add_special_tokens=False).to(model.device)
    torch.cuda.synchronize()
    tokenize_ms = (time.perf_counter() - t0) * 1000
    input_len = inputs["input_ids"].shape[-1]

    gen_kwargs = dict(pad_token_id=tok.pad_token_id, do_sample=False)
    if model_name == "sguard":
        # SGuard는 5개 카테고리 special token을 5스텝 생성하도록 학습됨 (models.py 참고)
        gen_kwargs.update(max_new_tokens=5, return_dict_in_generate=True, output_logits=True)
    else:
        gen_kwargs.update(max_new_tokens=MAX_NEW_TOKENS)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)
    torch.cuda.synchronize()
    generate_ms = (time.perf_counter() - t0) * 1000

    seq = out.sequences if model_name == "sguard" else out
    output_len = seq.shape[-1] - input_len
    return input_len, output_len, tokenize_ms, generate_ms


def bench_model(model_name, inputs):
    tok, model = load(MODEL_IDS[model_name])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    for item in inputs[:N_WARMUP]:
        text = build_prompt_text(tok, model_name, item["text"])
        generate_one(model_name, tok, model, text)

    rows = []
    for item in inputs:
        text = build_prompt_text(tok, model_name, item["text"])
        input_len, output_len, tokenize_ms, generate_ms = generate_one(model_name, tok, model, text)
        rows.append(
            {
                "model": model_name,
                "input_id": item["id"],
                "input_len_tokens": input_len,
                "output_len_tokens": output_len,
                "tokenize_ms": round(tokenize_ms, 3),
                "generate_ms": round(generate_ms, 3),
            }
        )
        print(f"{model_name}: {item['id']} generate_ms={generate_ms:.1f}")

    del model
    torch.cuda.empty_cache()
    return rows


def summarize(raw_rows):
    by_model = {}
    for r in raw_rows:
        by_model.setdefault(r["model"], []).append(r)

    summary = []
    for model_name, rows in by_model.items():
        gen_times = sorted(r["generate_ms"] for r in rows)
        out_tokens = [r["output_len_tokens"] for r in rows]
        n = len(gen_times)
        p95_idx = min(n - 1, int(round(0.95 * (n - 1))))
        total_out_tokens = sum(out_tokens)
        total_gen_ms = sum(r["generate_ms"] for r in rows)
        summary.append(
            {
                "model": model_name,
                "n": n,
                "median_ms": round(stats.median(gen_times), 3),
                "mean_ms": round(stats.mean(gen_times), 3),
                "std_ms": round(stats.stdev(gen_times) if n > 1 else 0.0, 3),
                "p95_ms": round(gen_times[p95_idx], 3),
                "mean_output_tokens": round(stats.mean(out_tokens), 2),
                "ms_per_output_token": round(total_gen_ms / total_out_tokens, 3) if total_out_tokens else None,
            }
        )
    return summary


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def write_meta(path, n_inputs):
    meta = {
        "models": MODEL_IDS,
        "dtype": "bfloat16",
        "n_samples_per_model": n_inputs,
        "n_warmup": N_WARMUP,
        "max_new_tokens": MAX_NEW_TOKENS,
        "batch_size": 1,
        "date": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip(),
    }
    with open(path, "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    os.makedirs("results/final", exist_ok=True)
    inputs = build_inputs()

    raw_rows = []
    for model_name in MODEL_IDS:
        raw_rows.extend(bench_model(model_name, inputs))

    write_csv(
        "results/final/latency_raw.csv",
        raw_rows,
        ["model", "input_id", "input_len_tokens", "output_len_tokens", "tokenize_ms", "generate_ms"],
    )
    write_meta("results/final/latency_raw.meta.json", len(inputs))

    summary_rows = summarize(raw_rows)
    write_csv(
        "results/final/latency_summary.csv",
        summary_rows,
        ["model", "n", "median_ms", "mean_ms", "std_ms", "p95_ms", "mean_output_tokens", "ms_per_output_token"],
    )
    write_meta("results/final/latency_summary.meta.json", len(inputs))

    print("saved: results/final/latency_raw.csv, results/final/latency_summary.csv")
