# EXP-2: Response harmfulness 재현 (docs/NEW_PLAN.md 참고)
#
# 판단 근거(진행 전 스모크 테스트로 확인, 사용자 승인 완료):
# - 기존 results/{model}_{lang}.csv는 PolyGuard/SGuard도 response=""(빈 응답)로 실행됐음이
#   확인되어 재활용 불가 — 계획서의 "PolyGuard는 재실행 불필요" 가정을 뒤집고 3모델 전부 재실행.
# - PGPrompts 데이터셋의 정답 필드는 response_harm_label (harmful/unharmful).
# - LG3-1B는 user+assistant 두 턴을 chat template에 넣으면 자동으로 "Agent 메시지 평가" 모드로
#   전환됨 — 별도 프롬프트 포맷 불필요, 기존 safe/unsafe 첫 줄 파서(parse_prompt_harm) 재사용.
# - PolyGuard는 raw_output에 이미 "Harmful response: yes/no" 필드가 있어 파서만 추가
#   (models.parse_response_harm).
# - SGuard는 모델 카드 예제대로 {"role":"user","prompt":...,"response":...} 입력을 지원
#   (models.py에 이미 구현됨). 단, response 포함 판정은 5카테고리 안전 결정을 다시 생성하는
#   generate_batch의 logit 기반 경로를 반드시 써야 함(일반 decode는 특수 토큰이라 빈 문자열 나옴,
#   스모크 테스트로 확인). target 파라미터는 sguard에는 영향 없음(집계 방식 하나뿐).

import csv
import json
import os
import subprocess
from datetime import datetime, timezone

import torch
from datasets import load_dataset
from sklearn.metrics import f1_score

from models import build_prompt_text, generate_batch, load

MODEL_IDS = {
    "polyguard": "ToxicityPrompts/PolyGuard-Qwen-Smol",
    "llamaguard": "meta-llama/Llama-Guard-3-1B",
    "sguard": "SamsungSDS-Research/SGuard-ContentFilter-2B-v1",
}
LANG_NAMES = {"en": "English", "ko": "Korean"}
BATCH_SIZE = 16
MAX_NEW_TOKENS = 64


def load_examples(lang):
    ds = load_dataset("ToxicityPrompts/PolyGuardPrompts", split="test")
    ds = ds.filter(lambda x: x["language"] == LANG_NAMES[lang] and x["prompt_harm_label"] is not None)
    n_total = len(ds)

    missing_label = ds.filter(lambda x: x["response_harm_label"] is None)
    n_missing_label = len(missing_label)
    ds = ds.filter(lambda x: x["response_harm_label"] is not None)

    empty_response = ds.filter(lambda x: not x["response"] or not x["response"].strip())
    n_empty_response = len(empty_response)
    ds = ds.filter(lambda x: x["response"] and x["response"].strip())

    return ds, {
        "n_total_prompt_harm_labeled": n_total,
        "excluded_missing_response_harm_label": n_missing_label,
        "excluded_empty_response": n_empty_response,
        "n_evaluated": len(ds),
    }


def run_model_lang(model_name, lang, ds):
    tok, model = load(MODEL_IDS[model_name])
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    rows = []
    for start in range(0, len(ds), BATCH_SIZE):
        batch = ds[start : start + BATCH_SIZE]
        texts = [
            build_prompt_text(tok, model_name, p, r) for p, r in zip(batch["prompt"], batch["response"])
        ]
        pred_labels, raw_outputs, _ = generate_batch(model_name, tok, model, texts, MAX_NEW_TOKENS, target="response")

        for i, (pred, raw) in enumerate(zip(pred_labels, raw_outputs)):
            rows.append(
                {
                    "id": batch["id"][i],
                    "text": batch["prompt"][i],
                    "response": batch["response"][i],
                    "pred": pred,
                    "label": batch["response_harm_label"][i],
                    "raw_output": raw,
                }
            )
        print(f"{model_name}/{lang} response_harm: {min(start + BATCH_SIZE, len(ds))}/{len(ds)}")

    del model
    torch.cuda.empty_cache()
    return rows


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def score(rows):
    valid = [r for r in rows if r["pred"] is not None]
    fail = [r for r in rows if r["pred"] is None]
    fail_rate = len(fail) / len(rows) if rows else 0.0
    f1 = f1_score([r["label"] for r in valid], [r["pred"] for r in valid], pos_label="harmful") if valid else None
    return f1, fail_rate, valid, fail


def prompt_axis_f1(model_name, lang):
    path = f"results/{model_name}_{lang}.csv"
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    valid = [r for r in rows if r["pred_label"]]
    return f1_score([r["true_label"] for r in valid], [r["pred_label"] for r in valid], pos_label="harmful")


if __name__ == "__main__":
    os.makedirs("results/final", exist_ok=True)

    summary_rows = []
    exclusion_meta = {}
    for lang in ["en", "ko"]:
        ds, excl = load_examples(lang)
        exclusion_meta[lang] = excl
        for model_name in MODEL_IDS:
            out_path = f"results/final/response_harm_{model_name}_{lang}.csv"
            fieldnames = ["id", "text", "response", "pred", "label", "raw_output"]
            if os.path.exists(out_path):
                # OOM으로 중단됐다 재개하는 경우 이미 끝난 model/lang은 재실행하지 않고 기존 CSV 재사용
                print(f"{model_name}/{lang}: 기존 결과 재사용 ({out_path})")
                with open(out_path, newline="") as f:
                    rows = list(csv.DictReader(f))
                for r in rows:
                    r["pred"] = r["pred"] or None
            else:
                rows = run_model_lang(model_name, lang, ds)
                write_csv(out_path, rows, fieldnames)

            f1, fail_rate, valid, fail = score(rows)
            print(f"{model_name}/{lang}: n={len(rows)} parse_fail_rate={fail_rate:.2%} f1={f1}")
            if fail:
                write_csv(f"results/final/response_harm_parsefail_{model_name}_{lang}.csv", fail, fieldnames)

            summary_rows.append(
                {
                    "model": model_name,
                    "lang": lang,
                    "axis": "response",
                    "f1": round(f1, 4) if f1 is not None else None,
                    "n": len(valid),
                    "parse_fail_rate": round(fail_rate, 4),
                }
            )
            summary_rows.append(
                {
                    "model": model_name,
                    "lang": lang,
                    "axis": "prompt",
                    "f1": round(prompt_axis_f1(model_name, lang), 4),
                    "n": None,
                    "parse_fail_rate": None,
                }
            )

    write_csv(
        "results/final/harm_axis_summary.csv",
        summary_rows,
        ["model", "lang", "axis", "f1", "n", "parse_fail_rate"],
    )

    meta = {
        "models": MODEL_IDS,
        "dataset": "ToxicityPrompts/PolyGuardPrompts",
        "label_field": "response_harm_label",
        "dtype": "bfloat16",
        "batch_size": BATCH_SIZE,
        "max_new_tokens": MAX_NEW_TOKENS,
        "exclusions": exclusion_meta,
        "date": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip(),
    }
    with open("results/final/response_harm.meta.json", "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("saved: results/final/response_harm_{model}_{lang}.csv, harm_axis_summary.csv")
