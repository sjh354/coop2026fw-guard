# 모델 로드 + 프롬프트 빌드 + 출력 파싱
# Phase 2 스모크 테스트에서 raw 출력을 먼저 확인한 뒤 파서를 작성할 것 (PLAN.md Phase 2 참고)

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def load(model_id):
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    return tok, model
