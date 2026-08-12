"""D9 확장: Track A(PolyGuardPrompts es/hi/th/ar/ru) anchor 30문장에 대해 response +
response_harm_label을 추출한다. `exp/extract_multilingual_base.py`와 동일한 anchor_ids
선정 로직(es 앵커 → parallel id로 5개 언어 추출)을 재사용해 `data/multilingual_base.csv`의
Track A 행과 1:1 대응하는 base_id를 만든다.

출력: data/multilingual_track_a_response.csv
컬럼: base_id, lang, prompt, response, response_harm_label, source_id
"""
import pandas as pd
from datasets import load_dataset

N_UNSAFE, N_SAFE, RANDOM_STATE = 20, 10, 42

print("[TrackA-response] loading PolyGuardPrompts...")
ds = load_dataset("ToxicityPrompts/PolyGuardPrompts", split="test").to_pandas()

PG_LANGS = ["Spanish", "Hindi", "Thai", "Arabic", "Russian"]
PG_CODE = {"Spanish": "es", "Hindi": "hi", "Thai": "th", "Arabic": "ar", "Russian": "ru"}

es_pg = ds[ds.language == "Spanish"]
unsafe_ids = es_pg[es_pg.prompt_harm_label == "harmful"].sample(N_UNSAFE, random_state=RANDOM_STATE).id.tolist()
safe_ids = es_pg[es_pg.prompt_harm_label == "unharmful"].sample(N_SAFE, random_state=RANDOM_STATE).id.tolist()
anchor_ids = unsafe_ids + safe_ids

rows = []
for lang in PG_LANGS:
    sub = ds[(ds.language == lang) & (ds.id.isin(anchor_ids))]
    for _, r in sub.iterrows():
        rows.append({
            "base_id": f"pg_{r.id}",
            "lang": PG_CODE[lang],
            "prompt": r.prompt,
            "response": r.response,
            "response_harm_label": r.response_harm_label,
            "source_id": int(r.id),
        })

out_df = pd.DataFrame(rows)
counts = out_df.groupby("lang").size()
assert (counts == N_UNSAFE + N_SAFE).all(), f"언어별 30행 아님: {counts}"

n_missing_label = out_df["response_harm_label"].isna().sum()
n_missing_response = out_df["response"].isna().sum() | (out_df["response"] == "").sum()
print(f"missing response_harm_label: {n_missing_label} / {len(out_df)}")
print(f"missing/empty response: {n_missing_response} / {len(out_df)}")

out_path = "data/multilingual_track_a_response.csv"
out_df.to_csv(out_path, index=False)
print(f"wrote {len(out_df)} rows to {out_path}")
print(out_df.groupby("lang").response_harm_label.value_counts())
