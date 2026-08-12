"""7개 언어 base 문장 추출 — Track A/B 병행 설계 (docs/handoff_multilingual.md 기반).

Track A — In-distribution 레퍼런스 (PG 홈그라운드): PolyGuardPrompts 단일 소스, 5개 언어
  (es/hi/th/ar/ru), 언어 간 parallel id로 content-controlled. **이 트랙의 F1로 세 모델
  순위를 매기지 말 것** — PG 자체 데이터라 PG 편향.

Track B — OOD 공정 비교 (제3자, 7개 언어): unsafe는 언어별 native-written harmful 소스,
  safe는 전 언어 RTP-LX로 통일. 모델 순위 비교의 메인 트랙.
  - es/hi/ar/ru unsafe: Aya Redteaming (HF `CohereLabs/aya_redteaming`, 언어별 원어민 작성)
  - th/vi     unsafe: MultiJail (원어민 번역 harmful request, id 없음이라 th/vi만 커버)
  - id        unsafe: IndoSafety (원어민 작성, formal Indonesian) — HF에 미등재라 논문
    저자 GitHub(`falensiazmi/IndoSafety`)의 Eval-1.xlsx에서 가져옴. safe 라벨이 없는
    데이터셋이라 safe는 전 언어와 동일하게 RTP-LX id로 보충(§3 표의 명시된 폴백).
  - 전 언어 safe: RTP-LX, PromptAnnotations.Toxicity==1 (연속 toxicity 라벨 → 이진 매핑,
    Track A의 harm 이진 라벨과 의미가 다르므로 절대 F1을 Track A와 합치지 말 것)

두 트랙을 같은 세 모델에 돌려 **모델별 delta(A 대비 B 낙폭)** 비교가 메인 발표 축.

출력: data/multilingual_base.csv
컬럼: base_id, lang, track, text, label, source_dataset, source_id, source_label, adversarial
  - (base_id, track, lang)이 실질 PK. base_id는 Track A에서 5개 언어가 공유하는 콘텐츠 키라
    그 자체로는 유일하지 않음.
  - adversarial: PolyGuard 원본 필드 보존(Track A만 값 있음, 나머지는 원본에 없어 공란).

원본 파일 캐시(재다운로드 방지, git 미포함): .cache/rtp-lx/, .cache/multijail/, .cache/indosafety/
"""
import json
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd
from datasets import load_dataset

OUT_PATH = "data/multilingual_base.csv"
RTPLX_ZIP_URL = "https://raw.githubusercontent.com/microsoft/RTP-LX/main/RTP-LX/RTP-LX.zip"
RTPLX_ZIP_PASSWORD = b"rtp-lx-entries-4/8/24"
RTPLX_CACHE_DIR = Path(".cache/rtp-lx")
MULTIJAIL_URL = "https://raw.githubusercontent.com/DAMO-NLP-SG/multilingual-safety-for-LLMs/main/data/MultiJail.csv"
MULTIJAIL_CACHE = Path(".cache/multijail/MultiJail.csv")
INDOSAFETY_URL = "https://raw.githubusercontent.com/falensiazmi/IndoSafety/main/dataset/IndoSafety-Eval-1.xlsx"
INDOSAFETY_CACHE = Path(".cache/indosafety/IndoSafety-Eval-1.xlsx")
RANDOM_STATE = 42
N_UNSAFE = 20
N_SAFE = 10

TRACK_B_LANGS = ["es", "hi", "ar", "ru", "th", "vi", "id"]
RTPLX_LOCALE = {"es": "ES", "hi": "HI", "ar": "AR", "ru": "RU", "th": "TH", "vi": "VI", "id": "ID"}


# ============================================================
# 공통 로더 (원본 파일 캐시)
# ============================================================
def ensure_rtplx_extracted(locale):
    out_path = RTPLX_CACHE_DIR / f"RTP_LX_{locale}.json"
    if out_path.exists():
        return out_path
    RTPLX_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = RTPLX_CACHE_DIR / "RTP-LX.zip"
    if not zip_path.exists():
        print(f"[cache] downloading {RTPLX_ZIP_URL} ...")
        urllib.request.urlretrieve(RTPLX_ZIP_URL, zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(f"RTP_LX_{locale}.json", pwd=RTPLX_ZIP_PASSWORD) as src:
            out_path.write_bytes(src.read())
    return out_path


def load_rtplx(locale):
    path = ensure_rtplx_extracted(locale)
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    df["toxicity"] = df["PromptAnnotations"].apply(lambda a: a["Toxicity"])
    return df


def load_multijail():
    if not MULTIJAIL_CACHE.exists():
        MULTIJAIL_CACHE.parent.mkdir(parents=True, exist_ok=True)
        print(f"[cache] downloading {MULTIJAIL_URL} ...")
        urllib.request.urlretrieve(MULTIJAIL_URL, MULTIJAIL_CACHE)
    return pd.read_csv(MULTIJAIL_CACHE)


def load_indosafety():
    if not INDOSAFETY_CACHE.exists():
        INDOSAFETY_CACHE.parent.mkdir(parents=True, exist_ok=True)
        print(f"[cache] downloading {INDOSAFETY_URL} ...")
        urllib.request.urlretrieve(INDOSAFETY_URL, INDOSAFETY_CACHE)
    return pd.read_excel(INDOSAFETY_CACHE)


def rtplx_safe_rows(lang_code):
    """RTP-LX toxicity==1 표본에서 safe 10개를 뽑아 공통 스키마 행으로 변환."""
    rtp = load_rtplx(RTPLX_LOCALE[lang_code])
    safe = rtp[rtp.toxicity == 1].sample(N_SAFE, random_state=RANDOM_STATE)
    return [{
        "base_id": f"rtplx_{lang_code}_{r.Index}",
        "lang": lang_code,
        "text": r.Prompt,
        "label": "safe",
        "source_dataset": "rtp-lx",
        "source_id": int(r.Index),
        "source_label": f"toxicity={r.toxicity}",
        "adversarial": None,
    } for _, r in safe.iterrows()]


# ============================================================
# Track A — PolyGuardPrompts (es, hi, th, ar, ru), content-controlled
# ============================================================
print("[TrackA] loading PolyGuardPrompts...")
ds = load_dataset("ToxicityPrompts/PolyGuardPrompts", split="test").to_pandas()

PG_LANGS = ["Spanish", "Hindi", "Thai", "Arabic", "Russian"]
PG_CODE = {"Spanish": "es", "Hindi": "hi", "Thai": "th", "Arabic": "ar", "Russian": "ru"}

es_pg = ds[ds.language == "Spanish"]
unsafe_ids = es_pg[es_pg.prompt_harm_label == "harmful"].sample(N_UNSAFE, random_state=RANDOM_STATE).id.tolist()
safe_ids = es_pg[es_pg.prompt_harm_label == "unharmful"].sample(N_SAFE, random_state=RANDOM_STATE).id.tolist()
anchor_ids = unsafe_ids + safe_ids

track_a_rows = []
for lang in PG_LANGS:
    sub = ds[(ds.language == lang) & (ds.id.isin(anchor_ids))]
    for _, r in sub.iterrows():
        track_a_rows.append({
            "base_id": f"pg_{r.id}",
            "lang": PG_CODE[lang],
            "track": "A",
            "text": r.prompt,
            "label": "unsafe" if r.prompt_harm_label == "harmful" else "safe",
            "source_dataset": "polyguard",
            "source_id": int(r.id),
            "source_label": r.prompt_harm_label,
            "adversarial": bool(r.adversarial),
        })
track_a_df = pd.DataFrame(track_a_rows)

counts_a = track_a_df.groupby("lang").size()
assert (counts_a == N_UNSAFE + N_SAFE).all(), f"TrackA 언어별 {N_UNSAFE + N_SAFE}행 아님: {counts_a}"
id_sets = track_a_df.groupby("lang").source_id.apply(set)
assert len({frozenset(s) for s in id_sets}) == 1, "TrackA 언어 간 id 집합 불일치"

en_prompts = ds[(ds.language == "English") & (ds.id.isin(anchor_ids))].set_index("id").prompt
passthrough = track_a_df[track_a_df.apply(lambda r: r.text == en_prompts.get(r.source_id), axis=1)]
assert passthrough.empty, f"영어 원문 그대로 통과된 행 발견:\n{passthrough[['lang', 'source_id', 'text']]}"
print(f"[TrackA] OK - {len(track_a_df)}행, 언어별 {N_UNSAFE + N_SAFE}, id 집합 5개 언어 동일, 영어 미번역 통과 없음")

# ============================================================
# Track B — OOD 공정 비교 (7개 언어, native unsafe + RTP-LX safe)
# ============================================================
print("[TrackB] loading native unsafe sources (Aya / MultiJail / IndoSafety)...")

AYA_SPLIT = {"es": "spanish", "hi": "hindi", "ar": "arabic", "ru": "russian"}
track_b_rows = []

# --- es/hi/ar/ru unsafe: Aya Redteaming ---
for lang_code, split in AYA_SPLIT.items():
    aya = load_dataset("CohereLabs/aya_redteaming", split=split).to_pandas()
    aya = aya.reset_index().rename(columns={"index": "row_id"})
    unsafe = aya.sample(N_UNSAFE, random_state=RANDOM_STATE)
    for _, r in unsafe.iterrows():
        track_b_rows.append({
            "base_id": f"aya_{lang_code}_{r.row_id}",
            "lang": lang_code,
            "text": r.prompt,
            "label": "unsafe",
            "source_dataset": "aya_redteaming",
            "source_id": int(r.row_id),
            "source_label": f"harmful ({r.harm_category})",
            "adversarial": None,
        })
    track_b_rows.extend(rtplx_safe_rows(lang_code))

# --- th/vi unsafe: MultiJail ---
mj = load_multijail()
for lang_code in ["th", "vi"]:
    unsafe = mj.sample(N_UNSAFE, random_state=RANDOM_STATE)
    for _, r in unsafe.iterrows():
        track_b_rows.append({
            "base_id": f"multijail_{lang_code}_{r.id}",
            "lang": lang_code,
            "text": r[lang_code],
            "label": "unsafe",
            "source_dataset": "multijail",
            "source_id": int(r.id),
            "source_label": "harmful",
            "adversarial": None,
        })
    track_b_rows.extend(rtplx_safe_rows(lang_code))

# --- id unsafe: IndoSafety (safe 라벨이 없어 RTP-LX id로 보충) ---
indosafety = load_indosafety()
id_unsafe = indosafety.sample(N_UNSAFE, random_state=RANDOM_STATE)
for _, r in id_unsafe.iterrows():
    track_b_rows.append({
        "base_id": f"indosafety_id_{r.id}",
        "lang": "id",
        "text": r.prompt,
        "label": "unsafe",
        "source_dataset": "indosafety",
        "source_id": int(r.id),
        "source_label": f"harmful ({r.risk_area})",
        "adversarial": None,
    })
track_b_rows.extend(rtplx_safe_rows("id"))

track_b_df = pd.DataFrame(track_b_rows)
track_b_df["track"] = "B"

counts_b = track_b_df.groupby("lang").size()
assert (counts_b == N_UNSAFE + N_SAFE).all(), f"TrackB 언어별 {N_UNSAFE + N_SAFE}행 아님: {counts_b}"
assert set(track_b_df.lang.unique()) == set(TRACK_B_LANGS), f"TrackB 언어 불일치: {sorted(track_b_df.lang.unique())}"
print(f"[TrackB] OK - {len(track_b_df)}행, 언어별 {N_UNSAFE + N_SAFE}")

# ============================================================
# 병합 & 저장
# ============================================================
COLUMNS = ["base_id", "lang", "track", "text", "label", "source_dataset", "source_id", "source_label", "adversarial"]
final_df = pd.concat([track_a_df, track_b_df], ignore_index=True)[COLUMNS]

final_df.to_csv(OUT_PATH, index=False)
print(f"\n저장 완료: {OUT_PATH} ({len(final_df)}행)")

print("\n=== Track A (PG 홈그라운드, 순위용 아님) ===")
print(final_df[final_df.track == "A"].groupby(["lang", "label"]).size().unstack(fill_value=0))

print("\n=== Track B (OOD 공정 비교, 순위 메인) ===")
print(final_df[final_df.track == "B"].groupby(["lang", "label"]).size().unstack(fill_value=0))

print("\n소스 분포:")
print(final_df.groupby(["track", "source_dataset"]).size())

print("\n라이선스:")
print("  polyguard      : CC-BY-4.0 (ToxicityPrompts/PolyGuardPrompts, arXiv 2504.04377)")
print("  aya_redteaming : Apache-2.0 (CohereLabs/aya_redteaming)")
print("  rtp-lx         : MIT (microsoft/RTP-LX repo LICENSE) + NOTICE 서드파티 조항 확인 필요")
print("  multijail      : DAMO-NLP-SG/multilingual-safety-for-LLMs repo LICENSE 확인 필요 (ICLR 2024)")
print("  indosafety     : falensiazmi/IndoSafety repo LICENSE 확인 필요 (arXiv 2506.02573, ACL Rolling Review 심사 중 — 비공식 배포본)")
