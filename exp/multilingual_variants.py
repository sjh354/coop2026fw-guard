"""D4 확장: es/hi/ar/ru/th/vi/id 7개 언어에 대해 우회표기/띄어쓰기/번역투/code-switching
4종 variant를 생성한다.

**중요 caveat**: ko_probe/en_probe(`exp/en_variants.py`)는 문장을 사람이 직접 검토/작성했다.
이 스크립트는 그럴 능력(원어민 수준 검증)이 없는 7개 언어를 다뤄야 해서 규칙 기반 +
기계번역(Google Translate, `deep-translator`) 파이프라인으로 대체했다 — 사용자 승인 하에
"원어민 미검증"을 명시하고 진행. 방법론이 ko/en과 다르므로 flip rate 절대값을 ko/en과
직접 비교하지 말 것(variant 생성 방식 자체가 다른 confound).

- 우회표기(obfuscation): 문자 체계별 규칙
  - Latin(es/id/vi): leetspeak 치환(a→4, e→3, i→1, o→0, s→$), 주요 악센트 변형 포함
  - Cyrillic(ru): 시각적으로 동일한 라틴 문자로 치환하는 homoglyph 기법(а→a, е→e, о→o, р→p,
    с→c, у→y, х→x) — 스팸/욕설 필터 우회에 실제 쓰이는 기법
  - Arabic(ar): tatweel(ـ, U+0640) 삽입으로 글자를 늘여써 필터 우회(아랍어 문서에서 실제
    쓰이는 욕설 필터 우회 기법)
  - Devanagari(hi): ZWJ(zero-width joiner, U+200D) 문자 사이 삽입으로 토큰화 우회
  - Thai(th): ZWSP(zero-width space, U+200B) 문자 사이 삽입 — 태국어는 원래 띄어쓰기가 없어
    "띄어쓰기" variant와 겹치는 개념이지만 폭이 다름(zero-width는 렌더링 안 변함, 일반
    space는 보임)

- 띄어쓰기(spacing): 전체 문장 글자 사이에 일반 공백 삽입(스크립트 무관 범용 규칙). ko_probe는
  타겟 단어 하나에만 적용했던 것과 달리 여기선 문장 전체 — 규칙이 더 거칠다는 점을 명시.

- 번역투(translationese): target → en → target 왕복 기계번역(Google Translate). 왕복 후
  원문과 달라지는 어색한 표현이 "번역투" 근사치. 실제 번역투 라벨링이 아니라 왕복번역의 부산물.

- code-switching: 전체 문장을 영어로 기계번역한 뒤, 원문 단어 리스트의 뒤쪽 절반을 영어 번역
  단어 리스트의 뒤쪽 절반으로 치환(단어 수 기준 단순 접합). 언어학적으로 타당한 code-switching
  모델이 아니라 거친 근사치 — 문법적 정합성 보장 안 됨.
  **th(태국어) 알려진 결함**: `text.split()`은 공백 기준 분리라, 태국어는 원래 띄어쓰기가
  없는 언어라서 문장 전체가(또는 문장부호 기준 일부만) 단어 1개로 인식되는 경우가 많음 →
  `len(orig_words) < 2` 조건에 걸려 code-switch variant가 원문과 동일하게 나옴(사실상
  no-op). 실측(2026-08-12 생성분): th code-switching 30개 중 17개(56.7%)가 원문과 동일.
  태국어 word segmentation 라이브러리(pythainlp 등) 없이는 고칠 수 없음 — th의
  code-switching flip rate는 "절반 이상이 애초에 변형되지 않았다"로 해석해야 하며 다른
  언어와 비교 불가.

출력: data/multilingual_variants.csv (7언어 × 30문장 × 4variant = 840행)
"""
import csv
import random
import time

import pandas as pd
from deep_translator import GoogleTranslator

SRC = "data/multilingual_base.csv"
OUT = "data/multilingual_variants.csv"

LEET_MAP = {
    "a": "4", "e": "3", "i": "1", "o": "0", "s": "$",
    "á": "4", "é": "3", "í": "1", "ó": "0", "ú": "u",
    "à": "4", "è": "3", "ì": "1", "ò": "0",
}
CYRILLIC_HOMOGLYPH = {"а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y", "х": "x",
                       "А": "A", "Е": "E", "О": "O", "Р": "P", "С": "C", "У": "Y", "Х": "X"}
TATWEEL = "ـ"
ZWJ = "‍"
ZWSP = "​"

random.seed(0)


def obfuscate(lang, text):
    if lang in ("es", "id", "vi"):
        return "".join(LEET_MAP.get(c, c) for c in text)
    if lang == "ru":
        return "".join(CYRILLIC_HOMOGLYPH.get(c, c) for c in text)
    if lang == "ar":
        return TATWEEL.join(text)
    if lang == "hi":
        return ZWJ.join(text)
    if lang == "th":
        return ZWSP.join(text)
    raise ValueError(lang)


def spaced(text):
    return " ".join(text.replace(" ", ""))


def translationese(lang, text, tr_to_en, tr_from_en):
    en = tr_to_en.translate(text)
    if not en:
        return text
    back = tr_from_en.translate(en)
    return back or text


def code_switch(text, en_text):
    orig_words = text.split()
    en_words = en_text.split() if en_text else []
    if len(orig_words) < 2 or not en_words:
        return text
    cut_orig = len(orig_words) // 2
    cut_en = len(en_words) // 2
    return " ".join(orig_words[:cut_orig] + en_words[cut_en:])


def main():
    df = pd.read_csv(SRC)
    sub = df[df.track == "B"].copy()

    langs = ["es", "hi", "ar", "ru", "th", "vi", "id"]
    out_rows = []
    for lang in langs:
        tr_to_en = GoogleTranslator(source=lang, target="en")
        tr_from_en = GoogleTranslator(source="en", target=lang)
        lang_rows = sub[sub.lang == lang]
        print(f"[{lang}] {len(lang_rows)} base rows")
        for _, r in lang_rows.iterrows():
            text = r["text"]
            en_text = None
            try:
                en_text = tr_to_en.translate(text)
            except Exception as e:
                print(f"  WARN translate-to-en failed for {r['base_id']}: {e}")

            variants = {}
            try:
                variants["우회표기"] = obfuscate(lang, text)
            except Exception as e:
                print(f"  WARN obfuscate failed for {r['base_id']}: {e}")
                variants["우회표기"] = None
            variants["띄어쓰기"] = spaced(text)
            try:
                variants["번역투"] = translationese(lang, text, tr_to_en, tr_from_en)
            except Exception as e:
                print(f"  WARN translationese failed for {r['base_id']}: {e}")
                variants["번역투"] = None
            try:
                variants["code-switching"] = code_switch(text, en_text)
            except Exception as e:
                print(f"  WARN code-switch failed for {r['base_id']}: {e}")
                variants["code-switching"] = None

            for vtype, vtext in variants.items():
                out_rows.append({
                    "base_id": r["base_id"],
                    "lang": lang,
                    "variant_type": vtype,
                    "text": vtext,
                    "base_text": text,
                    "label": r["label"],
                    "source_dataset": r["source_dataset"],
                })
            time.sleep(0.05)

    out_df = pd.DataFrame(out_rows)
    out_df.to_csv(OUT, index=False)
    n_failed = out_df["text"].isna().sum()
    print(f"wrote {len(out_df)} rows to {OUT} ({n_failed} failed/None)")


if __name__ == "__main__":
    main()
