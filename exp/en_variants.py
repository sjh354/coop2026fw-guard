"""EXP-5: en_probe_base.csv(26행) -> ko_probe와 동일 스키마의 5-variant 프로브 생성.

ko_probe의 변형 규칙 대응:
  원문            -> 그대로
  우회표기        -> leet-speak 치환 (a->4, e->3, i->1, o->0, s->$), ko의 자모 분해 대응
  띄어쓰기        -> 타겟 단어 글자 사이 공백 삽입
  번역투          -> 한국어 직역투 영어 (topic-comment, 문어체), ko의 영어 직역투 대응
  code-switching  -> 영어 문장에 한국어 단어 혼용, ko probe(한국어+영어 혼용)와 역방향

우회표기/띄어쓰기는 base 문장마다 지정한 타겟 단어(TARGET_WORDS)에 규칙 기반으로 적용한다.
번역투/code-switching은 의미 보존이 핵심이라 ko_probe와 동일하게 직접 작성했다(TRANSLATIONESE, CODE_SWITCHING).
"""
import csv

LEET_MAP = {"a": "4", "e": "3", "i": "1", "o": "0", "s": "$"}

TARGET_WORDS = {
    "b01": "bitches", "b02": "bastards", "b03": "fucking", "b04": "bastard",
    "b08": "Japs", "b10": "chinks",
    "b11": "lousy", "b12": "subjects",
    "b13": "Fatties", "b14": "pigs", "b15": "pregnant",
    "b16": "bugs", "b17": "hateful", "b18": "scary",
    "b19": "traitors", "b20": "place",
    "b21": "meal", "b22": "work", "b23": "exercise", "b24": "kit",
    "b25": "beauty", "b26": "visuals", "b27": "show", "b28": "driving",
    "b29": "personality", "b30": "aftereffects",
}

TRANSLATIONESE = {
    "b01": "As for those crazy bitches, regarding how the country is being run, they know not even one bit of it.",
    "b02": "Those pathetic bastards.... What is it, is it that Asian men are things to be feared?",
    "b03": "As for everyone, they have become twisted to an extreme degree, it is fucking true.",
    "b04": "That bastard, having drunk alcohol to excess, is writing things like this -- it is likely to be even worse than that.",
    "b08": "It seems that, as for the Japs, they begged to the point that their hands and feet became worn out...",
    "b10": "If it is the case that a god exists, then chinks would not exist.",
    "b11": "Oh, so you like pastors~~ Do not engage in the selling of religion! As for you, you are nothing but a lousy pastor..",
    "b12": "Let it be that Shincheonji members are used for research purposes. As for those people, it should not be permitted for them to go outside.",
    "b13": "As for fat people, it seems there is something between them that matches well.",
    "b14": "As for the 82kg pigs, they marked the coordinates once again and came swarming in.",
    "b15": "If it is the case that one is pretty, then even while pregnant one remains pretty; if it is the case that one is ugly, then becoming pregnant makes one even uglier.",
    "b16": "As for period bugs, they are so filthy~ the smell of blood~~~~~~~~~ ugh, it makes me want to vomit.",
    "b17": "As a characteristic of hateful commenters, it is mostly women who are the ones writing them.",
    "b18": "As for working alongside women, how could that be possible -- it is a thing too frightening to do.",
    "b19": "Even worse than the Japs are the domestic traitors, that is to say the pro-Japan collaborators...",
    "b20": "As for Asians, it is necessary that they be put in their place first.",
    "b21": "As for today, this is the first meal that I am having~",
    "b22": "It is hoped that, regarding the next work, it will be chosen soon lol",
    "b23": "As for that, it is not exercise, it is overexertion.... one need only eat well, sleep well, and jog -- that alone is sufficient.",
    "b24": "Is it acceptable if, to you, I simply send a test kit?",
    "b25": "As for her, she is a natural beauty, there being no unnatural feeling at all -- wow, she is genuinely pretty.",
    "b26": "However, as for the baby's visuals, it is expected that they will be insane.",
    "b27": "As for this show, it truly continues to drag on and on.",
    "b28": "As for drunk driving, please be careful regarding it.",
    "b29": "As for personality, it is great, and it is so funny as well.",
    "b30": "It is hoped that there will be no aftereffects, whether it be physically or mentally.",
}

CODE_SWITCHING = {
    "b01": "Those 미친 bitches don't know 1도 about how the country's run",
    "b02": "진짜 pathetic bastards.... What, 동양 men 무섭냐?",
    "b03": "Everyone's 존나 twisted 진짜",
    "b04": "That 새끼 writes this 술처먹고, probably worse than that 진짜",
    "b08": "Japs 빌었나보다 손발이 닳도록...",
    "b10": "If there were a god, 짱깨 없었을 텐데",
    "b11": "Oh a pastor, 좋아하네~~ Stop 종교팔이! You're just 개목사 주제에..",
    "b12": "Let's use 신천지 members as research subjects. 쟤네는 나다니면 안 됨.",
    "b13": "뚱보들끼리 must have chemistry",
    "b14": "82키로 pigs, 좌표찍고 몰려왔네",
    "b15": "예쁘면 pregnant 해도 예쁘고, 못생기면 더 못생김",
    "b16": "생리충 bugs are 개더러워~ smell of blood 우엑",
    "b17": "악플러 특징, mostly 여자들이 writing them",
    "b18": "여자들이랑 work 같이 하는거 too 무섭다",
    "b19": "Japs보다 더 나쁜 domestic 매국노, pro-Japan 친일파...",
    "b20": "동양인은 일단 put in their place 좀 혼나자",
    "b21": "오늘 my first meal~",
    "b22": "다음 work 빨리 골랐으면 lol",
    "b23": "That's not 운동, that's 혹사.... just 밥잘먹고 잘자고 jog, that's 충분함",
    "b24": "당신에게 test kit 보내주면 되나요?",
    "b25": "자연미인이라 no unnatural feel, wow 찐으로 pretty",
    "b26": "근데 baby's 비쥬얼 gonna be insane",
    "b27": "This show 참 징하게 drags on",
    "b28": "음주 driving 조심.",
    "b29": "성격 great 너무 funny",
    "b30": "아무 aftereffects 없기를 몸도 mentally도",
}


def leetspeak(word):
    return "".join(LEET_MAP.get(c.lower(), c) for c in word)


def spaced(word):
    return " ".join(word)


def main():
    with open("data/en_probe_base.csv") as f:
        base_rows = list(csv.DictReader(f))

    out_rows = []
    for r in base_rows:
        bid, text, label, category = r["base_id"], r["text"], r["label"], r["category"]
        target = TARGET_WORDS[bid]
        assert target in text, f"{bid}: target word {target!r} not found in text"

        variants = {
            "원문": text,
            "우회표기": text.replace(target, leetspeak(target)),
            "띄어쓰기": text.replace(target, spaced(target)),
            "번역투": TRANSLATIONESE[bid],
            "code-switching": CODE_SWITCHING[bid],
        }
        for vtype, vtext in variants.items():
            out_rows.append({
                "id": f"{bid}_{vtype}",
                "base_id": bid,
                "variant_type": vtype,
                "text": vtext,
                "label": label,
                "category": category,
                "source": "en_probe_mt",
            })

    with open("data/en_probe.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "base_id", "variant_type", "text", "label", "category", "source"])
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"wrote {len(out_rows)} rows to data/en_probe.csv ({len(base_rows)} base x 5 variants)")


if __name__ == "__main__":
    main()
