# -*- coding: utf-8 -*-
"""답변 생성 (Qwen3.6). 판단은 하지 않고, 고른 근거로 답만 쓴다. 4단계: 답 고치기 A1~A4."""
import re

from .judge import numbered

NOT_FOUND = "제공된 문서에서 해당 내용을 찾을 수 없습니다."

ANSWER_PROMPT = """너는 회사 문서 기반 질의응답 도우미다. 아래 [근거]만 사용해 [질문]에 한국어로 답하라.
- 근거에 있는 숫자·조건·기간을 정확히 옮기고, 끝에 사용한 근거 번호를 [1]처럼 붙인다.
- 근거에 답이 없으면 추측하지 말고 정확히 "{nf}" 라고만 답한다.
- 3문장 이내로 간결하게.

{hist}[근거]
{ctx}

[질문]
{q}"""


#------------------------------------------------------------------
# 이전 대화 → 글
#=> [{"q","a"}, …] 를 "사용자: …\n도우미: …" 줄로 바꾼다.
#
# -in: history = 이전 대화 목록 (None 이면 빈 목록)
#
# -out: text = 대화 글 (없으면 빈 문자열)
# -out: error = 없음
#------------------------------------------------------------------
def history_text(history: list[dict] | None) -> str:
    return "\n".join(f"사용자: {h['q']}\n도우미: {h['a']}" for h in history or [])


#------------------------------------------------------------------
# 판단용 질문 만들기
#=> 멀티턴이면 decide 가 앞 대화를 보고 판단하도록 "[이전 대화] … [현재 질문] …" 으로 감싼다.
#
# -in: question = 현재 질문
# -in: history = 이전 대화 목록
#
# -out: text = 판단용 질문 (이전 대화가 없으면 질문 그대로)
# -out: error = 없음
#------------------------------------------------------------------
def judge_question(question: str, history: list[dict] | None) -> str:
    if not history:
        return question
    return f"[이전 대화]\n{history_text(history)}\n[현재 질문] {question}"


#------------------------------------------------------------------
# 답변 생성
#=> 근거 청크에 번호를 붙여 프롬프트에 넣고 Qwen 에 한 번 묻는다.
#    1) 이전 대화가 있으면 프롬프트에 함께 넣는다 ("그럼 그건?" 같은 질문을 이해하도록)
#    2) 근거가 없으면 "(없음)" 을 넣어 모델이 거절 문구로 답하게 한다
#
# -in: llm = LLMClient
# -in: question = 현재 질문
# -in: ctx = 근거 청크 목록
# -in: history = 이전 대화 (기본 None)
# -in: max_tokens = 최대 생성 토큰 (기본 400)
#
# -out: (answer, info) = 답 문자열, 사용량 정보 dict
# -out: error = LLM 서버 실패 시 RuntimeError
#------------------------------------------------------------------
def generate_answer(llm, question: str, ctx: list[dict], history: list[dict] | None = None,
                    max_tokens: int = 400) -> tuple[str, dict]:
    hist = f"[이전 대화]\n{history_text(history)}\n\n" if history else ""
    prompt = ANSWER_PROMPT.format(nf=NOT_FOUND, hist=hist, ctx=numbered(ctx) if ctx else "(없음)", q=question)
    return llm.chat([{"role": "user", "content": prompt}], max_tokens=max_tokens)


REFUSAL_KEYS = ("찾을수없", "확인할수없", "나와있지않", "언급되어있지않", "포함되어있지않")


#------------------------------------------------------------------
# 거절 답변인지 판별
#=> "찾을 수 없습니다" 류 문구가 있으면 거절로 본다 (A3 재시도 조건, 평가의 거절 집계).
#   띄어쓰기·따옴표 차이를 없애고 비교한다.
#
# -in: ans = 모델 답
#
# -out: refused = True/False
# -out: error = 없음
#------------------------------------------------------------------
def is_refusal(ans: str) -> bool:
    a = re.sub(r"[\s|\"'“”‘’·]", "", ans or "")
    nf = re.sub(r"[\s|\"'“”‘’·]", "", NOT_FOUND)
    return nf in a or any(k in a for k in REFUSAL_KEYS)


#------------------------------------------------------------------
# 한자 비율 (A4)
#=> 답 글자 중 한자(중국어)가 차지하는 비율. 한글+한자 글자 수를 분모로 한다.
#   Qwen 이 가끔 중국어로 답하는 것을 잡으려는 것이다 (2단계 평가에서 B004).
#
# -in: text = 답 문자열
#
# -out: ratio = 0~1 (한글·한자가 하나도 없으면 0)
# -out: error = 없음
#------------------------------------------------------------------
def han_ratio(text: str) -> float:
    han = len(re.findall(r"[一-鿿]", text or ""))
    hangul = len(re.findall(r"[가-힣]", text or ""))
    return han / (han + hangul) if han + hangul else 0.0


# "1천만", "2백만" 같은 겹친 단위를 먼저 맞춘다 (뒤의 "만" 까지 곱해야 값이 맞다)
NUM_RE = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*(천만|백만|억|만|천|백)?")
UNIT = {"천만": 10_000_000, "백만": 1_000_000, "억": 100_000_000, "만": 10_000, "천": 1_000, "백": 100, None: 1}


#------------------------------------------------------------------
# 글에 나온 숫자 값 모으기 (A2)
#=> "50만원", "500,000원", "3일" 같은 숫자를 값으로 바꿔 모은다.
#    1) 근거 번호 [1] 은 숫자가 아니므로 먼저 지운다
#    2) 쉼표를 지우고, 뒤에 만·억·천·백(겹친 "천만" 포함)이 붙으면 곱한다 ("50만" → 500000, "1천만" → 10000000)
#    3) 단위를 곱하기 전 값도 넣는다 (근거에 "50" 만 있고 답이 "50만원" 인 경우를 놓치지 않게)
#
# -in: text = 문자열
#
# -out: (vals, found) = 비교용 값 집합, [(글에 쓰인 모양, 값)] 목록
# -out: error = 없음
#------------------------------------------------------------------
def numbers_in(text: str) -> tuple[set, list]:
    text = re.sub(r"\[\d+\]", " ", text or "")
    vals, found = set(), []
    for m in NUM_RE.finditer(text):
        try:
            raw = float(m.group(1).replace(",", "").rstrip("."))
        except ValueError:  # "1,,2" 같은 깨진 모양은 건너뛴다
            continue
        v = raw * UNIT[m.group(2)]
        vals.update({raw, v})
        found.append((m.group(0).strip(), v))
    return vals, found


#------------------------------------------------------------------
# 근거에 없는 숫자 찾기 (A2)
#=> 답에 쓴 숫자 중 근거·질문·이전 대화 어디에도 없는 것을 돌려준다.
#   계산해서 나온 숫자(예: 50만 + 20만 = 70만)일 수도, 잘못 옮긴 숫자일 수도 있다.
#   0·1 같은 작은 수는 "1회", "1부" 처럼 흔해 오탐이 많으므로 보지 않는다.
#
# -in: answer = 모델 답
# -in: sources = 근거 청크 본문·질문·이전 대화 문자열 목록
#
# -out: nums = 근거에 없는 숫자의 글 모양 목록 (중복 제거, 나온 순서)
# -out: error = 없음
#------------------------------------------------------------------
def unsupported_numbers(answer: str, sources: list[str]) -> list[str]:
    known, _ = numbers_in("\n".join(sources))
    _, found = numbers_in(answer)
    out = []
    for shown, v in found:
        if v > 1 and v not in known and shown not in out:
            out.append(shown)
    return out


# A4 는 앞 답을 보여 주고 고치라고 하면 다시 중국어로 답했다 (B004 시험). 원래 질문 끝에 이 줄을 붙여 새로 답하게 한다
KOREAN_SUFFIX = "\n\n(반드시 한국어로 답하라.)"
FIX_REFUSAL = ("근거 [{n}] 에 질문과 관련된 내용이 있다. 근거를 다시 읽고 근거에 있는 내용으로 답하라. "
               "근거에 정말로 답이 없을 때만 정확히 \"{nf}\" 라고 답한다. 3문장 이내, 근거 번호를 붙인다.")
FIX_NUMBERS = ("답에 쓴 숫자 {nums} 는 근거에 그대로 나오지 않는다. 근거에 나온 숫자만으로 계산식을 한 줄 적고 "
               "(예: 50만원 + 20만원 = 70만원), 그 결과로 답을 다시 써라. 계산이 필요 없는 질문이면 근거의 숫자를 그대로 옮겨라. "
               "3문장 이내, 근거 번호를 붙인다.")


#------------------------------------------------------------------
# 답변 생성 + 고치기 (4단계 A1~A4, 설계서 10장)
#=> 기본 답을 만든 뒤, 걸리는 문제가 있으면 Qwen 에 한 번씩만 더 묻는다. 어느 고치기를 켤지는 [answer] 설정.
#    1) A1: thinking=True 면 생각 모드로 답한다 (추론 문항 판단은 호출하는 쪽에서 decide 로 한다).
#           생각하다 토큰 한도에 걸려 답이 비면 생각 모드 없이 다시 답한다
#    2) A4: 한자 비율이 max_han_ratio 를 넘으면 원래 프롬프트 끝에 "반드시 한국어로" 를 붙여 새로 답하게 한다
#    3) A3: 충분성 p_suff ≥ retry_refusal_min_suff 인데 거절하면, 가장 관련 높은 근거 번호를 짚어 다시 답하게 한다
#    4) A2: 답에 근거에 없는 숫자가 있으면 근거 숫자로 계산식을 쓰고 다시 답하게 한다
#    A2·A3 은 앞 답을 대화로 이어 붙여 묻는다 (모델이 자기 답의 어디가 문제인지 보게).
#    A2 가 거절로 바뀌면 앞 답을 둔다 (숫자를 고치라 했는데 답을 포기하면 더 나쁘다)
#
# -in: llm = LLMClient
# -in: question = 현재 질문
# -in: ctx = 근거 청크 목록 (관련성 순)
# -in: history = 이전 대화
# -in: acfg = cfg["answer"] dict (없으면 모두 끔)
# -in: max_tokens = 답 최대 토큰
# -in: p_suff = decide 충분성 확률 (A3 조건)
# -in: rel = 근거 청크별 관련성 확률 목록 (A3 에서 짚을 번호, 없으면 1번)
# -in: thinking = 생각 모드로 답할지 (A1)
#
# -out: (answer, info) = 최종 답, {"latency"(모든 호출 합), "fixes": [{"fix","before","latency",…}], "thinking", …}
# -out: error = LLM 서버 실패 시 RuntimeError (고치기 단계 실패도 전파)
#------------------------------------------------------------------
def answer_with_fixes(llm, question: str, ctx: list[dict], history: list[dict] | None, acfg: dict | None,
                      max_tokens: int = 400, p_suff: float = 0.0, rel: list[float] | None = None,
                      thinking: bool = False) -> tuple[str, dict]:
    acfg = acfg or {}
    hist = f"[이전 대화]\n{history_text(history)}\n\n" if history else ""
    prompt = ANSWER_PROMPT.format(nf=NOT_FOUND, hist=hist, ctx=numbered(ctx) if ctx else "(없음)", q=question)
    msgs = [{"role": "user", "content": prompt}]
    fixes = []
    if thinking:
        # 생각 토큰도 max_tokens 에 들어가므로 한도를 따로 크게 준다
        ans, info = llm.chat(msgs, max_tokens=acfg.get("thinking_max_tokens", 4000), thinking=True)
        if not ans.strip():
            fixes.append({"fix": "thinking_fallback", "finish_reason": info.get("finish_reason"),
                          "latency": info["latency"]})
            ans, info2 = llm.chat(msgs, max_tokens=max_tokens, thinking=False)
            info = {**info2, "latency": info["latency"] + info2["latency"]}
    else:
        ans, info = llm.chat(msgs, max_tokens=max_tokens)
    total = info["latency"]

    # 한 번 더 묻는다: follow 가 있으면 앞 답 뒤에 지시를 잇고, 없으면 new_msgs 로 새로 묻는다
    #   빈 답이 오거나 keep_if_refusal 인데 거절로 바뀌면 앞 답을 그대로 둔다
    def retry(name, follow=None, new_msgs=None, keep_if_refusal=False, **extra):
        nonlocal ans, total
        m = new_msgs or msgs + [{"role": "assistant", "content": ans}, {"role": "user", "content": follow}]
        new, inf = llm.chat(m, max_tokens=max_tokens, thinking=False)
        kept = not new.strip() or (keep_if_refusal and is_refusal(new) and not is_refusal(ans))
        fixes.append({"fix": name, "before": ans, "latency": inf["latency"], "kept_before": kept,
                      **({"rejected": new} if kept and new.strip() else {}), **extra})
        total += inf["latency"]
        if not kept:
            ans = new

    if acfg.get("force_korean") and han_ratio(ans) > acfg.get("max_han_ratio", 0.2):
        retry("A4_korean", new_msgs=[{"role": "user", "content": prompt + KOREAN_SUFFIX}],
              han_ratio=round(han_ratio(ans), 3))
    if (acfg.get("retry_refusal") and ctx and is_refusal(ans)
            and p_suff >= acfg.get("retry_refusal_min_suff", 0.8)):
        # 관련성이 가장 높은 근거 번호를 짚어 준다 (번호는 1부터)
        n = 1 + max(range(len(ctx)), key=lambda i: rel[i]) if rel and len(rel) == len(ctx) else 1
        retry("A3_refusal", FIX_REFUSAL.format(n=n, nf=NOT_FOUND), p_suff=round(p_suff, 3), pointed=n)
    if acfg.get("verify_numbers") and ctx and not is_refusal(ans):
        # 질문·이전 대화에 나온 숫자는 사용자가 준 것이므로 근거에 있는 것으로 친다
        srcs = [c["text"] for c in ctx] + [question] + [f"{h['q']} {h['a']}" for h in history or []]
        nums = unsupported_numbers(ans, srcs)
        if nums:
            retry("A2_numbers", FIX_NUMBERS.format(nums=", ".join(nums)), keep_if_refusal=True, numbers=nums)
    return ans, {**info, "latency": total, "fixes": fixes, "thinking": thinking}
