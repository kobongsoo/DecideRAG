# -*- coding: utf-8 -*-
"""답 고치기 A1~A4 시험 (가짜 LLM, 서버 불필요).

  python tests\\test_answer_fixes.py

각 고치기가 조건이 맞을 때만 한 번 더 묻는지, 꺼져 있으면 묻지 않는지 본다.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.stdout.reconfigure(encoding="utf-8")

from decide_rag.answer import NOT_FOUND, answer_with_fixes, han_ratio, unsupported_numbers  # noqa: E402

CTX = [{"chunk_id": "d1#000", "title": "경조사 규정", "path": [], "text": "| 본인 결혼 | 5 | 500,000 | 화환 |"},
       {"chunk_id": "d1#001", "title": "경조사 규정", "path": [], "text": "자녀 결혼 200,000원"}]


#------------------------------------------------------------------
# 가짜 LLM
#=> 정해 둔 답을 차례로 돌려주고, 받은 호출(생각 모드·마지막 지시)을 기록한다.
#
# -필드: replies = 돌려줄 답 목록 (차례로)
# -필드: calls = [(thinking, 마지막 user 메시지)] 기록
#------------------------------------------------------------------
class ScriptLLM:
    #------------------------------------------------------------------
    # 만들기
    #=> 답 목록을 받는다.
    #
    # -in: replies = 답 문자열 목록
    #
    # -out: 없음
    # -out: error = 없음
    #------------------------------------------------------------------
    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    #------------------------------------------------------------------
    # 채팅 흉내
    #=> LLMClient.chat 과 같은 모양. 답 목록에서 하나씩 꺼낸다.
    #
    # -in: messages, max_tokens, temperature, thinking = LLMClient.chat 과 같음
    #
    # -out: (text, info)
    # -out: error = 답이 모자라면 IndexError
    #------------------------------------------------------------------
    def chat(self, messages, max_tokens=512, temperature=0.0, thinking=None):
        self.calls.append((thinking, messages[-1]["content"]))
        return self.replies.pop(0), {"latency": 0.1, "finish_reason": "stop"}


#------------------------------------------------------------------
# 시험 하나 실행
#=> 가짜 LLM 으로 answer_with_fixes 를 부르고 (최종 답, 고치기 이름 목록, 호출 수) 를 돌려준다.
#
# -in: replies = 가짜 답 목록
# -in: acfg = [answer] 설정
# -in: kw = answer_with_fixes 의 나머지 인자 (p_suff, rel, thinking)
#
# -out: (ans, fixes, llm) = 최종 답, 고치기 이름 목록, 가짜 LLM (호출 기록)
# -out: error = 예외 전파
#------------------------------------------------------------------
def run(replies, acfg, **kw):
    llm = ScriptLLM(replies)
    ans, info = answer_with_fixes(llm, "결혼하면 얼마 줘?", CTX, None, acfg, 400, **kw)
    return ans, [f["fix"] for f in info["fixes"]], llm


#------------------------------------------------------------------
# 전체 시험
#=> 고치기마다 걸리는 경우·안 걸리는 경우를 확인하고 결과를 출력한다.
#
# -in: 없음
#
# -out: 없음 (실패가 있으면 종료 코드 1)
# -out: error = 없음
#------------------------------------------------------------------
def main():
    ok = []

    # 결과 한 줄 기록
    def check(name, cond):
        ok.append(cond)
        print(("통과" if cond else "실패"), name)

    on = {"force_korean": True, "retry_refusal": True, "verify_numbers": True}
    # 기본: 모두 끄면 한 번만 묻는다
    ans, fx, llm = run(["본인 결혼 시 50만원 [1]"], {})
    check("모두 끔 → 1회 호출", len(llm.calls) == 1 and fx == [])
    # 걸릴 것이 없으면 켜 있어도 1회
    ans, fx, llm = run(["본인 결혼 시 50만원, 자녀 20만원 [1][2]"], on, p_suff=0.9)
    check("문제 없음 → 1회 (만원 단위 숫자도 근거와 맞음)", len(llm.calls) == 1 and fx == [])
    # A4
    ans, fx, llm = run(["本人结婚 50万", "본인 결혼 시 50만원 [1]"], on, p_suff=0.9)
    check("A4 중국어 → 한국어로 다시", fx == ["A4_korean"] and ans.startswith("본인"))
    check("A4 는 원래 프롬프트 + '반드시 한국어로' 로 새로 묻는다", llm.calls[1][1].endswith("(반드시 한국어로 답하라.)")
          and "[근거]" in llm.calls[1][1])
    # A2 가 거절로 바뀌면 앞 답을 둔다
    ans, fx, llm = run(["합계 70만원 [1]", NOT_FOUND], on, p_suff=0.9)
    check("A2 고친 답이 거절 → 앞 답 유지", ans == "합계 70만원 [1]")
    # A3: 충분성 높고 거절 → 다시, 짚은 번호는 관련성 최고(2번)
    ans, fx, llm = run([NOT_FOUND, "자녀 결혼 20만원 [2]"], on, p_suff=0.9, rel=[0.2, 0.8])
    check("A3 거절 + p_suff 0.9 → 다시 (근거 [2] 짚음)", fx == ["A3_refusal"] and "근거 [2]" in llm.calls[1][1])
    # A3: 충분성 낮으면 거절 그대로
    ans, fx, llm = run([NOT_FOUND], on, p_suff=0.5)
    check("A3 거절 + p_suff 0.5 → 그대로 거절", fx == [] and ans == NOT_FOUND)
    # A2: 근거에 없는 숫자
    ans, fx, llm = run(["합계 70만원 [1]", "50만원 + 20만원 = 70만원 [1][2]"], on, p_suff=0.9)
    check("A2 근거에 없는 70만 → 계산식으로 다시", fx == ["A2_numbers"] and "70만" in llm.calls[1][1])
    # A1: 생각 모드, 답이 비면 생각 없이 다시
    ans, fx, llm = run(["", "본인 결혼 시 50만원 [1]"], {}, thinking=True)
    check("A1 생각 모드 빈 답 → 생각 없이 다시", llm.calls[0][0] is True and llm.calls[1][0] is False
          and fx == ["thinking_fallback"])
    # 고치기가 빈 답을 내면 앞 답을 둔다
    ans, fx, llm = run(["本人结婚", ""], on)
    check("고치기 빈 답 → 앞 답 유지", ans == "本人结婚")
    # 보조 함수
    check("han_ratio", han_ratio("이사 등与公司间交易等") > 0.5 and han_ratio("본인 결혼") == 0)
    check("천만 단위", unsupported_numbers("1천만 원 이하", ["1천만원 이하"]) == [])
    print(f"{sum(ok)}/{len(ok)} 통과")
    sys.exit(0 if all(ok) else 1)


if __name__ == "__main__":
    main()
