# -*- coding: utf-8 -*-
"""검색 → decide 판단 → 재검색 → 답변 루프."""
import time

from .answer import NOT_FOUND, answer_with_fixes, judge_question
from .clients import DecideBusy, DecideUnavailable, LLMUnavailable
from .hybrid import reset_notes, take_notes


#------------------------------------------------------------------
# 답변 단계 (4단계 A1~A4)
#=> 고른 근거로 Qwen 답을 만든다. 설정 [answer] 에 따라 답을 고친다 (answer.answer_with_fixes).
#    1) A1: thinking_on_reasoning 이면 decide 에 "계산·조건 비교가 필요한 질문인가?" 를 묻고,
#           확률이 thinking_min_prob 이상이면 생각 모드로 답한다. degraded 이거나 decide 가 안 되면 묻지 않는다
#    2) A2~A4 는 answer_with_fixes 안에서 답을 본 뒤 필요할 때만 한 번 더 묻는다
#    평가에서 같은 근거로 답만 바꿔 비교할 수 있도록 run_agentic 밖으로 뺐다 (eval\answer_variants.py)
#
# -in: llm = LLMClient
# -in: judge = DecideJudge (A1 판단용, None 이면 A1 을 건너뛴다)
# -in: cfg = 설정 dict
# -in: question = 현재 질문
# -in: qj = 판단용 질문 (이전 대화 포함)
# -in: ctx = 근거 청크 목록
# -in: history = 이전 대화
# -in: p_suff = 충분성 확률 (A3)
# -in: rel = 근거별 관련성 확률 목록 (A3)
# -in: degraded = decide 장애 이유 (있으면 A1 을 묻지 않는다)
#
# -out: (answer, info) = 답, 사용량·고치기 기록 (+ "p_reason")
# -out: error = LLM 실패 시 RuntimeError 등 전파 (호출하는 쪽에서 LLMUnavailable 로 바꾼다)
#------------------------------------------------------------------
def answer_step(llm, judge, cfg: dict, question: str, qj: str, ctx: list[dict], history, p_suff: float,
                rel: list[float] | None, degraded=None) -> tuple[str, dict]:
    acfg = cfg.get("answer", {})
    p_reason, thinking = None, False
    if acfg.get("thinking_on_reasoning") and judge is not None and not degraded:
        try:
            p_reason = judge.needs_reasoning(qj)
            thinking = p_reason >= acfg.get("thinking_min_prob", 0.3)
        except (DecideUnavailable, DecideBusy):  # 판단을 못 하면 생각 모드 없이 답한다 (답을 막지 않는다)
            p_reason = None
    ans, info = answer_with_fixes(llm, question, ctx, history, acfg, cfg["llm"].get("max_tokens", 400),
                                  p_suff, rel, thinking)
    info["p_reason"] = p_reason
    return ans, info


#------------------------------------------------------------------
# 다음 라운드로 갈지 판단 (재검색 상한)
#=> 재검색을 무한정 하지 않도록, 다음 라운드를 시작하기 전에 상한을 확인한다.
#    1) 근거가 충분하면 멈춘다
#    2) 마지막 라운드면 멈춘다 (max_rounds)
#    3) 다음 라운드에 쓸 decide 호출(라우팅 1 + 새 청크 최대 N + 충분성 1)을 더하면 상한을 넘으면 멈춘다
#    4) 판단 시간이 예산을 넘었으면 멈춘다 (decide 대기열이 밀릴 때 응답 시간을 묶어 둔다)
#
# -in: r = 방금 끝난 라운드 번호 (0 부터)
# -in: p_suff = 방금 잰 충분성 확률
# -in: stats = 판단 호출 통계
# -in: cfg = 설정 dict
#
# -out: reason = 멈출 이유 문자열, 계속하면 None
# -out: error = 없음
#------------------------------------------------------------------
def stop_reason(r: int, p_suff: float, stats, cfg: dict) -> str | None:
    A, R = cfg["agent"], cfg["retrieval"]
    if p_suff >= A["suff_threshold"]:
        return "sufficient"
    if r >= A["max_rounds"] - 1:
        return "max_rounds"
    if stats.calls + R["research_new_max"] + 2 > A.get("max_decide_calls", 10 ** 9):
        return "max_decide_calls"
    if stats.seconds >= A.get("judge_budget_sec", float("inf")):
        return "judge_budget"
    return None


#------------------------------------------------------------------
# 에이전트 실행 (질문 하나)
#=> 검색 → decide 재정렬·충분성 → 부족하면 재검색 → Qwen 답변.
#    검색은 index 가 맡는다: BM25 만, 또는 BM25+벡터(RRF)+리랭커 (7단계, hybrid.make_retriever).
#    벡터·리랭커를 건너뛴 검색이 있으면 degraded 에 그 이유를 붙인다 (설계서 8.2).
#    1) 검색 질의는 최근 이전 질문을 이어 붙여 만들고, decide 에는 이전 대화까지 보여 준다
#    2) 라운드마다 새로 찾은 청크만 decide 로 판단한다 (이미 판단한 청크는 다시 묻지 않는다)
#    3) 관련성 상위 context_k 개로 충분성을 재고, stop_reason 이 멈추라고 하면 답변으로 간다
#    4) 아니면 재검색 계획(라우팅·PRF·제목)대로 다시 검색한다
#    5) decide 를 쓸 수 없으면(DecideUnavailable) 판단 없이 답한다 (설계서 8.2, degraded)
#       - 이미 판단한 청크가 있으면 그 순위 상위 context_k 개, 없으면 1라운드 BM25 상위 context_k 개
#    6) 근거가 0개면 LLM 을 부르지 않고 "찾을 수 없습니다"
#
# -in: question = 사용자 질문
# -in: index = 검색 엔진 (MemoryEngine·OpenSearchEngine, 또는 그것을 감싼 HybridEngine·RerankEngine)
# -in: judge = DecideJudge
# -in: llm = LLMClient
# -in: cfg = 설정 dict
# -in: history = 이전 대화 [{"q","a"}, …] (기본 없음)
# -in: answer = False 면 답변 생성을 건너뛴다 (검색·판단만 평가할 때, 기본 True)
# -in: acl = 사용자 권한 그룹 목록 (None = 필터 없음, 사용자 연결은 2.5단계)
#
# -out: trace = {"answer","context","rounds","sufficient","stop","query","timing","judge_stats","answer_info","degraded",
#                 "fixes"(4단계 답 고치기 기록),"thinking","p_reason"}
# -out: error = LLM 실패 시 LLMUnavailable(근거 목록 포함), decide 대기열 가득 차면 DecideBusy, 검색 엔진 오류는 전파
#              (decide 실패는 예외로 내보내지 않고 degraded 로 답한다)
#------------------------------------------------------------------
def run_agentic(question: str, index, judge, llm, cfg: dict, history: list[dict] | None = None,
                answer: bool = True, acl=None) -> dict:
    R, A = cfg["retrieval"], cfg["agent"]
    history = history or []
    judge.new_question()  # 질문마다 통계·문서 후보 캐시를 새로 시작한다
    reset_notes()  # 이 요청의 벡터·리랭커 시간과 건너뛴 이유를 새로 모은다 (7단계)
    t0 = time.perf_counter()
    t_ret = 0.0
    rel: dict[str, float] = {}        # chunk_id → 관련성 확률
    order: dict[str, int] = {}        # chunk_id → 처음 본 순서 (동점일 때 BM25 순서 유지)
    by_id: dict[str, dict] = {}
    qj = judge_question(question, history)
    query = judge.contextualize(question, history)
    rounds, plan, p_suff, ctx, stop = [], [(query, None)], 0.0, [], None
    degraded, first_bm25 = None, []
    try:
        ctx, p_suff, stop, t_ret = _loop(index, judge, cfg, acl, qj, query, plan, rel, order, by_id, rounds,
                                         first_bm25)
    except DecideUnavailable as e:
        # decide 없이도 답은 해야 한다: 판단해 둔 것이 있으면 그 순서, 없으면 BM25 순서
        degraded = str(e)[:300]
        stop = "degraded"
        p_suff = 0.0
        if rel:
            ranked = sorted(rel, key=lambda cid: (-rel[cid], order[cid]))
            ctx = [by_id[cid] for cid in ranked[: R["context_k"]]]
        else:
            ctx = first_bm25[: R["context_k"]]
        rounds.append({"round": len(rounds), "error": degraded, "context": [c["chunk_id"] for c in ctx]})
        t_ret = sum(rr.get("t_retrieval", 0.0) for rr in rounds)
    # 벡터·리랭커를 건너뛴 검색이 있었으면 판단은 했어도 degraded 로 알린다 (설계서 8.2)
    notes = take_notes()
    if notes["skipped"]:
        degraded = "; ".join(([degraded] if degraded else []) + notes["skipped"])[:500]
    sufficient = p_suff >= A["suff_threshold"]
    ctx_ids = [c["chunk_id"] for c in ctx]
    if not answer:
        # 검색·판단만 볼 때는 LLM 을 부르지 않는다 (LLM 서버 없이도 회귀 시험 가능)
        ans, ainfo = "", {"latency": 0.0}
    elif not ctx:
        # 근거가 하나도 없으면 LLM 이 지어낼 여지만 있으므로 부르지 않는다 (설계서 8.2)
        ans, ainfo = NOT_FOUND, {"latency": 0.0, "skipped": "no_context"}
    elif not sufficient and not degraded and A.get("refuse_when_insufficient"):
        ans, ainfo = NOT_FOUND, {"latency": 0.0}
    else:
        try:
            ans, ainfo = answer_step(llm, judge, cfg, question, qj, ctx, history, p_suff,
                                     [rel.get(c["chunk_id"], 0.0) for c in ctx], degraded)
        except Exception as e:  # 답은 못 썼어도 찾은 근거는 돌려줄 수 있게 함께 넘긴다
            raise LLMUnavailable(f"{type(e).__name__}: {str(e)[:300]}", ctx_ids) from e
    for rr in rounds:
        rr.pop("t_retrieval", None)
    return {"answer": ans, "context": ctx_ids, "rounds": rounds, "sufficient": sufficient,
            "stop": stop, "query": query, "degraded": degraded,
            "fixes": ainfo.get("fixes", []), "thinking": ainfo.get("thinking", False), "p_reason": ainfo.get("p_reason"),
            "timing": {"retrieval": round(t_ret, 3), "judge": round(judge.stats.seconds, 3),
                       "answer": round(ainfo["latency"], 3), "total": round(time.perf_counter() - t0, 3),
                       # retrieval 안에 든 질의 임베딩·벡터 검색·리랭커 시간 (쓰지 않으면 0)
                       "embed": round(notes["embed"], 3), "vector": round(notes["vector"], 3),
                       "rerank": round(notes["rerank"], 3)},
            "judge_stats": judge.stats.as_dict(), "answer_info": ainfo}


#------------------------------------------------------------------
# 검색·판단 라운드 반복
#=> run_agentic 의 본체. decide 가 실패하면 DecideUnavailable 이 밖으로 나가고, 그때까지 채운 rel·rounds 는 남는다.
#
# -in: index, judge, cfg, acl, qj, query = run_agentic 과 같음
# -in: plan = 첫 검색 계획 [(질의, None)]
# -in: rel, order, by_id = 관련성·순서·청크 (이 함수가 채운다)
# -in: rounds = 라운드 기록 목록 (이 함수가 채운다, 라운드마다 "t_retrieval" 임시 키 포함)
# -in: first_bm25 = 1라운드 BM25 결과를 담을 목록 (degraded 대체용, 이 함수가 채운다)
#
# -out: (ctx, p_suff, stop, t_ret) = 근거, 충분성, 멈춤 이유, 검색 시간 합
# -out: error = decide 실패 시 DecideUnavailable, 대기열 가득 차면 DecideBusy, 엔진 오류 전파
#------------------------------------------------------------------
def _loop(index, judge, cfg, acl, qj, query, plan, rel, order, by_id, rounds, first_bm25):
    R, A = cfg["retrieval"], cfg["agent"]
    t_ret, p_suff, ctx, stop = 0.0, 0.0, [], None
    for r in range(A["max_rounds"]):
        ts = time.perf_counter()
        new = []
        k = R["bm25_top_k"] if r == 0 else R["research_top_k"]
        # 질의들을 한 번에 보낸다(_msearch). 1단계는 질의마다 앞 질의가 고른 청크를 빼고 k 개씩 받았으므로,
        # 같은 결과가 되도록 질의마다 k × 질의 수 만큼 받은 뒤 앞에서 고른 것을 건너뛰며 k 개씩 고른다.
        hits = index.search_many([(q, k * len(plan), docs, set(rel)) for q, docs in plan], acl)
        for hs in hits:
            taken = {x["chunk_id"] for x in new}
            new.extend([c for c, _ in hs if c["chunk_id"] not in taken][:k])
        new = new[: R["bm25_top_k"] if r == 0 else R["research_new_max"]]
        dt = time.perf_counter() - ts
        t_ret += dt
        if r == 0:
            first_bm25.extend(new)
        # 판단 중 decide 가 실패해도 검색 시간은 남도록 라운드 기록을 먼저 열어 둔다
        rec = {"round": r, "queries": [[q, sorted(d) if d else None] for q, d in plan],
               "new": [c["chunk_id"] for c in new], "t_retrieval": dt}
        rounds.append(rec)
        for c, p in zip(new, judge.rerank(qj, new)):
            rel[c["chunk_id"]] = p
            order[c["chunk_id"]] = len(order)
            by_id[c["chunk_id"]] = c
        ranked = sorted(rel, key=lambda cid: (-rel[cid], order[cid]))
        ctx = [by_id[cid] for cid in ranked[: R["context_k"]]]
        p_suff = judge.sufficient(qj, ctx)
        rec.update({"rel": {c["chunk_id"]: round(rel[c["chunk_id"]], 3) for c in new},
                    "context": [c["chunk_id"] for c in ctx], "p_suff": round(p_suff, 3)})
        stop = stop_reason(r, p_suff, judge.stats, cfg)
        if stop:
            break
        # 근거가 부족하고 상한에도 안 걸렸으면 재검색 계획을 세운다
        plan, info = judge.research_plan(qj, query, ctx, acl)
        rec["plan_info"] = info
    return ctx, p_suff, stop, t_ret
