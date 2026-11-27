# -*- coding: utf-8 -*-
"""검색 구성 비교 (설계서 14.7): BM25+decide / BM25+벡터 / BM25+벡터+리랭커 / BM25+벡터+decide.

  python eval\\compare_configs.py --config hs                       # 평가셋 100문항 → eval\\runs\\ans_cmp_hs.jsonl
  python eval\\compare_configs.py --config hs --mismatch            # 단어 불일치 75문항(답 생성 없음) → eval\\runs\\mm_hs.jsonl
  구성: bm25_decide (지금 서비스) · hs · hs_rr (bge-reranker-base, CPU) · hs_rr_ko (bge-reranker-v2-m3-ko, A30 서버) · hs_decide
        · hs_rr_ko_decide (혼합 검색 → 한국어 리랭커 → decide)
  --embed m3 : 벡터를 A30 서버의 bge-m3 로 (기본 e5). --tag _m3 로 결과 파일을 따로 둔다

- 답변 단계는 모두 같다 (answer_step, [answer] 설정 그대로).
  decide 가 없는 구성은 충분성 확률이 없어 A3(거절 재시도)가 걸리지 않는다. A4 는 같다.
- 정답률은 eval\\ac.py 로 채점한다 (실행 이름 cmp_<구성>).
"""
import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))
sys.stdout.reconfigure(encoding="utf-8")

from decide_rag.agent import answer_step, run_agentic  # noqa: E402
from decide_rag.answer import NOT_FOUND, is_refusal, judge_question  # noqa: E402
from decide_rag.config import load_config  # noqa: E402
from run_eval import evidence_ratio, read_jsonl  # noqa: E402

CONFIGS = ("bm25_decide", "hs", "hs_rr", "hs_rr_ko", "hs_decide", "hs_rr_ko_decide", "service")
# decide 루프(run_agentic)로 도는 구성 (service = 서비스 검색 경로 그대로)
DECIDE_CONFIGS = ("bm25_decide", "hs_decide", "hs_rr_ko_decide", "service")


#------------------------------------------------------------------
# 검색 질의 만들기
#=> 서비스와 같은 규칙: 최근 이전 사용자 질문 history_turns 개를 앞에 이어 붙인다 (DecideJudge.contextualize).
#
# -in: question = 현재 질문
# -in: history = 이전 대화
# -in: turns = 이어 붙일 이전 질문 수 (0 = 전부)
#
# -out: query = 검색 질의
# -out: error = 없음
#------------------------------------------------------------------
def make_query(question: str, history: list[dict], turns: int) -> str:
    prev = [h["q"] for h in history or []]
    if turns:
        prev = prev[-turns:]
    return " ".join(prev + [question])


#------------------------------------------------------------------
# 시스템 준비
#=> 구성에 맞는 검색 엔진·판단기·리랭커를 만든다.
#
# -in: cfg = 설정 dict
# -in: name = 구성 이름
# -in: embed = 벡터 임베딩 "e5" (이 PC ONNX) 또는 "m3" (A30 서버 bge-m3)
#
# -out: s = {"index","llm","judge","reranker"}
# -out: error = 모델·서버 준비 실패 시 예외 전파
#------------------------------------------------------------------
def build(cfg: dict, name: str, embed: str = "e5") -> dict:
    from decide_rag.cli import load_system
    from decide_rag.judge import DecideJudge
    s = load_system(cfg)
    if name == "service":
        # 서비스 검색 경로 그대로 (config.toml [vector] 설정, 7단계)
        return s
    # 비교 구성은 감싸기 전 BM25 엔진에서 시작한다
    s["index"] = s["engine"]
    if name != "bm25_decide":
        from decide_rag.hybrid import make_hybrid
        s["index"] = make_hybrid(cfg, s["index"], embed)
    if name == "hs_rr":
        from decide_rag.hybrid import Reranker
        s["reranker"] = Reranker(cfg["vector"]["reranker_dir"])
    if name in ("hs_rr_ko", "hs_rr_ko_decide"):
        # 한국어 리랭커는 A30 서버(vLLM /v1/rerank)에 있다
        from decide_rag.hybrid import HttpReranker
        s["reranker"] = HttpReranker(cfg["vector"]["reranker_url"], cfg["vector"].get("reranker_model", "bge-reranker-v2-m3-ko"))
    if name == "hs_rr_ko_decide":
        # 혼합 검색 → 리랭커가 다시 줄 세운 후보를 decide 가 판단한다 (재검색 질의도 같은 길)
        from decide_rag.hybrid import RerankEngine
        s["index"] = RerankEngine(s["index"], s["reranker"], cfg["vector"].get("rerank_candidates", 20))
    if name != "bm25_decide":
        # 판단기도 바뀐 엔진으로 검색하게 다시 만든다 (재검색·라우팅 후보 모두)
        s["judge"] = DecideJudge(s["decide"], s["index"], cfg["agent"], cfg["decide"].get("chunk_clip", 1200))
    return s


#------------------------------------------------------------------
# 문항 하나 실행
#=> 구성대로 근거를 고르고, answer=True 면 답까지 만든다.
#    - decide 구성: run_agentic 그대로
#    - hs: 혼합 검색 상위 context_k 개
#    - hs_rr: 혼합 검색 상위 rerank_candidates 개 → 리랭커 → 상위 context_k 개
#
# -in: s = build 결과
# -in: cfg = 설정 dict
# -in: name = 구성 이름
# -in: it = 문항
# -in: answer = 답 생성 여부
#
# -out: rec = {"answer","context","timing","fixes",…}
# -out: error = 서버 오류 전파 (호출하는 쪽이 기록)
#------------------------------------------------------------------
def run_one(s: dict, cfg: dict, name: str, it: dict, answer: bool) -> dict:
    hist = it.get("history") or []
    if name in DECIDE_CONFIGS:
        # 리랭크 엔진이면 이 질문에 리랭커가 쓴 시간을 따로 센다
        tr = run_agentic(it["question"], s["index"], s["judge"], s["llm"], cfg, hist, answer)
        if tr.get("degraded"):
            raise RuntimeError(f"degraded: {tr['degraded']}")
        # 리랭커·벡터 시간은 run_agentic 이 timing 에 넣는다 (요청별 기록)
        timing = tr["timing"]
        return {"answer": tr["answer"], "context": tr["context"], "timing": timing,
                "fixes": tr.get("fixes", []), "thinking": tr.get("thinking", False),
                "rounds": len(tr["rounds"]), "decide_calls": tr["judge_stats"]["calls"]}
    R = cfg["retrieval"]
    t0 = time.perf_counter()
    q = make_query(it["question"], hist, cfg["agent"].get("history_turns", 2))
    if name == "hs":
        ctx = [c for c, _ in s["index"].search(q, R["context_k"])]
        t_rr = 0.0
    else:
        cands = [c for c, _ in s["index"].search(q, cfg["vector"].get("rerank_candidates", 20))]
        t1 = time.perf_counter()
        sc = s["reranker"].score(q, [c["text"] for c in cands])
        t_rr = time.perf_counter() - t1
        ctx = [c for _, c in sorted(zip(sc, cands), key=lambda x: -x[0])][: R["context_k"]]
    t_ret = time.perf_counter() - t0
    ans, info = "", {"latency": 0.0, "fixes": []}
    if answer:
        if not ctx:
            ans = NOT_FOUND
        else:
            # 판단기가 없으므로 A1·A3 은 걸리지 않는다 (p_suff 0)
            ans, info = answer_step(s["llm"], None, cfg, it["question"], judge_question(it["question"], hist),
                                    ctx, hist, 0.0, None)
    return {"answer": ans, "context": [c["chunk_id"] for c in ctx], "fixes": info.get("fixes", []),
            "thinking": False, "rounds": 1, "decide_calls": 0,
            "timing": {"retrieval": round(t_ret, 3), "rerank": round(t_rr, 3), "judge": 0.0,
                       "answer": round(info["latency"], 3), "total": round(t_ret + info["latency"], 3)}}


#------------------------------------------------------------------
# 단어 불일치 문항 준비
#=> evalset_mismatch 의 질문에 원래 평가셋 같은 ID 문항의 근거 인용문을 붙인다 (근거 포함률 계산용).
#    정답 문서는 1단계 옛 ID(D01…) 라서 제목으로 지금 문서 ID 에 맞춘다.
#
# -in: cfg = 설정 dict
#
# -out: items = [{"id","question","history":[],"evidence","answerable","gold_docs"}]
# -out: error = 파일이 없으면 FileNotFoundError
#------------------------------------------------------------------
def mismatch_items(cfg: dict) -> list[dict]:
    v2 = {x["id"]: x for x in read_jsonl(HERE / "evalset_v2.jsonl")}
    data = Path(cfg["paths"]["data_dir"])
    legacy = {}
    for l in (data / "legacy_stage1" / "chunks.jsonl").read_text(encoding="utf-8").splitlines():
        c = json.loads(l)
        legacy[c["doc_id"]] = c["title"]
    by_title = {e["title"]: d for d, e in json.loads((data / "manifest.json").read_text(encoding="utf-8")).items()}
    out = []
    for x in read_jsonl(HERE / "evalset_mismatch.jsonl"):
        src = v2[x["id"]]
        out.append({"id": x["id"], "type": x["type"], "question": x["question"], "history": [],
                    "evidence": src["evidence"], "answerable": True,
                    "gold_docs": [by_title[legacy[d]] for d in x["doc_ids"] if legacy.get(d) in by_title]})
    return out


#------------------------------------------------------------------
# 구성 하나 실행
#=> 문항마다 run_one 결과를 한 줄씩 쓴다 (끊겨도 이어서, 오류 줄은 다음 실행 때 다시).
#
# -in: name = 구성 이름
# -in: mismatch = True 면 단어 불일치 75문항 (답 생성 없이 근거만)
# -in: limit = 문항 수 제한 (0 = 전부)
# -in: embed = 벡터 임베딩 "e5" / "m3"
# -in: tag = 결과 파일 이름 뒤에 붙일 글 (같은 구성을 다른 조건으로 다시 돌릴 때, 기본 "")
#
# -out: 없음 (eval\runs\ans_cmp_<구성><tag>.jsonl 또는 mm_<구성><tag>.jsonl)
# -out: error = 준비 실패 시 예외 전파
#------------------------------------------------------------------
def run(name: str, mismatch: bool, limit: int = 0, embed: str = "e5", tag: str = ""):
    cfg = load_config()
    s = build(cfg, name, embed)
    items = mismatch_items(cfg) if mismatch else read_jsonl(HERE / "evalset_v2.jsonl")
    items = items[: limit or None]
    path = HERE / "runs" / (f"mm_{name}{tag}.jsonl" if mismatch else f"ans_cmp_{name}{tag}.jsonl")
    done = set()
    if path.exists():
        keep = [l for l in path.read_text(encoding="utf-8").splitlines() if l.strip() and "error" not in json.loads(l)]
        done = {json.loads(l)["id"] for l in keep}
        path.write_text("".join(l + "\n" for l in keep), encoding="utf-8")
    t0 = time.perf_counter()
    for it in items:
        if it["id"] in done:
            continue
        try:
            r = run_one(s, cfg, name, it, answer=not mismatch)
            ctx = s["index"].get_chunks(r["context"])
            rec = {"id": it["id"], "variant": name, **r, "evidence_ratio": evidence_ratio(it, ctx),
                   "refused": is_refusal(r["answer"]) if not mismatch else None}
            if mismatch:
                rec["gold_hit"] = any(c["doc_id"] in it["gold_docs"] for c in ctx)
        except Exception as e:  # 서버 일시 오류는 기록하고 다음 실행 때 다시
            rec = {"id": it["id"], "variant": name, "error": f"{type(e).__name__}: {e}"}
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(it["id"], rec.get("evidence_ratio", rec.get("error")), rec.get("timing", {}).get("total"), flush=True)
    print(f"완료 {time.perf_counter() - t0:.0f}s → {path}", flush=True)


#------------------------------------------------------------------
# 명령 실행
#=> --config, --mismatch, --limit, --embed, --tag 를 받아 run 을 부른다.
#
# -in: 없음 (sys.argv)
#
# -out: 없음
# -out: error = 인자 오류 시 종료
#------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, choices=CONFIGS)
    ap.add_argument("--mismatch", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--embed", default="e5", choices=["e5", "m3"], help="벡터 임베딩: e5 (이 PC) / m3 (A30 bge-m3)")
    ap.add_argument("--tag", default="", help="결과 파일 이름 꼬리 (예: _m3)")
    a = ap.parse_args()
    run(a.config, a.mismatch, a.limit, a.embed, a.tag)


if __name__ == "__main__":
    main()
