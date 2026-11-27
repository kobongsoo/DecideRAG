# -*- coding: utf-8 -*-
"""명령줄 진입점 (python run.py …).

  python run.py build [--full-rebuild]          # 증분 색인: 바뀐 문서만 추출 → 청킹 → 검색 엔진
  python run.py ask "질문" [--history h.json] [--full]
  python run.py eval [--limit N] [--out eval\\runs\\decide.jsonl]
  python run.py summary [runs 파일 …]          # 판정 없이 셀 수 있는 지표 요약
  python run.py serve [--host H --port P]       # HTTP API (service.py)
"""
import argparse
import json
import sys

from .config import load_config


#------------------------------------------------------------------
# 검색 시스템 한 벌 준비
#=> 설정의 검색 엔진(OpenSearch 또는 메모리), LLM 클라이언트, decide 대기열, decide 판단기를 만든다.
#   검색 경로는 [vector] 설정에 따라 BM25 만 / BM25+벡터(RRF)+리랭커 (hybrid.make_retriever).
#   decide 는 대기열(QueuedDecide)을 거친다: 한 번에 한 요청, 실패하면 잠시 차단 (설계서 8.1·8.2).
#   판단기(judge)는 질문마다 상태가 있으므로 동시 요청에서는 new_judge() 로 요청마다 새로 만든다.
#
# -in: cfg = 설정 dict
#
# -out: sys = {"index"(검색 경로),"engine"(BM25 엔진 자체),"llm","decide","judge","new_judge"}
# -out: error = 메모리 엔진인데 청크가 없으면 ZeroDivisionError (먼저 build), OpenSearch 연결 실패는 첫 검색에서
#------------------------------------------------------------------
def load_system(cfg: dict) -> dict:
    from .clients import DecideClient, LLMClient, QueuedDecide
    from .judge import DecideJudge
    from .hybrid import make_retriever
    from .search_engine import make_engine
    engine = make_engine(cfg)
    # [vector] 설정대로 BM25 엔진을 혼합 검색(BM25+벡터)·리랭커로 감싼다 (7단계). 끄면 engine 그대로
    index = make_retriever(cfg, engine)
    d = cfg["decide"]
    decide = QueuedDecide(DecideClient(d), d.get("max_queue", 50), d.get("cooldown_sec", 30))
    clip = d.get("chunk_clip", 1200)

    # 같은 엔진·대기열을 쓰는 판단기를 새로 만든다 (요청마다 하나)
    def new_judge():
        return DecideJudge(decide, index, cfg["agent"], clip)

    return {"index": index, "engine": engine, "llm": LLMClient(cfg["llm"]), "decide": decide, "judge": new_judge(),
            "new_judge": new_judge}


#------------------------------------------------------------------
# 명령 실행
#=> 하위 명령(build/ask/eval/summary/serve)을 골라 실행한다.
#
# -in: argv = 명령줄 인자 목록 (기본 None = sys.argv)
#
# -out: 없음 (결과는 data\, eval\ 폴더와 화면)
# -out: error = 각 단계 예외 전파
#------------------------------------------------------------------
def main(argv: list[str] | None = None):
    # Windows 콘솔(cp949)에서 한글이 깨지지 않게 출력 인코딩을 UTF-8 로
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(prog="run.py")
    ap.add_argument("cmd", choices=["build", "ask", "eval", "summary", "serve"])
    ap.add_argument("args", nargs="*", help="ask: 질문 / summary: runs 파일")
    ap.add_argument("--history", help="ask: 이전 대화 JSON 파일 [{\"q\",\"a\"}, …]")
    ap.add_argument("--full", action="store_true", help="ask: 라운드 trace 전체 출력")
    ap.add_argument("--limit", type=int, default=0, help="eval: 질문 수 제한 (0 = 전부)")
    ap.add_argument("--evalset", help="eval: 평가셋 경로 (기본 eval\\evalset_v2.jsonl)")
    ap.add_argument("--judge-only", action="store_true", help="eval: 답변 생성 없이 검색·판단만")
    ap.add_argument("--out", help="eval: 결과 파일 (기본 eval\\runs\\decide.jsonl)")
    ap.add_argument("--full-rebuild", action="store_true", help="build: 색인을 지우고 모든 문서를 다시 처리")
    ap.add_argument("--groups", help="ask: 이 그룹 권한으로 검색 (쉼표 구분, 공개 그룹은 자동 포함). 없으면 권한 필터 없음(운영자용)")
    ap.add_argument("--host", help="serve: 주소 (기본 [service] host)")
    ap.add_argument("--port", type=int, help="serve: 포트 (기본 [service] port)")
    a = ap.parse_args(argv)
    cfg = load_config()

    if a.cmd == "build":
        from .indexer import build
        print(json.dumps(build(cfg, full=a.full_rebuild), ensure_ascii=False))
    elif a.cmd == "ask":
        from .agent import run_agentic
        if not a.args:
            ap.error("ask 에는 질문이 필요합니다")
        hist = json.loads(open(a.history, encoding="utf-8").read()) if a.history else []
        from .acl import user_acl
        from .tracelog import write_trace
        s = load_system(cfg)
        acl = user_acl(a.groups.split(","), cfg.get("service", {}).get("public_group", "all")) if a.groups else None
        q = " ".join(a.args)
        tr = run_agentic(q, s["index"], s["judge"], s["llm"], cfg, hist, acl=acl)
        write_trace(cfg["paths"]["log_dir"], {"id": "cli", "groups": acl}, q, hist, tr, source="cli")
        if a.full:
            print(json.dumps({k: v for k, v in tr.items() if k != "answer_info"}, ensure_ascii=False, indent=1))
        else:
            print(tr["answer"], "\n")
            # 근거 청크 본문은 엔진에서 ID 로 다시 가져온다 (운영 규모에서는 청크를 메모리에 두지 않는다)
            for i, c in enumerate(s["index"].get_chunks(tr["context"]), 1):
                print(f"[{i}] {c['chunk_id']}  {' > '.join([c['title']] + c['path'])}")
            if tr.get("degraded"):
                print(f"\n[degraded] decide 를 쓸 수 없어 판단 없이 답했습니다: {tr['degraded']}")
            print(f"\n라운드 {len(tr['rounds'])} · 멈춤 {tr['stop']} · 충분 {tr['sufficient']} · "
                  f"decide {tr['judge_stats']['calls']}회 · 시간 {tr['timing']}")
    elif a.cmd == "eval":
        sys.path.insert(0, str(_eval_dir()))
        from run_eval import run_eval
        run_eval(cfg, load_system(cfg), a.limit, a.evalset, a.out, answer=not a.judge_only)
    elif a.cmd == "serve":
        from .service import serve
        serve(cfg, a.host, a.port)
    elif a.cmd == "summary":
        sys.path.insert(0, str(_eval_dir()))
        from run_eval import summarize
        summarize(cfg, a.args)


#------------------------------------------------------------------
# 평가 폴더 경로
#=> 평가 코드(eval\run_eval.py)는 서비스 패키지 밖에 두므로 경로를 따로 구한다.
#
# -in: 없음
#
# -out: path = 프로젝트\eval 폴더
# -out: error = 없음
#------------------------------------------------------------------
def _eval_dir():
    from .config import ROOT
    return ROOT / "eval"
