# -*- coding: utf-8 -*-
"""평가 실행·요약 (판정 없이 셀 수 있는 지표).

RAGAS 4지표·Answer Correctness 채점은 프로토타입(D:\\Project\\Q&A\\AgenticRAG)의 ragas.py ·
answer_correctness.py 로 한다. 여기서는 근거 포함률·거절·시간·decide 호출 수만 센다.
"""
import json
import re
import time
from collections import Counter
from pathlib import Path

from decide_rag.agent import run_agentic
from decide_rag.answer import is_refusal  # 답 고치기(A3)와 같은 거절 판별을 쓴다

HERE = Path(__file__).resolve().parent


#------------------------------------------------------------------
# 비교용 정규화
#=> 공백·표 기호·따옴표를 지워 "근거 문장이 본문에 그대로 있는가" 를 띄어쓰기 차이 없이 비교한다.
#
# -in: s = 문자열
#
# -out: norm = 정규화한 문자열
# -out: error = 없음
#------------------------------------------------------------------
def norm(s: str) -> str:
    return re.sub(r"[\s|\"'“”‘’·]", "", s or "")


#------------------------------------------------------------------
# 근거 포함 비율
#=> 정답 근거 인용문 중 최종 근거 청크 본문 안에 들어 있는 비율.
#   추론 문항은 근거가 2개 이상이라 일부만 찾으면 0~1 사이 값이 된다.
#
# -in: item = 평가 항목
# -in: ctx_chunks = 최종 근거 청크 목록 (엔진에서 get_chunks 로 가져온 것)
#
# -out: ratio = 0~1 (문서에 없는 질문은 None)
# -out: error = 없음
#------------------------------------------------------------------
def evidence_ratio(item: dict, ctx_chunks: list[dict]):
    if not item["answerable"] or not item["evidence"]:
        return None
    # 청크 ID 는 2단계부터 해시라 평가셋의 옛 ID(D01#000)와 다르다 → 인용문이 본문에 있는지로만 판단
    body = [norm(c["body"]) for c in ctx_chunks]
    found = sum(any(norm(e["quote"]) in b for b in body) for e in item["evidence"])
    return found / len(item["evidence"])


#------------------------------------------------------------------
# jsonl 읽기
#=> 빈 줄을 건너뛰고 dict 목록으로.
#
# -in: p = 경로
#
# -out: rows = dict 목록
# -out: error = 파일이 없으면 FileNotFoundError
#------------------------------------------------------------------
def read_jsonl(p) -> list[dict]:
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


#------------------------------------------------------------------
# 평가 실행
#=> 평가셋 전체를 돌려 결과 파일에 한 줄씩 저장한다 (중간에 끊겨도 이어서).
#    1) 이미 저장된 질문 ID 는 건너뛰고, 오류 줄은 지운 뒤 다시 시도
#    2) 멀티턴 문항은 history 를 같이 넘긴다
#    3) trace + 근거 포함 비율 + 거절 여부를 저장
#
# -in: cfg = 설정 dict
# -in: s = load_system 결과 (index, llm, judge)
# -in: limit = 질문 수 제한 (0 = 전부)
# -in: evalset = 평가셋 경로 (None = eval/evalset_v2.jsonl)
# -in: out = 결과 파일 (None = eval/runs/decide.jsonl)
# -in: answer = False 면 답변 생성 없이 검색·판단만 (거절 지표는 None, 기본 True)
#
# -out: 없음 (파일 저장)
# -out: error = 평가셋이 없으면 FileNotFoundError, 질문별 서버 오류는 기록하고 계속
#------------------------------------------------------------------
def run_eval(cfg: dict, s: dict, limit: int = 0, evalset: str | None = None, out: str | None = None,
             answer: bool = True):
    items = read_jsonl(evalset or HERE / "evalset_v2.jsonl")
    if limit:
        items = items[:limit]
    path = Path(out) if out else HERE / "runs" / "decide.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if path.exists():
        keep = [l for l in path.read_text(encoding="utf-8").splitlines() if l.strip() and "error" not in json.loads(l)]
        done = {json.loads(l)["id"] for l in keep}
        # 오류 줄은 지우고 다시 시도한다
        path.write_text("".join(l + "\n" for l in keep), encoding="utf-8")
    t0 = time.perf_counter()
    for it in items:
        if it["id"] in done:
            continue
        try:
            tr = run_agentic(it["question"], s["index"], s["judge"], s["llm"], cfg, it.get("history") or [], answer)
            if tr.get("degraded"):
                # decide 장애로 판단 없이 낸 답은 평가 점수에 섞지 않는다 → 오류로 남기고 다음 실행 때 다시
                raise RuntimeError(f"degraded: {tr['degraded']}")
            rec = {"id": it["id"], "config": "decide", **{k: v for k, v in tr.items() if k != "answer_info"},
                   "evidence_ratio": evidence_ratio(it, s["index"].get_chunks(tr["context"])), "refused": is_refusal(tr["answer"]) if answer else None}
        except Exception as e:  # 서버 일시 오류는 기록 후 계속
            rec = {"id": it["id"], "config": "decide", "error": f"{type(e).__name__}: {e}"}
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(it["id"], rec.get("evidence_ratio", rec.get("error")), rec.get("timing", {}).get("total"), flush=True)
    print(f"완료 {time.perf_counter() - t0:.0f}s → {path}", flush=True)


#------------------------------------------------------------------
# 분위수
#=> 정렬한 값에서 p 위치(내림)의 값을 고른다 (프로토타입 보고서와 같은 방식).
#
# -in: xs = 숫자 목록
# -in: p = 0~1
#
# -out: v = 분위수 값 (빈 목록이면 None)
# -out: error = 없음
#------------------------------------------------------------------
def quantile(xs: list[float], p: float):
    if not xs:
        return None
    xs = sorted(xs)
    return xs[int(p * (len(xs) - 1))]


#------------------------------------------------------------------
# 결과 파일 요약
#=> runs 파일마다 근거 포함률·거절·시간·decide 호출·라운드 분포를 한 표로 출력한다.
#   여러 파일을 주면 나란히 비교한다 (예: 프로토타입 decide.jsonl 과 이번 결과).
#
# -in: cfg = 설정 dict (쓰지 않음, 명령 형식을 맞추기 위함)
# -in: files = runs 파일 목록 (비면 eval/runs/decide.jsonl)
#
# -out: 없음 (화면 출력)
# -out: error = 파일이 없으면 FileNotFoundError
#------------------------------------------------------------------
def summarize(cfg: dict, files: list[str]):
    items = {x["id"]: x for x in read_jsonl(HERE / "evalset_v2.jsonl")}
    files = files or [str(HERE / "runs" / "decide.jsonl")]
    rows = {}
    for f in files:
        rs = [r for r in read_jsonl(f) if "error" not in r]
        ans = [r for r in rs if items[r["id"]]["answerable"]]
        una = [r for r in rs if not items[r["id"]]["answerable"]]
        m = lambda xs: (sum(xs) / len(xs) if xs else None) if None not in xs else None  # 평균 (빈 목록·답변 없음은 None)
        row = {
            "문항": len(rs),
            "오류": sum("error" in r for r in read_jsonl(f)),
            "근거 포함률": m([r["evidence_ratio"] for r in ans]),
        }
        for t in ("single", "multi", "reasoning"):
            row[f"  └ {t}"] = m([r["evidence_ratio"] for r in ans if items[r["id"]]["type"] == t])
        row.update({
            "답 있는데 거절": m([r["refused"] for r in ans]),
            "없는 질문 거절": m([r["refused"] for r in una]),
            "전체 시간 평균": m([r["timing"]["total"] for r in rs]),
            "전체 시간 p90": quantile([r["timing"]["total"] for r in rs], 0.9),
            "판단 시간 평균": m([r["timing"]["judge"] for r in rs]),
            "판단 시간 p90": quantile([r["timing"]["judge"] for r in rs], 0.9),
            "답변 시간 평균": m([r["timing"]["answer"] for r in rs]),
            "decide 호출 평균": m([r["judge_stats"]["calls"] for r in rs]),
            "decide 호출 최대": max((r["judge_stats"]["calls"] for r in rs), default=None),
            "재검색률": m([len(r["rounds"]) > 1 for r in rs]),
            "라운드 분포": dict(sorted(Counter(len(r["rounds"]) for r in rs).items())),
            "멈춤 이유": dict(Counter(r.get("stop", "-") for r in rs)),
        })
        rows[f] = row
    names = [Path(f).parent.parent.name + "/" + Path(f).name if len(files) > 1 else Path(f).name for f in files]
    keys = list(next(iter(rows.values())))
    fmt = lambda v: "-" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))  # 값 표시
    w = max(len(k) for k in keys) + 2
    print("".ljust(w) + " | ".join(n[-34:].ljust(34) for n in names))
    for k in keys:
        print(k.ljust(w) + " | ".join(fmt(rows[f][k]).ljust(34) for f in files))
