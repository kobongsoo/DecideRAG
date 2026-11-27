# -*- coding: utf-8 -*-
"""답변 단계만 바꿔 비교하기 (4단계 A1~A4).

검색·판단 결과(근거 청크, 충분성, 관련성)는 judge-only 실행 파일에서 그대로 가져오고, 답만 새로 만든다.
그래서 변형끼리 차이는 답변 단계 설정에서만 생긴다.

  python eval\\answer_variants.py --name base
  python eval\\answer_variants.py --name a2 --set verify_numbers=true
  python eval\\answer_variants.py --name a1 --set thinking_on_reasoning=true      # decide 필요
  → eval\\runs\\ans_<name>.jsonl (끊겨도 이어서)
"""
import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.stdout.reconfigure(encoding="utf-8")

from decide_rag.agent import answer_step  # noqa: E402
from decide_rag.answer import is_refusal, judge_question  # noqa: E402
from decide_rag.config import load_config  # noqa: E402


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
# --set 값 해석
#=> "verify_numbers=true" 를 ("verify_numbers", True) 로. 숫자·true/false 는 TOML 처럼 바꾼다.
#
# -in: kv = "키=값" 문자열
#
# -out: (key, value)
# -out: error = "=" 가 없으면 ValueError
#------------------------------------------------------------------
def parse_set(kv: str):
    k, v = kv.split("=", 1)
    low = v.strip().lower()
    if low in ("true", "false"):
        return k.strip(), low == "true"
    try:
        return k.strip(), float(v) if "." in v else int(v)
    except ValueError:  # 숫자가 아니면 문자열 그대로
        return k.strip(), v


#------------------------------------------------------------------
# 판단 기록에서 충분성·관련성 꺼내기
#=> judge-only 실행의 rounds 에서 마지막 충분성과, 라운드마다 잰 청크 관련성을 합친다.
#
# -in: rec = judge-only 실행 한 줄
#
# -out: (p_suff, rel) = 마지막 충분성 확률(없으면 0), {chunk_id: 관련성}
# -out: error = 없음
#------------------------------------------------------------------
def judged(rec: dict) -> tuple[float, dict]:
    p_suff, rel = 0.0, {}
    for r in rec.get("rounds", []):
        rel.update(r.get("rel") or {})
        if "p_suff" in r:
            p_suff = r["p_suff"]
    return p_suff, rel


#------------------------------------------------------------------
# 변형 하나 실행
#=> 평가셋 문항마다 judge-only 실행의 근거로 답을 만들고 결과를 한 줄씩 쓴다.
#    1) 이미 쓴 문항은 건너뛴다 (오류 줄은 지우고 다시)
#    2) 근거가 0개면 서비스와 같이 LLM 을 부르지 않고 NOT_FOUND
#    3) A1 을 켰을 때만 decide 판단기를 만든다 (나머지는 LLM 만 쓴다)
#
# -in: name = 변형 이름 (결과 파일 ans_<name>.jsonl)
# -in: sets = [answer] 덮어쓸 값 dict
# -in: base_run = judge-only 실행 파일
# -in: limit = 문항 수 제한 (0 = 전부)
#
# -out: 없음 (eval\runs\ans_<name>.jsonl)
# -out: error = 근거 파일·평가셋이 없으면 FileNotFoundError, 문항별 오류는 기록 후 계속
#------------------------------------------------------------------
def run(name: str, sets: dict, base_run: str, limit: int = 0):
    cfg = load_config()
    cfg["answer"] = {**cfg.get("answer", {}), **sets}
    from decide_rag.search_engine import make_engine
    from decide_rag.clients import LLMClient
    index, llm = make_engine(cfg), LLMClient(cfg["llm"])
    judge = None
    if cfg["answer"].get("thinking_on_reasoning"):
        from decide_rag.cli import load_system
        judge = load_system(cfg)["judge"]
    items = read_jsonl(HERE / "evalset_v2.jsonl")[: limit or None]
    base = {r["id"]: r for r in read_jsonl(base_run) if "error" not in r}
    path = HERE / "runs" / f"ans_{name}.jsonl"
    done = set()
    if path.exists():
        keep = [l for l in path.read_text(encoding="utf-8").splitlines() if l.strip() and "error" not in json.loads(l)]
        done = {json.loads(l)["id"] for l in keep}
        path.write_text("".join(l + "\n" for l in keep), encoding="utf-8")
    t0 = time.perf_counter()
    for it in items:
        if it["id"] in done:
            continue
        b = base[it["id"]]
        hist = it.get("history") or []
        p_suff, rel = judged(b)
        try:
            ctx = index.get_chunks(b["context"])
            if not ctx:
                ans, info = "제공된 문서에서 해당 내용을 찾을 수 없습니다.", {"latency": 0.0, "fixes": []}
            else:
                if judge is not None:
                    judge.new_question()
                ans, info = answer_step(llm, judge, cfg, it["question"], judge_question(it["question"], hist), ctx,
                                        hist, p_suff, [rel.get(c["chunk_id"], 0.0) for c in ctx])
            rec = {"id": it["id"], "variant": name, "answer": ans, "context": b["context"], "p_suff": p_suff,
                   "refused": is_refusal(ans), "fixes": info.get("fixes", []), "thinking": info.get("thinking", False),
                   "p_reason": info.get("p_reason"), "timing": {"answer": round(info["latency"], 3)},
                   "completion_tokens": info.get("completion_tokens")}
        except Exception as e:  # 서버 일시 오류는 기록하고 다음 실행 때 다시 한다
            rec = {"id": it["id"], "variant": name, "error": f"{type(e).__name__}: {e}"}
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fx = ",".join(x["fix"] for x in rec.get("fixes", []))
        print(it["id"], rec.get("timing", {}).get("answer", rec.get("error")), fx, flush=True)
    print(f"완료 {time.perf_counter() - t0:.0f}s → {path}", flush=True)


#------------------------------------------------------------------
# 명령 실행
#=> --name, --set(여러 번), --base-run, --limit 을 받아 run 을 부른다.
#
# -in: 없음 (sys.argv)
#
# -out: 없음
# -out: error = 인자 오류 시 종료
#------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--set", action="append", default=[], help="[answer] 값 덮어쓰기, 예: verify_numbers=true")
    ap.add_argument("--base-run", default=str(HERE / "runs" / "stage3_os_judgeonly.jsonl"))
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    run(a.name, dict(parse_set(x) for x in a.set), a.base_run, a.limit)


if __name__ == "__main__":
    main()
