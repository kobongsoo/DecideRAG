# -*- coding: utf-8 -*-
"""부하 시험 (6단계): 실제 서비스(/ask)에 동시 사용자 N명이 질문을 연달아 보낸다.

  $env:DECIDE_GATEWAY_TOKEN = "<서비스와 같은 토큰>"
  python tools\\load_test.py --url http://127.0.0.1:8091 --users 1 2 4 8 --per-level 32

- 사용자마다 답을 받으면 곧바로 다음 질문을 보낸다 (쉬는 시간 없음 → 가장 바쁜 경우).
- 질문은 평가셋 100문항을 섞어 차례로 쓴다 (멀티턴은 이전 대화 포함).
- 결과: eval\\runs\\load_test.json (단계별 요약 + 요청별 기록)
"""
import argparse
import json
import os
import random
import sys
import threading
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.stdout.reconfigure(encoding="utf-8")


#------------------------------------------------------------------
# 분위수
#=> 정렬한 값에서 p 위치(내림)의 값 (run_eval 과 같은 방식).
#
# -in: xs = 숫자 목록
# -in: p = 0~1
#
# -out: v = 분위수 (빈 목록이면 None)
# -out: error = 없음
#------------------------------------------------------------------
def quantile(xs: list[float], p: float):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else None


#------------------------------------------------------------------
# 요청 하나 보내기
#=> /ask 에 게이트웨이 헤더를 붙여 묻고, 걸린 시간과 서버가 알려 준 시간·decide 대기를 기록한다.
#
# -in: url = 서비스 주소
# -in: token = 게이트웨이 토큰
# -in: user = 사용자 번호 (X-User-Id)
# -in: item = 평가셋 문항
# -in: timeout = 응답 대기 상한 (초)
#
# -out: rec = {"id","user","status","sec",…(성공이면 "timing","wait","calls","degraded","rounds")}
# -out: error = 없음 (연결 오류도 rec["error"] 로)
#------------------------------------------------------------------
def ask_once(url: str, token: str, user: int, item: dict, timeout: float) -> dict:
    body = {"question": item["question"], "history": item.get("history") or [], "debug": True}
    hdr = {"X-Gateway-Token": token, "X-User-Id": f"load{user}", "X-User-Groups": "all"}
    t0 = time.perf_counter()
    rec = {"id": item["id"], "user": user, "start": round(t0, 3)}
    try:
        r = requests.post(url.rstrip("/") + "/ask", json=body, headers=hdr, timeout=timeout)
        rec["status"] = r.status_code
        if r.status_code == 200:
            d = r.json()
            js = (d.get("trace") or {}).get("judge_stats") or {}
            rec.update({"timing": d["timing"], "wait": js.get("wait"), "calls": js.get("calls"),
                        "degraded": d.get("degraded"), "rounds": d.get("rounds")})
    except Exception as e:  # 연결 끊김·시간 초과도 결과로 남긴다
        rec["status"], rec["error"] = None, f"{type(e).__name__}: {str(e)[:120]}"
    rec["sec"] = round(time.perf_counter() - t0, 3)
    return rec


#------------------------------------------------------------------
# 한 단계 실행 (동시 사용자 n명)
#=> 사용자 n명이 같은 질문 줄에서 하나씩 꺼내 total 개를 다 보낼 때까지 연달아 묻는다.
#
# -in: url, token = 서비스 주소, 토큰
# -in: n = 동시 사용자 수
# -in: questions = 섞은 문항 목록 (돌려 쓴다)
# -in: total = 이 단계에서 보낼 요청 수
# -in: timeout = 요청 대기 상한
#
# -out: (summary, recs) = 단계 요약 dict, 요청별 기록 목록
# -out: error = 없음
#------------------------------------------------------------------
def run_level(url: str, token: str, n: int, questions: list[dict], total: int, timeout: float):
    lock = threading.Lock()
    nxt = {"i": 0}
    recs = []

    # 사용자 한 명: 질문 줄이 빌 때까지 묻는다
    def user(u):
        while True:
            with lock:
                if nxt["i"] >= total:
                    return
                item = questions[nxt["i"] % len(questions)]
                nxt["i"] += 1
            r = ask_once(url, token, u, item, timeout)
            with lock:
                recs.append(r)

    t0 = time.perf_counter()
    ths = [threading.Thread(target=user, args=(u,)) for u in range(n)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    wall = time.perf_counter() - t0
    ok = [r for r in recs if r.get("status") == 200]
    lat = [r["sec"] for r in ok]
    mean = lambda xs: round(sum(xs) / len(xs), 3) if xs else None  # noqa: E731
    s = {"users": n, "requests": len(recs), "ok": len(ok), "wall_sec": round(wall, 1),
         "per_min": round(len(ok) / wall * 60, 1),
         "status": {str(k): sum(1 for r in recs if r.get("status") == k) for k in sorted({r.get("status") for r in recs}, key=str)},
         "degraded": sum(1 for r in ok if r.get("degraded")),
         "latency_p50": quantile(lat, 0.5), "latency_p90": quantile(lat, 0.9), "latency_max": max(lat) if lat else None,
         "latency_mean": mean(lat),
         "judge_mean": mean([r["timing"]["judge"] for r in ok]), "answer_mean": mean([r["timing"]["answer"] for r in ok]),
         "decide_wait_mean": mean([r["wait"] for r in ok if r.get("wait") is not None]),
         "decide_wait_p90": quantile([r["wait"] for r in ok if r.get("wait") is not None], 0.9),
         "decide_calls_mean": mean([r["calls"] for r in ok if r.get("calls") is not None])}
    return s, recs


#------------------------------------------------------------------
# 명령 실행
#=> 단계(동시 사용자 수)마다 run_level 을 돌리고 요약을 출력·저장한다.
#
# -in: 없음 (sys.argv, 환경변수 DECIDE_GATEWAY_TOKEN)
#
# -out: 없음 (eval\runs\load_test.json)
# -out: error = 토큰 환경변수가 없으면 SystemExit
#------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8090")
    ap.add_argument("--users", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--per-level", type=int, default=32)
    ap.add_argument("--timeout", type=float, default=180)
    ap.add_argument("--out", default=str(ROOT / "eval" / "runs" / "load_test.json"))
    a = ap.parse_args()
    token = os.environ.get("DECIDE_GATEWAY_TOKEN")
    if not token:
        sys.exit("환경변수 DECIDE_GATEWAY_TOKEN 이 필요합니다 (서비스와 같은 값)")
    items = [json.loads(l) for l in (ROOT / "eval" / "evalset_v2.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    # 단계마다 같은 질문 순서를 쓰려고 씨앗을 고정한다
    random.Random(11).shuffle(items)
    out = {"url": a.url, "per_level": a.per_level, "levels": [], "records": {}}
    for n in a.users:
        s, recs = run_level(a.url, token, n, items, a.per_level, a.timeout)
        out["levels"].append(s)
        out["records"][str(n)] = recs
        print(json.dumps(s, ensure_ascii=False), flush=True)
        Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("→", a.out)


if __name__ == "__main__":
    main()
