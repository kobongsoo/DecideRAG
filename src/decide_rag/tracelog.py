# -*- coding: utf-8 -*-
"""질문별 trace 로그 (설계서 7.1·12.4).

logs\\trace_YYYYMMDD.jsonl 에 질문마다 한 줄:
  시각, 사용자 ID·그룹, 질문, 이전 대화 수, 답, 근거 chunk_id, 라운드 기록, 멈춤 이유, 시간, decide 호출·토큰
사용자 질문 원문이 들어가므로 보관 기간은 13장 열린 질문 4 로 정한다. API 키·비밀번호는 넣지 않는다.
"""
import json
import threading
import time
from pathlib import Path

_lock = threading.Lock()


#------------------------------------------------------------------
# 오늘 trace 파일 경로
#=> 날짜별로 파일을 나눠 보관·삭제를 쉽게 한다.
#
# -in: log_dir = 로그 폴더
# -in: now = 시각 (기본 지금, time.struct_time)
#
# -out: path = logs\trace_YYYYMMDD.jsonl
# -out: error = 없음
#------------------------------------------------------------------
def trace_path(log_dir: str, now=None) -> Path:
    return Path(log_dir) / f"trace_{time.strftime('%Y%m%d', now or time.localtime())}.jsonl"


#------------------------------------------------------------------
# trace 한 줄 쓰기
#=> run_agentic 결과에서 남길 항목만 골라 JSON 한 줄로 덧붙인다 (degraded·답 고치기 기록 포함). 여러 요청이 동시에 써도 줄이 섞이지 않게 잠근다.
#
# -in: log_dir = 로그 폴더
# -in: user = {"id","groups"} 요청한 사용자
# -in: question = 질문 원문
# -in: history = 이전 대화 목록
# -in: tr = run_agentic 결과 (없으면 None — 오류 요청)
# -in: error = 오류 메시지 (없으면 None)
# -in: source = "api" / "cli"
#
# -out: rec = 기록한 dict
# -out: error = 쓰기 실패 시 OSError 전파
#------------------------------------------------------------------
def write_trace(log_dir: str, user: dict, question: str, history: list, tr: dict | None,
                error: str | None = None, source: str = "api") -> dict:
    rec = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "source": source,
           "user_id": user.get("id"), "groups": user.get("groups"),
           "question": question, "history_turns": len(history or []),
           # 이전 대화 글도 남긴다 (멀티턴 질문을 사람 검수 평가셋으로 옮길 때 필요, tools\review_set.py)
           "history": history or []}
    if tr:
        rec.update({k: tr.get(k) for k in ("answer", "context", "sufficient", "stop", "query", "rounds",
                                          "timing", "judge_stats", "degraded", "fixes", "thinking")})
        # 고치기(A3·A4)가 바꾸기 전 답도 fixes[].before 에 남는다 (어떤 답을 왜 고쳤는지 나중에 볼 수 있게)
    if error:
        rec["error"] = error
    line = json.dumps(rec, ensure_ascii=False) + "\n"
    p = trace_path(log_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    with _lock:
        with p.open("a", encoding="utf-8") as f:
            f.write(line)
    return rec
