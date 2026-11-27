# -*- coding: utf-8 -*-
"""HTTP API (설계서 7.1, 12.4).

  POST /ask            {"question", "history": [{"q","a"}], "debug": false}
  GET  /health         decide·LLM·검색 엔진 상태
  POST /index/rebuild  {"full": false}   (관리자 그룹만, 백그라운드 실행)
  GET  /index/status   마지막 색인 결과 (관리자 그룹만)

사용자·그룹은 인증 게이트웨이가 넣은 헤더에서만 읽는다. 요청 본문에 그룹을 넣어도 쓰지 않는다.
  X-Gateway-Token : 환경변수 [service] gateway_token_env (기본 DECIDE_GATEWAY_TOKEN) 와 같아야 한다
  X-User-Id       : 사용자 ID
  X-User-Groups   : 쉼표로 구분한 그룹 목록
토큰 환경변수가 없으면 dev_mode = true 일 때만 설정의 dev_user / dev_groups 로 동작한다 (헤더는 무시).

실행: python run.py serve   (uvicorn, [service] host·port)
"""
import hmac
import os
import threading
import time

import requests
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from .acl import user_acl
from .agent import run_agentic
from .clients import DecideBusy, LLMUnavailable
from .tracelog import write_trace


#------------------------------------------------------------------
# 이전 대화 한 턴
#=> 클라이언트가 보내는 대화 기록. 서버는 대화를 저장하지 않는다.
#
# -필드: q = 사용자 질문
# -필드: a = 시스템 답
#------------------------------------------------------------------
class Turn(BaseModel):
    model_config = ConfigDict(extra="ignore")
    q: str = Field(max_length=2000)
    a: str = Field(default="", max_length=4000)


#------------------------------------------------------------------
# /ask 요청 본문
#=> 모르는 필드(예: 클라이언트가 넣은 groups)는 무시한다. 권한은 헤더로만 받는다.
#
# -필드: question = 질문 (1~1000자)
# -필드: history = 이전 대화 (최근 20턴까지)
# -필드: debug = true 면 라운드 trace 를 함께 돌려준다
#------------------------------------------------------------------
class AskBody(BaseModel):
    model_config = ConfigDict(extra="ignore")
    question: str = Field(min_length=1, max_length=1000)
    history: list[Turn] = Field(default_factory=list, max_length=20)
    debug: bool = False


#------------------------------------------------------------------
# /index/rebuild 요청 본문
#
# -필드: full = true 면 색인을 지우고 전부 다시
#------------------------------------------------------------------
class RebuildBody(BaseModel):
    model_config = ConfigDict(extra="ignore")
    full: bool = False


#------------------------------------------------------------------
# 서버 상태 묶음
#=> 설정, 검색 시스템, 잠금, 마지막 색인 결과를 한곳에 둔다.
#
# -필드: cfg = 설정 dict
# -필드: sys = load_system 결과 {"index","llm","judge"}
# -필드: reload_lock = 메모리 엔진을 다시 만들 때 sys 를 바꿔 끼우는 잠금
#   (decide 는 sys["decide"] 대기열이 차례로 보내므로 /ask 자체는 동시에 처리한다 — 3단계)
# -필드: build_lock = 색인을 동시에 두 번 돌리지 않게 하는 잠금
# -필드: build_status = 마지막 색인 상태 {"state","started","finished","summary","error"}
#------------------------------------------------------------------
class State:
    #------------------------------------------------------------------
    # 상태 만들기
    #=> 검색 시스템을 한 번 준비한다 (OpenSearch 연결·Kiwi 준비).
    #
    # -in: cfg = 설정 dict
    #
    # -out: 없음
    # -out: error = 검색 엔진 준비 실패 시 예외 전파
    #------------------------------------------------------------------
    def __init__(self, cfg: dict):
        from .cli import load_system
        self.cfg = cfg
        self.sys = load_system(cfg)
        self.reload_lock = threading.Lock()
        self.build_lock = threading.Lock()
        self.build_status: dict = {"state": "idle"}


#------------------------------------------------------------------
# 앱 만들기
#=> FastAPI 앱에 경로를 붙인다. 시험에서 설정을 바꿔 끼울 수 있게 함수로 만든다.
#
# -in: cfg = 설정 dict
#
# -out: app = FastAPI 앱 (app.state.s = State)
# -out: error = 검색 엔진 준비 실패 시 예외 전파
#------------------------------------------------------------------
def create_app(cfg: dict) -> FastAPI:
    sc = cfg.get("service", {})
    public = sc.get("public_group", "all")
    admin = sc.get("admin_group", "admin")
    token_env = sc.get("gateway_token_env", "DECIDE_GATEWAY_TOKEN")
    st = State(cfg)
    app = FastAPI(title="DecideRAG", version="2.5")
    app.state.s = st

    #------------------------------------------------------------------
    # 요청한 사용자 확인
    #=> 게이트웨이 토큰이 맞을 때만 헤더의 사용자·그룹을 믿는다.
    #    1) 토큰 환경변수가 있으면: X-Gateway-Token 비교(시간 일정 비교) → X-User-Id 필수
    #    2) 없으면: dev_mode 일 때만 설정의 dev_user / dev_groups (헤더 무시), 아니면 503
    #    3) 그룹에 공개 그룹("all")을 더한다
    #
    # -in: request = FastAPI 요청
    #
    # -out: user = {"id","groups"(검색에 쓸 acl)}
    # -out: error = 토큰 불일치·사용자 없음 401, 인증 미설정 503
    #------------------------------------------------------------------
    def current_user(request: Request) -> dict:
        expected = os.environ.get(token_env, "")
        if expected:
            got = request.headers.get("x-gateway-token", "")
            if not hmac.compare_digest(got.encode(), expected.encode()):
                raise HTTPException(401, "게이트웨이 토큰이 맞지 않습니다")
            uid = request.headers.get("x-user-id", "").strip()
            if not uid:
                raise HTTPException(401, "X-User-Id 가 없습니다")
            groups = request.headers.get("x-user-groups", "").split(",")
        elif sc.get("dev_mode"):
            uid, groups = sc.get("dev_user", "dev"), list(sc.get("dev_groups", []))
        else:
            raise HTTPException(503, f"인증이 설정되지 않았습니다 (환경변수 {token_env} 또는 [service] dev_mode)")
        return {"id": uid, "groups": user_acl(groups, public)}

    #------------------------------------------------------------------
    # 관리자 확인
    #=> 사용자 그룹에 관리자 그룹이 있어야 색인 명령을 쓸 수 있다.
    #
    # -in: user = current_user 결과
    #
    # -out: user = 그대로
    # -out: error = 관리자가 아니면 403
    #------------------------------------------------------------------
    def admin_user(user: dict = Depends(current_user)) -> dict:
        if admin not in user["groups"]:
            raise HTTPException(403, "관리자 그룹만 쓸 수 있습니다")
        return user

    #------------------------------------------------------------------
    # 질문에 답하기 (POST /ask)
    #=> 사용자 권한으로 검색·판단·답변을 하고 trace 를 남긴다 (설계서 8.2 장애 처리).
    #    1) 판단기는 요청마다 새로 만든다 (질문별 상태가 섞이지 않게). decide 요청은 대기열이 차례로 보낸다
    #    2) decide 장애 → 에이전트가 판단 없이 답하고 degraded: true (200)
    #    3) decide 대기열 가득 → 503 (잠시 후 다시)
    #    4) LLM 장애 → 502 + 찾아 둔 근거 목록 (sources)
    #    5) 검색 엔진 장애 등 그 밖의 오류 → 502
    #
    # -in: body = AskBody
    # -in: user = current_user 결과
    #
    # -out: {"answer","sources","sufficient","stop","rounds","timing","degraded","degraded_reason"(+ "trace")}
    # -out: error = 인증 401/503, 대기열 503, LLM·엔진 오류 502
    #------------------------------------------------------------------
    @app.post("/ask")
    def ask(body: AskBody, user: dict = Depends(current_user)):
        hist = [t.model_dump() for t in body.history]
        s = st.sys
        log_dir = st.cfg["paths"]["log_dir"]
        trace_on = sc.get("trace", True)

        # 근거 chunk_id → 출처 정보 (제목·헤딩 경로·쪽)
        def sources(ids):
            return [{"chunk_id": c["chunk_id"], "title": c["title"], "path": c["path"], "page": c.get("page")}
                    for c in s["index"].get_chunks(ids)]

        try:
            tr = run_agentic(body.question, s["index"], s["new_judge"](), s["llm"], st.cfg, hist,
                             acl=user["groups"])
            src = sources(tr["context"])
        except DecideBusy as e:
            if trace_on:
                write_trace(log_dir, user, body.question, hist, None, f"DecideBusy: {e}")
            raise HTTPException(503, "요청이 많습니다. 잠시 후 다시 시도해 주세요")
        except LLMUnavailable as e:
            # 답은 못 썼지만 찾은 근거는 알려 준다 (설계서 8.2)
            if trace_on:
                write_trace(log_dir, user, body.question, hist, None, f"LLMUnavailable: {e}")
            try:
                src = sources(e.context)
            except Exception:  # 엔진까지 안 되면 근거 없이
                src = []
            return JSONResponse(status_code=502, content={"error": f"답변 생성 실패: {str(e)[:300]}",
                                                          "sources": src})
        except Exception as e:  # 검색 엔진 등 그 밖의 오류
            if trace_on:
                write_trace(log_dir, user, body.question, hist, None, f"{type(e).__name__}: {e}")
            raise HTTPException(502, f"처리 중 오류: {type(e).__name__}: {str(e)[:300]}")
        if trace_on:
            write_trace(log_dir, user, body.question, hist, tr)
        out = {"answer": tr["answer"], "sources": src, "sufficient": tr["sufficient"], "stop": tr["stop"],
               "rounds": len(tr["rounds"]), "timing": tr["timing"], "degraded": bool(tr.get("degraded")),
               "degraded_reason": tr.get("degraded")}
        if body.debug:
            out["trace"] = {"query": tr["query"], "rounds": tr["rounds"], "judge_stats": tr["judge_stats"]}
        return out

    #------------------------------------------------------------------
    # 상태 확인 (GET /health)
    #=> decide·LLM 서버 응답, 검색 엔진의 문서·청크 수, 파싱 실패 수, 인증 방식을 돌려준다. 인증 없이 부를 수 있다.
    #
    # -in: 없음
    #
    # -out: {"ok","decide"(+ 대기열·차단기),"llm","embed","rerank"(켰을 때, + 차단기),"engine","index","auth","build"}
    # -out: error = 없음 (각 항목 실패는 값에 기록)
    #------------------------------------------------------------------
    @app.get("/health")
    def health():
        from .indexer import load_manifest
        out: dict = {}
        for name, url in (("decide", st.cfg["decide"]["url"].rstrip("/") + "/health"),
                          ("llm", st.cfg["llm"]["url"].rstrip("/") + "/v1/models")):
            t = time.perf_counter()
            try:
                r = requests.get(url, timeout=3)
                out[name] = {"ok": r.status_code < 400, "status": r.status_code,
                             "ms": round((time.perf_counter() - t) * 1000)}
            except requests.RequestException as e:
                out[name] = {"ok": False, "error": type(e).__name__}
        # 대기열에 줄 선 요청 수, 차단 남은 시간, 마지막 오류
        out["decide"]["queue"] = st.sys["decide"].status()
        # 임베딩·리랭커 서버 (7단계). 꺼져도 /ask 는 그 단계를 건너뛰고 답하므로 전체 ok 에는 넣지 않는다
        v = st.cfg.get("vector", {})
        parts = {}
        if v.get("enabled"):
            parts["embed"] = v["embed_url"]
        if v.get("rerank_enabled"):
            parts["rerank"] = v["reranker_url"]
        for name, base in parts.items():
            t = time.perf_counter()
            try:
                r = requests.get(base.rstrip("/") + "/v1/models", timeout=3)
                out[name] = {"ok": r.status_code < 400, "status": r.status_code,
                             "ms": round((time.perf_counter() - t) * 1000)}
            except requests.RequestException as e:
                out[name] = {"ok": False, "error": type(e).__name__}
        # 검색 경로의 차단기 상태 (리랭크 엔진 → 혼합 엔진 순으로 감싸져 있다)
        w = st.sys["index"]
        while hasattr(w, "breaker"):
            out.setdefault(w.breaker.name if w.breaker.name != "vector" else "embed", {})["breaker"] = w.breaker.status()
            w = w.base
        eng = st.sys.get("engine", st.sys["index"])
        try:
            docs, chunks = eng.counts() if hasattr(eng, "counts") else (len(eng.doc_by_id), len(eng.chunk_by_id))
            out["engine"] = {"ok": True, "name": eng.name, "docs": docs, "chunks": chunks}
        except Exception as e:  # 엔진이 내려가도 health 자체는 답한다
            out["engine"] = {"ok": False, "name": eng.name, "error": type(e).__name__}
        m = load_manifest(st.cfg)
        out["index"] = {"manifest_docs": len(m), "parse_errors": sum(1 for v in m.values() if v.get("error"))}
        out["auth"] = "gateway" if os.environ.get(token_env) else ("dev" if sc.get("dev_mode") else "unconfigured")
        out["build"] = st.build_status.get("state")
        out["ok"] = all(out[k]["ok"] for k in ("decide", "llm", "engine")) and out["auth"] != "unconfigured"
        return out

    #------------------------------------------------------------------
    # 색인 다시 하기 (POST /index/rebuild)
    #=> 증분 색인을 백그라운드로 돌리고 바로 202 를 돌려준다. 끝나면 메모리 엔진이면 검색 시스템을 다시 만든다.
    #
    # -in: body = RebuildBody
    # -in: user = admin_user 결과
    #
    # -out: {"state": "running"}
    # -out: error = 관리자 아님 403, 이미 실행 중 409
    #------------------------------------------------------------------
    @app.post("/index/rebuild", status_code=202)
    def rebuild(body: RebuildBody, user: dict = Depends(admin_user)):
        if not st.build_lock.acquire(blocking=False):
            raise HTTPException(409, "색인이 이미 실행 중입니다")
        st.build_status = {"state": "running", "started": time.strftime("%Y-%m-%d %H:%M:%S"), "by": user["id"]}

        # 백그라운드에서 색인 → 상태 기록 → 잠금 해제
        def work():
            from .cli import load_system
            from .indexer import build
            try:
                summary = build(st.cfg, full=body.full, log=lambda *_: None)
                if st.sys.get("engine", st.sys["index"]).name == "memory":
                    # 새 시스템을 다 만든 뒤 한 번에 바꿔 끼운다 (진행 중인 요청은 옛 것을 끝까지 쓴다)
                    new = load_system(st.cfg)
                    with st.reload_lock:
                        st.sys = new
                st.build_status.update({"state": "done", "summary": summary})
            except Exception as e:  # 실패 내용을 상태로 남긴다
                st.build_status.update({"state": "failed", "error": f"{type(e).__name__}: {e}"})
            finally:
                st.build_status["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
                st.build_lock.release()

        threading.Thread(target=work, daemon=True).start()
        return {"state": "running"}

    #------------------------------------------------------------------
    # 색인 상태 (GET /index/status)
    #=> 마지막 색인의 상태와 요약, 파싱 실패 문서 목록(경로)을 돌려준다.
    #
    # -in: user = admin_user 결과
    #
    # -out: {"build": build_status, "parse_errors": [{"doc_id","rel","error"}]}
    # -out: error = 관리자 아님 403
    #------------------------------------------------------------------
    @app.get("/index/status")
    def index_status(user: dict = Depends(admin_user)):
        from .indexer import load_manifest
        errs = [{"doc_id": d, "rel": v.get("rel"), "error": v.get("error")}
                for d, v in load_manifest(st.cfg).items() if v.get("error")]
        return {"build": st.build_status, "parse_errors": errs}

    return app


#------------------------------------------------------------------
# 서버 실행
#=> uvicorn 으로 앱을 띄운다. 기본은 이 PC 에서만 접속(127.0.0.1).
#
# -in: cfg = 설정 dict
# -in: host = 주소 (None = 설정값)
# -in: port = 포트 (None = 설정값)
#
# -out: 없음 (종료할 때까지 실행)
# -out: error = 포트 사용 중 등은 uvicorn 예외 전파
#------------------------------------------------------------------
def serve(cfg: dict, host: str | None = None, port: int | None = None):
    import uvicorn
    sc = cfg.get("service", {})
    uvicorn.run(create_app(cfg), host=host or sc.get("host", "127.0.0.1"), port=port or sc.get("port", 8090),
                log_level="info")
