"""장애 처리·동시성 시험 (설계서 8.1·8.2, 14장 3단계) — OpenSearch(decide_* 색인) 필요. decide·Qwen 서버는 일부 항목만.

  1) decide 꺼짐       : /ask 200 + degraded, 근거 = BM25 상위 5, 두 번째 요청은 차단기로 decide 를 부르지 않음
  2) decide 응답 지연  : 응답 대기 시간 초과 → degraded
  3) decide 중간 실패  : 1라운드 판단은 받고 재검색 중 실패 → 판단해 둔 순위로 답
  4) LLM 꺼짐          : /ask 502 + 근거 목록
  5) 검색 0건          : LLM 을 부르지 않고 "찾을 수 없습니다"
  6) 동시 요청 4개     : decide 는 동시에 1개만, LLM 답변은 겹쳐서 진행
  7) 대기열 가득       : DecideBusy → /ask 503
  8) 임베딩 꺼짐 (7단계) : BM25+리랭커로 답, degraded 에 vector 이유, 두 번째는 차단기
  9) 리랭커 꺼짐 (7단계) : 혼합 검색 순서로 답, degraded 에 rerank 이유
  python tests\\test_degraded.py
"""
import http.server
import json
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fastapi.testclient import TestClient  # noqa: E402

from decide_rag.answer import NOT_FOUND  # noqa: E402
from decide_rag.clients import DecideBusy, DecideClient, QueuedDecide  # noqa: E402
from decide_rag.config import load_config  # noqa: E402
from decide_rag.judge import DecideJudge  # noqa: E402
from decide_rag.service import create_app  # noqa: E402

Q = "나 다음달 결혼하는데 회사에서 축의금이랑 휴가 얼마나 줘?"
DEAD = "http://127.0.0.1:1"   # 아무것도 듣지 않는 주소 (연결 거부)


#------------------------------------------------------------------
# 조건 확인
#=> 틀리면 무엇이 틀렸는지 보여 주고 AssertionError.
#
# -in: ok = 조건
# -in: msg = 설명
#
# -out: 없음
# -out: error = 조건이 거짓이면 AssertionError
#------------------------------------------------------------------
def check(ok: bool, msg: str):
    print(("통과 " if ok else "실패 ") + msg, flush=True)
    assert ok, msg


#------------------------------------------------------------------
# 시험용 설정
#=> dev_mode 로 인증을 단순화하고 trace 는 끈다. decide·LLM 주소와 시간 설정을 바꿔 끼운다.
#
# -in: decide_url = decide 주소 (None = 그대로)
# -in: llm_url = LLM 주소 (None = 그대로)
# -in: **decide_kw = [decide] 에 덮어쓸 값
#
# -out: cfg = 설정 dict
# -out: error = config.toml 이 없으면 예외 전파
#------------------------------------------------------------------
def cfg_with(decide_url=None, llm_url=None, **decide_kw) -> dict:
    cfg = load_config()
    cfg["service"] = {**cfg["service"], "dev_mode": True, "dev_groups": [], "trace": False}
    if decide_url:
        cfg["decide"]["url"] = decide_url
    if llm_url:
        cfg["llm"]["url"] = llm_url
    cfg["decide"].update(decide_kw)
    return cfg


#------------------------------------------------------------------
# 판단 결과를 바로 주는 가짜 decide
#=> 관련성·충분성 0.9, 라우팅은 첫 후보. 동시에 몇 개가 처리 중인지 센다 (최대값 기록).
#
# -필드: delay = 요청당 처리 시간(초)
# -필드: active / max_active = 지금·최대 동시 처리 수
# -필드: calls = 받은 요청 수
#------------------------------------------------------------------
class FakeDecide:
    #------------------------------------------------------------------
    # 만들기
    #=> 처리 시간을 받는다.
    #
    # -in: delay = 요청당 처리 시간(초)
    #
    # -out: 없음
    # -out: error = 없음
    #------------------------------------------------------------------
    def __init__(self, delay: float = 0.02):
        self.delay, self.active, self.max_active, self.calls = delay, 0, 0, 0
        self._l = threading.Lock()

    #------------------------------------------------------------------
    # 판단 요청 흉내
    #=> DecideClient.ask 와 같은 모양으로 고정 확률을 돌려준다.
    #
    # -in: state, questions, temperature = DecideClient.ask 와 같음
    #
    # -out: (answers, info)
    # -out: error = 없음
    #------------------------------------------------------------------
    def ask(self, state, questions, temperature=1.0):
        with self._l:
            self.active += 1
            self.calls += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(self.delay)
        out = {}
        for name, q in questions.items():
            if q["type"] == "choice":
                keys = list(q["criteria"])
                out[name] = {"probabilities": {k: (1.0 if i == 0 else 0.0) for i, k in enumerate(keys)}}
            else:
                out[name] = {"noul": 0.9}
        with self._l:
            self.active -= 1
        return out, {"latency": self.delay, "input_tokens": 10}


#------------------------------------------------------------------
# 몇 번 답한 뒤 실패하는 decide
#=> 진짜 decide 를 n 번 부른 다음부터 연결 오류를 낸다 (중간 실패 흉내).
#
# -필드: inner = 진짜 DecideClient
# -필드: n = 성공시킬 횟수
#------------------------------------------------------------------
class FailAfter:
    #------------------------------------------------------------------
    # 만들기
    #=> 진짜 클라이언트와 성공 횟수를 받는다.
    #
    # -in: inner = DecideClient
    # -in: n = 성공시킬 횟수
    #
    # -out: 없음
    # -out: error = 없음
    #------------------------------------------------------------------
    def __init__(self, inner, n: int):
        self.inner, self.n = inner, n

    #------------------------------------------------------------------
    # 판단 요청
    #=> n 번까지는 진짜로 보내고, 그다음은 ConnectionError.
    #
    # -in: state, questions, temperature = DecideClient.ask 와 같음
    #
    # -out: (answers, info)
    # -out: error = n 번 뒤에는 ConnectionError
    #------------------------------------------------------------------
    def ask(self, *a):
        if self.n <= 0:
            raise ConnectionError("가짜 연결 끊김")
        self.n -= 1
        return self.inner.ask(*a)


#------------------------------------------------------------------
# 느린 가짜 LLM
#=> 답 하나에 delay 초. 동시에 몇 개가 처리 중인지 센다.
#
# -필드: delay = 답 하나 처리 시간(초)
# -필드: active / max_active = 지금·최대 동시 처리 수
#------------------------------------------------------------------
class FakeLLM:
    #------------------------------------------------------------------
    # 만들기
    #=> 처리 시간을 받는다.
    #
    # -in: delay = 처리 시간(초)
    #
    # -out: 없음
    # -out: error = 없음
    #------------------------------------------------------------------
    def __init__(self, delay: float = 0.5):
        self.delay, self.active, self.max_active = delay, 0, 0
        self._l = threading.Lock()

    #------------------------------------------------------------------
    # 채팅 흉내
    #=> LLMClient.chat 과 같은 모양.
    #
    # -in: messages, max_tokens, temperature, thinking = LLMClient.chat 과 같음 (thinking 은 무시)
    #
    # -out: (text, info)
    # -out: error = 없음
    #------------------------------------------------------------------
    def chat(self, messages, max_tokens=512, temperature=0.0, thinking=None):
        with self._l:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(self.delay)
        with self._l:
            self.active -= 1
        return "가짜 답 [1]", {"latency": self.delay}


#------------------------------------------------------------------
# 응답을 늦게 주는 가짜 서버
#=> POST 를 받으면 delay 초 뒤에 답한다 (decide 응답 지연 흉내). 별도 스레드에서 돈다.
#
# -in: delay = 응답 지연(초)
#
# -out: (server, url) = 서버 객체(끝낼 때 shutdown), 주소
# -out: error = 포트를 못 열면 OSError
#------------------------------------------------------------------
def slow_server(delay: float):
    # 요청마다 delay 초 쉬고 빈 JSON 을 돌려주는 처리기
    class H(http.server.BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            time.sleep(delay)
            try:
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"{}")
            except OSError:
                pass

        def log_message(self, *a):  # 시험 출력이 지저분해지지 않게
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


#------------------------------------------------------------------
# 가짜 부품 끼우기
#=> 앱의 decide 대기열·판단기 공장·LLM 을 바꿔 끼운다.
#
# -in: app = create_app 결과
# -in: decide = ask() 를 가진 객체 (QueuedDecide 로 감싼다)
# -in: llm = chat() 을 가진 객체 (None = 그대로)
# -in: max_queue = 대기열 상한
#
# -out: q = 끼운 QueuedDecide
# -out: error = 없음
#------------------------------------------------------------------
def plug(app, decide, llm=None, max_queue: int = 50):
    s = app.state.s
    q = QueuedDecide(decide, max_queue=max_queue, cooldown_sec=30)
    s.sys["decide"] = q
    s.sys["new_judge"] = lambda: DecideJudge(q, s.sys["index"], s.cfg["agent"], 1200)
    if llm:
        s.sys["llm"] = llm
    return q


#------------------------------------------------------------------
# 시험 본체
#=> 위 1)~7) 을 차례로 확인한다.
#
# -in: 없음
#
# -out: 없음
# -out: error = 하나라도 틀리면 AssertionError
#------------------------------------------------------------------
def main():
    sys.stdout.reconfigure(encoding="utf-8")
    base = load_config()

    # 1) decide 꺼짐 (연결 거부) — LLM 은 진짜
    app = create_app(cfg_with(DEAD))
    c = TestClient(app)
    eng = app.state.s.sys["index"]
    bm25_top5 = [h[0]["chunk_id"] for h in eng.search(Q, 5, acl=["all"])]
    t = time.perf_counter()
    r = c.post("/ask", json={"question": Q, "debug": True})
    t1 = time.perf_counter() - t
    b = r.json()
    check(r.status_code == 200 and b["degraded"] and b["answer"], f"decide 꺼짐 → 200 degraded, 답 있음 ({t1:.1f}초)")
    check([x["chunk_id"] for x in b["sources"]] == bm25_top5, "근거 = BM25 상위 5개 (판단 없음)")
    print("   답:", b["answer"][:80].replace("\n", " "), "| 이유:", b["degraded_reason"][:80])
    t = time.perf_counter()
    r2 = c.post("/ask", json={"question": Q, "debug": True})
    t2 = time.perf_counter() - t
    check(r2.status_code == 200 and r2.json()["degraded"] and "차단" in r2.json()["degraded_reason"],
          f"두 번째 요청은 차단기로 바로 degraded ({t2:.1f}초, 첫 요청 {t1:.1f}초)")
    hq = c.get("/health").json()["decide"]
    check(not hq["ok"] and hq["queue"]["down_for_sec"] > 0, f"/health 에 decide 차단 표시 {hq}")

    # 2) decide 응답 지연 (응답 대기 1초, 재시도 없음, 서버는 3초 뒤 응답)
    srv, url = slow_server(3.0)
    try:
        c = TestClient(create_app(cfg_with(url, timeout_sec=1, retries=0)))
        t = time.perf_counter()
        r = c.post("/ask", json={"question": Q})
        dt = time.perf_counter() - t
        check(r.status_code == 200 and r.json()["degraded"] and "Timeout" in r.json()["degraded_reason"],
              f"decide 응답 지연 → degraded ({dt:.1f}초)")
    finally:
        srv.shutdown()

    # 3) decide 중간 실패: 재검색까지 가는 문항에서 1라운드 판단 11회(재정렬 10 + 충분성 1) 뒤 실패
    #    (7단계 혼합 검색+리랭커에서는 A010 이 1라운드에 끝나서, 2라운드 이상 가는 A014 로 바꿨다)
    item = next(json.loads(l) for l in open(ROOT / "eval" / "evalset_v2.jsonl", encoding="utf-8")
                if json.loads(l)["id"] == "A014")
    app = create_app(cfg_with())
    plug(app, FailAfter(DecideClient(base["decide"]), 11))
    r = TestClient(app).post("/ask", json={"question": item["question"], "debug": True})
    b = r.json()
    r0 = b["trace"]["rounds"][0]
    check(r.status_code == 200 and b["degraded"] and [x["chunk_id"] for x in b["sources"]] == r0["context"],
          "재검색 중 실패 → 1라운드 판단 순위로 답 (BM25 순서가 아님)")

    # 4) LLM 꺼짐 — decide 는 가짜
    app = create_app(cfg_with(llm_url=DEAD))
    plug(app, FakeDecide())
    r = TestClient(app).post("/ask", json={"question": Q})
    check(r.status_code == 502 and r.json().get("sources"), f"LLM 꺼짐 → 502 + 근거 {len(r.json().get('sources', []))}개")

    # 5) 검색 0건 — LLM 은 꺼진 주소 (불리면 502 가 됨)
    #    벡터 검색은 언제나 가까운 청크를 돌려주므로 0건이 나지 않는다 → 벡터를 끄고 BM25 길만 본다
    cfg5 = cfg_with(llm_url=DEAD)
    cfg5["vector"] = {**cfg5["vector"], "enabled": False, "rerank_enabled": False}
    app = create_app(cfg5)
    fd = FakeDecide()
    plug(app, fd)
    r = TestClient(app).post("/ask", json={"question": "zzqxv qqqzz"})
    check(r.status_code == 200 and r.json()["answer"] == NOT_FOUND and not r.json()["sources"],
          f"검색 0건 → LLM 없이 찾을 수 없음 (decide 호출 {fd.calls}회)")

    # 6) 동시 요청 4개 — 가짜 decide(호출당 0.02초)·가짜 LLM(0.5초)
    app = create_app(cfg_with())
    fd, fl = FakeDecide(0.02), FakeLLM(0.5)
    plug(app, fd, fl)
    c = TestClient(app)
    codes, ts = [], []

    # 질문 하나를 보내고 결과·시간을 기록
    def one():
        t0 = time.perf_counter()
        codes.append(c.post("/ask", json={"question": Q}).status_code)
        ts.append(time.perf_counter() - t0)

    t = time.perf_counter()
    th = [threading.Thread(target=one) for _ in range(4)]
    for x in th:
        x.start()
    for x in th:
        x.join()
    wall = time.perf_counter() - t
    check(codes == [200] * 4, f"동시 4개 모두 200 {codes}")
    check(fd.max_active == 1, f"decide 동시 처리 최대 {fd.max_active}개 (대기열로 1개씩)")
    check(fl.max_active >= 2, f"LLM 답변은 겹쳐서 진행 (최대 동시 {fl.max_active}개, 전체 {wall:.2f}초)")

    # 7) 대기열 가득 → 503
    class Busy:  # 항상 대기열이 가득 찬 것처럼
        def ask(self, *a):
            raise DecideBusy("가득")
    app = create_app(cfg_with())
    s = app.state.s
    s.sys["new_judge"] = lambda: DecideJudge(Busy(), s.sys["index"], s.cfg["agent"], 1200)
    check(TestClient(app).post("/ask", json={"question": Q}).status_code == 503, "decide 대기열 가득 → 503")
    # QueuedDecide 의 상한 자체도 확인: 느린 요청 하나가 처리 중, 하나가 대기 중이면 세 번째는 DecideBusy
    q = QueuedDecide(FakeDecide(0.5), max_queue=1)
    errs = []

    # 요청 하나 (DecideBusy 면 기록)
    def put():
        try:
            q.ask("s", {"x": {"type": "noul", "instructions": "?"}})
        except DecideBusy:
            errs.append(1)

    th = [threading.Thread(target=put) for _ in range(3)]
    for x in th:
        x.start()
        time.sleep(0.05)
    for x in th:
        x.join()
    check(len(errs) == 1, f"max_queue=1: 처리 1 + 대기 1, 세 번째는 DecideBusy ({len(errs)}건)")

    # 8) 임베딩 서버 꺼짐 (7단계) — 가짜 decide·LLM. 벡터를 건너뛰고 BM25 → 리랭커로 답한다
    if base.get("vector", {}).get("enabled"):
        cfg8 = cfg_with()
        cfg8["vector"] = {**cfg8["vector"], "embed_url": DEAD}
        app = create_app(cfg8)
        plug(app, FakeDecide(), FakeLLM())
        c = TestClient(app)
        t = time.perf_counter()
        b = c.post("/ask", json={"question": Q}).json()
        t8 = time.perf_counter() - t
        check(b.get("degraded") and "vector" in (b.get("degraded_reason") or "") and b.get("sources"),
              f"임베딩 꺼짐 → 답 있음, degraded=vector ({t8:.1f}초, 근거 {len(b.get('sources', []))}개)")
        b2 = c.post("/ask", json={"question": Q}).json()
        check("차단" in (b2.get("degraded_reason") or ""), "두 번째 요청은 임베딩 차단기로 바로 건너뜀")
        h = c.get("/health").json()
        check(not h["embed"]["ok"] and h["embed"]["breaker"]["down_for_sec"] > 0, f"/health 에 임베딩 꺼짐·차단 {h['embed']}")

    # 9) 리랭커 서버 꺼짐 (7단계) — 혼합 검색(RRF) 순서로 답한다
    if base.get("vector", {}).get("rerank_enabled"):
        cfg9 = cfg_with()
        cfg9["vector"] = {**cfg9["vector"], "reranker_url": DEAD}
        app = create_app(cfg9)
        plug(app, FakeDecide(), FakeLLM())
        b = TestClient(app).post("/ask", json={"question": Q}).json()
        check(b.get("degraded") and "rerank" in (b.get("degraded_reason") or "") and b.get("sources"),
              f"리랭커 꺼짐 → 답 있음, degraded=rerank (근거 {len(b.get('sources', []))}개)")
    print("모두 통과")


if __name__ == "__main__":
    main()
