# -*- coding: utf-8 -*-
"""모델 서버 클라이언트: LLM(OpenAI 호환) · decide(/v1/systemone)."""
import json
import queue
import re
import threading
import time
from concurrent.futures import Future

import requests


#------------------------------------------------------------------
# decide 를 지금 쓸 수 없음
#=> 재시도 후에도 실패했거나, 최근 실패로 차단기가 열려 있을 때.
#   에이전트는 이 예외를 받으면 판단 없이 답한다 (degraded).
#------------------------------------------------------------------
class DecideUnavailable(RuntimeError):
    pass


#------------------------------------------------------------------
# decide 대기열이 가득 참
#=> 기다리는 요청이 상한을 넘으면 새 요청을 받지 않는다. 서비스는 503 "잠시 후 다시" 로 돌려준다.
#------------------------------------------------------------------
class DecideBusy(RuntimeError):
    pass


#------------------------------------------------------------------
# 요청 자체가 잘못됨 (HTTP 4xx)
#=> 서버는 살아 있고 이 요청만 문제다 (예: 입력이 너무 김). 다시 보내도 같으므로 재시도하지 않고,
#   decide 차단기도 열지 않는다 (다른 사용자 요청까지 막지 않게).
#------------------------------------------------------------------
class ClientRequestError(RuntimeError):
    pass


#------------------------------------------------------------------
# 생성 LLM 실패
#=> 근거는 찾았지만 답을 쓰지 못했을 때. 서비스는 근거 목록과 함께 502 로 돌려준다.
#
# -필드: context = 찾아 둔 근거 chunk_id 목록
#------------------------------------------------------------------
class LLMUnavailable(RuntimeError):
    #------------------------------------------------------------------
    # 예외 만들기
    #=> 메시지와 함께 근거 chunk_id 목록을 들고 간다.
    #
    # -in: msg = 오류 메시지
    # -in: context = 근거 chunk_id 목록
    #
    # -out: 없음
    # -out: error = 없음
    #------------------------------------------------------------------
    def __init__(self, msg: str, context: list[str]):
        super().__init__(msg)
        self.context = context


#------------------------------------------------------------------
# 재시도 포함 POST
#=> 연결 오류·5xx 는 점점 길게 쉬며 다시 보낸다. 4xx 는 요청 자체 문제라 바로 실패한다.
#
# -in: session = requests.Session
# -in: url = 요청 주소
# -in: body = JSON 본문
# -in: timeout = 응답 기다리는 시간(초)
# -in: retries = 재시도 횟수 (기본 3)
# -in: connect_timeout = 연결 기다리는 시간(초). 서버가 꺼져 있을 때 빨리 포기하려고 따로 짧게 둔다 (기본 3)
# -in: backoff = 첫 재시도 전 쉬는 시간(초). 회마다 두 배, 최대 10초 (기본 1)
#
# -out: (data, latency) = 응답 JSON, 마지막 성공 요청의 걸린 시간(초)
# -out: error = 4xx 면 바로 ClientRequestError, 재시도 후에도 실패하면 RuntimeError
#------------------------------------------------------------------
def post_json(session, url: str, body: dict, timeout: float, retries: int = 3, connect_timeout: float = 3.0,
              backoff: float = 1.0) -> tuple[dict, float]:
    last = None
    for attempt in range(retries + 1):
        if attempt:
            # 마지막 시도 뒤에는 쉬지 않는다 (실패를 빨리 알리기 위해)
            time.sleep(min(backoff * 2 ** (attempt - 1), 10))
        t0 = time.perf_counter()
        try:
            r = session.post(url, json=body, timeout=(connect_timeout, timeout))
            dt = time.perf_counter() - t0
            if r.status_code < 400:
                return r.json(), dt
            last = f"HTTP {r.status_code}: {r.text[:300]}"
            # 입력이 잘못된 요청은 다시 보내도 같다
            if r.status_code < 500:
                raise ClientRequestError(f"POST {url} → {last}")
        except requests.RequestException as e:
            last = f"{type(e).__name__}: {e}"
    raise RuntimeError(f"POST {url} 실패 ({retries + 1}회): {last}")


#------------------------------------------------------------------
# 생성 LLM 클라이언트
#=> OpenAI 호환 /v1/chat/completions 로 Qwen 을 부른다. 걸린 시간·토큰 수를 같이 돌려준다.
#
# -필드: url, model = 서버 주소, 모델 이름
# -필드: enable_thinking = Qwen 생각 모드 사용 여부
# -필드: session = HTTP 세션
#------------------------------------------------------------------
class LLMClient:
    #------------------------------------------------------------------
    # 클라이언트 만들기
    #=> 설정의 [llm] 값으로 세션을 준비한다.
    #
    # -in: c = cfg["llm"] dict
    #
    # -out: 없음
    # -out: error = 없음
    #------------------------------------------------------------------
    def __init__(self, c: dict):
        self.url = c["url"].rstrip("/") + "/v1/chat/completions"
        self.model = c["model"]
        self.timeout = c.get("timeout_sec", 300)
        self.enable_thinking = c.get("enable_thinking", False)
        self.retries = c.get("retries", 1)
        self.connect_timeout = c.get("connect_timeout_sec", 3)
        self.session = requests.Session()

    #------------------------------------------------------------------
    # 채팅 한 번
    #=> 메시지를 보내 답 문자열과 사용량 정보를 받는다.
    #    1) temperature 0 으로 같은 입력엔 최대한 같은 답
    #    2) 생각 모드는 chat_template_kwargs 로 끈다 (켜면 수 초~수십 초 더 걸림).
    #       thinking 을 주면 이 호출만 설정값 대신 그 값을 쓴다 (4단계 A1: 추론 문항만 켜기)
    #    3) 혹시 남은 <think>…</think> 는 지운다
    #
    # -in: messages = [{"role","content"}, …]
    # -in: max_tokens = 최대 생성 토큰 (기본 512). 생각 모드면 생각 토큰도 여기에 포함된다
    # -in: temperature = 샘플링 온도 (기본 0)
    # -in: thinking = 이 호출의 생각 모드 (기본 None = 설정값 enable_thinking)
    #
    # -out: (text, info) = 답 문자열, {"latency","prompt_tokens","completion_tokens","peak_memory_gb","finish_reason"}
    # -out: error = 서버 실패 시 RuntimeError
    #------------------------------------------------------------------
    def chat(self, messages: list[dict], max_tokens: int = 512, temperature: float = 0.0,
             thinking: bool | None = None) -> tuple[str, dict]:
        think = self.enable_thinking if thinking is None else thinking
        body = {"model": self.model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature,
                "chat_template_kwargs": {"enable_thinking": think}}
        data, dt = post_json(self.session, self.url, body, self.timeout, self.retries, self.connect_timeout)
        text = data["choices"][0]["message"].get("content") or ""
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
        u = data.get("usage") or {}
        tm = data.get("timings") or {}
        # finish_reason "length" = 토큰 한도에서 잘림 (생각 모드에서 답을 쓰기 전에 잘렸는지 보려고)
        return text, {"latency": dt, "prompt_tokens": u.get("prompt_tokens"),
                      "completion_tokens": u.get("completion_tokens"), "peak_memory_gb": tm.get("peak_memory"),
                      "finish_reason": data["choices"][0].get("finish_reason")}

    #------------------------------------------------------------------
    # JSON 으로 답 받기
    #=> chat 결과에서 첫 { … } 덩어리를 찾아 파싱한다. 코드블록(```json)도 허용.
    #
    # -in: messages = 메시지 목록
    # -in: max_tokens = 최대 생성 토큰
    # -in: temperature = 샘플링 온도 (기본 0)
    #
    # -out: (obj, info) = 파싱한 dict (실패 시 None), 사용량 정보 (+ "raw" 원문)
    # -out: error = 서버 실패 시 RuntimeError, JSON 파싱 실패는 obj=None
    #------------------------------------------------------------------
    def chat_json(self, messages: list[dict], max_tokens: int = 512,
                  temperature: float = 0.0) -> tuple[dict | None, dict]:
        text, info = self.chat(messages, max_tokens, temperature)
        info["raw"] = text
        m = re.search(r"\{.*\}", text, flags=re.S)
        if not m:
            return None, info
        try:
            return json.loads(m.group(0)), info
        except json.JSONDecodeError:
            return None, info


#------------------------------------------------------------------
# decide 판단 모델 클라이언트
#=> kanana-2-v-decide 공식 서버의 POST /v1/systemone 을 부른다.
#   질문 유형: noul(예/아니오 확률), choice(선택지 확률), score(단계 점수).
#
# -필드: url = /v1/systemone 주소
# -필드: session = 인증 헤더가 붙은 HTTP 세션
#------------------------------------------------------------------
class DecideClient:
    #------------------------------------------------------------------
    # 클라이언트 만들기
    #=> 설정의 [decide] 값과 API 키로 세션을 준비한다.
    #
    # -in: c = cfg["decide"] dict (url, api_key, timeout_sec, retries, connect_timeout_sec, backoff_sec)
    #
    # -out: 없음
    # -out: error = 없음 (키가 틀리면 첫 요청에서 RuntimeError)
    #------------------------------------------------------------------
    def __init__(self, c: dict):
        self.url = c["url"].rstrip("/") + "/v1/systemone"
        self.timeout = c.get("timeout_sec", 60)
        self.retries = c.get("retries", 2)
        self.connect_timeout = c.get("connect_timeout_sec", 3)
        self.backoff = c.get("backoff_sec", 0.5)
        self.session = requests.Session()
        if c.get("api_key"):
            self.session.headers["Authorization"] = f"Bearer {c['api_key']}"

    #------------------------------------------------------------------
    # 판단 요청
    #=> 상황 글(state)과 질문 묶음을 보내 질문별 확률을 받는다.
    #
    # -in: state = 판단 대상 글 (질문 + 근거 청크 등)
    # -in: questions = {"이름": {"type","instructions","criteria"}, …}
    # -in: temperature = 확률 온도 (기본 1.0)
    #
    # -out: (answers, info) = {"이름": {...}}, {"latency","input_tokens"}
    # -out: error = 서버 실패 시 RuntimeError
    #------------------------------------------------------------------
    def ask(self, state: str, questions: dict, temperature: float = 1.0) -> tuple[dict, dict]:
        body = {"state": state, "questions": questions, "temperature": temperature}
        data, dt = post_json(self.session, self.url, body, self.timeout, self.retries, self.connect_timeout,
                             self.backoff)
        return data["answers"], {"latency": dt, "input_tokens": (data.get("usage") or {}).get("input_tokens")}


#------------------------------------------------------------------
# decide 대기열 + 차단기 (설계서 8.1·8.2)
#=> decide 서버는 한 번에 한 요청만 처리한다. 여러 질문이 동시에 와도 decide 요청은 일꾼 스레드 하나가 온 순서대로 보낸다.
#    1) ask() 는 요청을 대기열에 넣고 결과를 기다린다 (온 순서대로 처리)
#    2) 기다리는 요청이 max_queue 를 넘으면 DecideBusy
#    3) 재시도 후에도 실패하면 cooldown_sec 동안 차단기를 연다 → 그동안 ask() 는 바로 DecideUnavailable
#       (꺼진 서버에 질문마다 재시도하며 기다리지 않게). 시간이 지나면 다음 요청으로 다시 시도한다
#   DecideClient 와 같은 ask() 모양이라 DecideJudge 는 둘을 구분하지 않는다.
#
# -필드: client = 실제 DecideClient
# -필드: q = 대기열
# -필드: max_queue = 기다리는 요청 상한
# -필드: cooldown = 실패 후 차단 시간(초)
# -필드: down_until = 차단이 풀리는 시각 (time.monotonic)
# -필드: last_error = 마지막 실패 내용
#------------------------------------------------------------------
class QueuedDecide:
    #------------------------------------------------------------------
    # 대기열 만들기
    #=> 일꾼 스레드를 하나 띄운다 (프로그램이 끝나면 같이 끝나는 daemon).
    #
    # -in: client = DecideClient (또는 같은 ask() 를 가진 객체)
    # -in: max_queue = 기다리는 요청 상한 (기본 50)
    # -in: cooldown_sec = 실패 후 차단 시간 (기본 30초)
    #
    # -out: 없음
    # -out: error = 없음
    #------------------------------------------------------------------
    def __init__(self, client, max_queue: int = 50, cooldown_sec: float = 30.0):
        self.client = client
        self.max_queue = max_queue
        self.cooldown = cooldown_sec
        self.q: "queue.Queue[tuple]" = queue.Queue()
        self.down_until = 0.0
        self.last_error: str | None = None
        self._lock = threading.Lock()
        threading.Thread(target=self._worker, daemon=True, name="decide-queue").start()

    #------------------------------------------------------------------
    # 차단기 상태
    #=> 지금 차단 중이면 남은 초, 아니면 0.
    #
    # -in: 없음
    #
    # -out: seconds = 차단 남은 시간(초)
    # -out: error = 없음
    #------------------------------------------------------------------
    def down_for(self) -> float:
        return max(0.0, self.down_until - time.monotonic())

    #------------------------------------------------------------------
    # 일꾼 스레드
    #=> 대기열에서 하나씩 꺼내 실제 decide 에 보내고 결과(또는 예외)를 Future 에 넣는다.
    #   차단 중에 꺼낸 요청은 보내지 않고 바로 DecideUnavailable 로 끝낸다.
    #
    # -in: 없음
    #
    # -out: 없음 (끝나지 않는 반복)
    # -out: error = 예외는 Future 로 넘기고 반복은 계속
    #------------------------------------------------------------------
    def _worker(self):
        while True:
            fut, args, t_in = self.q.get()
            wait = time.perf_counter() - t_in
            try:
                if self.down_for() > 0:
                    raise DecideUnavailable(f"decide 차단 중 ({self.down_for():.0f}초 남음): {self.last_error}")
                ans, info = self.client.ask(*args)
                fut.set_result((ans, {**info, "wait": wait}))
            except DecideUnavailable as e:
                fut.set_exception(e)
            except ClientRequestError as e:  # 이 요청만의 문제 → 차단기는 그대로
                fut.set_exception(DecideUnavailable(f"{type(e).__name__}: {str(e)[:200]}"))
            except Exception as e:  # 재시도까지 실패 → 차단기를 열어 뒤에 기다리던 요청도 바로 실패시킨다
                with self._lock:
                    self.last_error = f"{type(e).__name__}: {str(e)[:200]}"
                    self.down_until = time.monotonic() + self.cooldown
                fut.set_exception(DecideUnavailable(self.last_error))
            finally:
                self.q.task_done()

    #------------------------------------------------------------------
    # 판단 요청 (대기열 경유)
    #=> DecideClient.ask 와 같은 모양. 대기열에 넣고 결과를 기다린다. info 에 대기 시간("wait")이 붙는다.
    #
    # -in: state = 판단 대상 글
    # -in: questions = 질문 묶음
    # -in: temperature = 확률 온도 (기본 1.0)
    #
    # -out: (answers, info) = DecideClient.ask 와 같음 + info["wait"]
    # -out: error = 차단 중·실패 시 DecideUnavailable, 대기열 가득 차면 DecideBusy
    #------------------------------------------------------------------
    def ask(self, state: str, questions: dict, temperature: float = 1.0) -> tuple[dict, dict]:
        if self.down_for() > 0:
            # 차단 중이면 대기열에 넣지도 않는다 (꺼진 서버를 기다리는 줄이 생기지 않게)
            raise DecideUnavailable(f"decide 차단 중 ({self.down_for():.0f}초 남음): {self.last_error}")
        if self.q.qsize() >= self.max_queue:
            raise DecideBusy(f"decide 대기열이 가득 찼습니다 ({self.max_queue})")
        fut: Future = Future()
        self.q.put((fut, (state, questions, temperature), time.perf_counter()))
        return fut.result()

    #------------------------------------------------------------------
    # 상태 요약 (/health 용)
    #=> 대기 중인 요청 수, 차단 남은 시간, 마지막 오류.
    #
    # -in: 없음
    #
    # -out: {"queued","down_for_sec","last_error"}
    # -out: error = 없음
    #------------------------------------------------------------------
    def status(self) -> dict:
        return {"queued": self.q.qsize(), "down_for_sec": round(self.down_for(), 1), "last_error": self.last_error}
