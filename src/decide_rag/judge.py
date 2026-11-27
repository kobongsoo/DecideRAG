# -*- coding: utf-8 -*-
"""에이전트 판단 — kanana-2-v-decide 판단 모델.

decide 는 글을 쓰지 않고 확률만 돌려주므로 판단을 모두 확률 질문으로 만든다.
  rerank(question, chunks)          → 청크마다 "질문의 답이 들어 있다" 확률 (noul, 청크마다 1회)
  sufficient(question, chunks)      → 이 근거만으로 답할 수 있는 확률 (noul, 1회)
  route(question, candidates)       → 후보 문서별 확률 (choice, 1회, 후보 최대 20개 — 설계서 12.1)
  research_plan(question, query, ctx) → 재검색 계획 [(질의, 문서범위 또는 None), …]
"""
from .clients import DecideBusy, DecideUnavailable

CHUNK_CLIP = 1200


#------------------------------------------------------------------
# 판단 호출 통계
#=> 질문 하나에서 decide 를 몇 번 불렀고 시간·토큰이 얼마나 들었는지 센다.
#   재검색 상한(호출 수·시간 예산)을 확인할 때도 이 값을 쓴다.
#
# -필드: calls = 호출 수
# -필드: seconds = 걸린 시간 합 (초, decide 처리 시간만 — 재검색 시간 예산에 쓴다)
# -필드: tokens = 입력 토큰 합
# -필드: wait = 대기열에서 기다린 시간 합 (초, 다른 질문이 decide 를 쓰는 동안)
#------------------------------------------------------------------
class Stats:
    #------------------------------------------------------------------
    # 통계 초기화
    #=> 0 으로 시작한다.
    #
    # -in: 없음
    #
    # -out: 없음
    # -out: error = 없음
    #------------------------------------------------------------------
    def __init__(self):
        self.calls, self.seconds, self.tokens, self.wait = 0, 0.0, 0, 0.0

    #------------------------------------------------------------------
    # 호출 한 번 기록
    #=> 시간과 토큰 수, 대기 시간을 더한다.
    #
    # -in: seconds = 걸린 시간
    # -in: tokens = 토큰 수 (None 이면 0)
    # -in: wait = 대기열 대기 시간 (기본 0)
    #
    # -out: 없음
    # -out: error = 없음
    #------------------------------------------------------------------
    def add(self, seconds: float, tokens, wait: float = 0.0):
        self.calls += 1
        self.seconds += seconds
        self.tokens += tokens or 0
        self.wait += wait or 0.0

    #------------------------------------------------------------------
    # dict 로 내보내기
    #=> trace 저장용.
    #
    # -in: 없음
    #
    # -out: d = {"calls","seconds","tokens","wait"}
    # -out: error = 없음
    #------------------------------------------------------------------
    def as_dict(self) -> dict:
        return {"calls": self.calls, "seconds": round(self.seconds, 3), "tokens": self.tokens,
                "wait": round(self.wait, 3)}


#------------------------------------------------------------------
# 청크에 번호 붙여 이어 쓰기
#=> "[1] 글\n\n[2] 글" 형식. 충분성 판단과 답변 프롬프트가 같은 번호 체계를 쓴다.
#
# -in: chunks = 청크 목록
# -in: clip = 청크당 최대 글자 수 (기본 1200, decide 입력 한도 때문)
#
# -out: text = 번호 붙은 글
# -out: error = 없음
#------------------------------------------------------------------
def numbered(chunks: list[dict], clip: int = CHUNK_CLIP) -> str:
    return "\n\n".join(f"[{i + 1}] {c['text'][:clip]}" for i, c in enumerate(chunks))


#------------------------------------------------------------------
# decide 판단기
#=> 재정렬·충분성·문서 라우팅을 decide 확률로 하고, 재검색 질의는 생성 없이 규칙으로 만든다.
#
# -필드: dc = DecideClient
# -필드: index = 검색 엔진 (PRF 확장어·문서 후보 검색용)
# -필드: route_top / route_min = 라우팅 문서 수 / 2순위부터의 최소 확률
# -필드: n_cand = 라우팅 후보 문서 수 상한 (decide 입력 한도 때문에 20)
# -필드: n_prf = PRF 확장어 수
# -필드: clip = decide 에 넣을 청크 글자 수
# -필드: stats = 호출 통계
#------------------------------------------------------------------
class DecideJudge:
    name = "decide"

    #------------------------------------------------------------------
    # 판단기 만들기
    #=> 클라이언트·검색 엔진과 설정값을 받아 둔다.
    #
    # -in: dc = DecideClient
    # -in: index = 검색 엔진 (MemoryEngine 또는 OpenSearchEngine)
    # -in: agent_cfg = cfg["agent"] (route_top_docs, route_min_prob, route_candidates, prf_terms, history_turns)
    # -in: clip = 청크 글자 수 상한 (기본 1200)
    #
    # -out: 없음
    # -out: error = 없음
    #------------------------------------------------------------------
    def __init__(self, dc, index, agent_cfg: dict, clip: int = CHUNK_CLIP):
        self.dc, self.index, self.clip = dc, index, clip
        self.n_cand = agent_cfg.get("route_candidates", 20)
        self.route_top = agent_cfg.get("route_top_docs", 2)
        self.route_min = agent_cfg.get("route_min_prob", 0.05)
        self.n_prf = agent_cfg.get("prf_terms", 5)
        self.history_turns = agent_cfg.get("history_turns", 0)
        self.stats = Stats()
        self._doc_hits: dict = {}

    #------------------------------------------------------------------
    # 새 질문 시작
    #=> 질문마다 decide 호출 통계와 문서 후보 검색 결과(캐시)를 비운다.
    #
    # -in: 없음
    #
    # -out: 없음
    # -out: error = 없음
    #------------------------------------------------------------------
    def new_question(self):
        self.stats = Stats()
        self._doc_hits = {}

    #------------------------------------------------------------------
    # decide 호출 + 통계
    #=> ask 를 부르고 시간·토큰·대기 시간을 stats 에 쌓는다.
    #   어떤 이유로든 decide 답을 못 받으면 DecideUnavailable 로 바꿔 올린다 → 에이전트가 degraded 로 답한다.
    #
    # -in: state = 판단 대상 글
    # -in: questions = 질문 묶음
    #
    # -out: answers = 질문별 답 dict
    # -out: error = decide 실패 시 DecideUnavailable, 대기열 가득 차면 DecideBusy
    #------------------------------------------------------------------
    def _ask(self, state: str, questions: dict) -> dict:
        try:
            a, info = self.dc.ask(state, questions)
        except (DecideUnavailable, DecideBusy):
            raise
        except Exception as e:  # 대기열 없이 직접 부른 경우(평가·CLI)도 같은 방식으로 처리
            raise DecideUnavailable(f"{type(e).__name__}: {str(e)[:200]}") from e
        self.stats.add(info["latency"], info["input_tokens"], info.get("wait", 0.0))
        return a

    #------------------------------------------------------------------
    # 청크 관련성 판단
    #=> 청크마다 "이 조각에 질문의 답이 들어 있는가?" 의 참 확률을 구한다.
    #   한 요청에 여러 질문을 넣으면 서버가 질문마다 state 를 다시 인코딩해 느리므로 청크마다 따로 보낸다.
    #
    # -in: question = 판단용 질문
    # -in: chunks = 후보 청크 목록
    #
    # -out: probs = 청크 순서대로 0~1 확률 목록
    # -out: error = 서버 실패 시 RuntimeError
    #------------------------------------------------------------------
    def rerank(self, question: str, chunks: list[dict]) -> list[float]:
        q = {"rel": {"type": "noul", "instructions": "이 문서 조각에 질문에 대한 답이 들어 있는가?"}}
        return [self._ask(f"질문: {question}\n\n문서 조각:\n{c['text'][:self.clip]}", q)["rel"]["noul"]
                for c in chunks]

    #------------------------------------------------------------------
    # 근거 충분성 판단
    #=> 상위 근거를 묶어 "이 조각들만으로 정확히 답할 수 있는가?" 의 참 확률을 구한다.
    #
    # -in: question = 판단용 질문
    # -in: chunks = 답변에 쓸 상위 청크
    #
    # -out: p = 0~1 확률 (청크가 없으면 0.0)
    # -out: error = 서버 실패 시 RuntimeError
    #------------------------------------------------------------------
    def sufficient(self, question: str, chunks: list[dict]) -> float:
        if not chunks:
            return 0.0
        q = {"suff": {"type": "noul",
                      "instructions": "제시된 문서 조각들만으로 질문에 정확하게 답할 수 있는가?"}}
        return self._ask(f"질문: {question}\n\n문서 조각:\n{numbered(chunks, self.clip)}", q)["suff"]["noul"]

    #------------------------------------------------------------------
    # 추론이 필요한 질문인지 판단 (4단계 A1)
    #=> "계산이나 조건 비교가 필요한 질문인가?" 의 참 확률을 구한다.
    #   확률이 높은 질문만 Qwen 생각 모드로 답하게 해서, 단순 질문의 답변 시간은 늘리지 않는다.
    #
    # -in: question = 판단용 질문 (멀티턴이면 이전 대화 포함)
    #
    # -out: p = 0~1 확률
    # -out: error = decide 실패 시 DecideUnavailable, 대기열 가득 차면 DecideBusy
    #------------------------------------------------------------------
    def needs_reasoning(self, question: str) -> float:
        q = {"reason": {"type": "noul",
                        "instructions": "이 질문에 답하려면 금액·기간 계산이나 여러 조건의 비교가 필요한가?"}}
        return self._ask(f"질문: {question}", q)["reason"]["noul"]

    #------------------------------------------------------------------
    # 라우팅 후보 문서 고르기 (설계서 12.1)
    #=> decide 에 문서 10만 개를 다 보여 줄 수 없으므로(문서당 약 50토큰) 후보를 먼저 좁힌다.
    #    1) 지금 근거 상위 청크가 속한 문서 (순서대로, 중복 제거)
    #    2) 문서 색인 BM25 상위 문서로 n_cand 개까지 채움 (권한 필터 적용)
    #
    # -in: query = 검색 질의
    # -in: ctx = 지금 근거 상위 청크
    # -in: acl = 사용자 권한 그룹 (None = 필터 없음)
    #
    # -out: cands = [{"doc_id","title","desc",…}, …] 최대 n_cand 개
    # -out: error = 엔진 오류 시 예외 전파
    #------------------------------------------------------------------
    def candidates(self, query: str, ctx: list[dict], acl=None) -> list[dict]:
        ids = list(dict.fromkeys(c["doc_id"] for c in ctx))[: self.n_cand]
        cands = self.index.docs_by_id(ids)
        have = {d["doc_id"] for d in cands}
        # 질의는 라운드가 바뀌어도 같으므로 문서 검색은 질문당 한 번만 한다 (100만 청크 측정에서 회당 약 5ms)
        key = (query, tuple(acl) if acl else None)
        if key not in self._doc_hits:
            self._doc_hits[key] = self.index.search_docs(query, self.n_cand, acl)
        for d in self._doc_hits[key]:
            if len(cands) >= self.n_cand:
                break
            if d["doc_id"] not in have:
                cands.append(d)
                have.add(d["doc_id"])
        return cands

    #------------------------------------------------------------------
    # 문서 라우팅
    #=> "이 질문은 어느 문서를 찾아봐야 하는가?" 를 choice 로 물어 후보 문서별 확률을 받는다.
    #
    # -in: question = 판단용 질문
    # -in: cands = candidates() 결과
    #
    # -out: probs = {doc_id: 확률}
    # -out: error = 서버 실패 시 RuntimeError
    #------------------------------------------------------------------
    def route(self, question: str, cands: list[dict]) -> dict:
        crit = {d["doc_id"]: f"{d['title']} — {d['desc']}" for d in cands}
        q = {"doc": {"type": "choice", "instructions": "이 질문에 답하려면 어느 문서를 찾아봐야 하는가?",
                     "criteria": crit}}
        return self._ask(f"질문: {question}", q)["doc"]["probabilities"]

    #------------------------------------------------------------------
    # 멀티턴 검색 질의 만들기 (규칙)
    #=> decide 는 글을 쓸 수 없으므로 최근 이전 사용자 질문과 현재 질문을 이어 붙여 검색어로 쓴다.
    #   "그럼 배우자 부모님은?" 처럼 생략된 주제어를 앞 질문에서 가져오기 위함.
    #
    # -in: question = 현재 질문
    # -in: history = [{"q","a"}, …] 이전 대화 (없으면 빈 목록)
    #
    # -out: query = 검색 질의 문자열
    # -out: error = 없음
    #------------------------------------------------------------------
    def contextualize(self, question: str, history: list[dict]) -> str:
        prev = [h["q"] for h in history or []]
        # 대화가 길면 오래된 주제가 검색을 흐리므로 최근 N 개만 쓴다 (0 = 전부)
        if self.history_turns:
            prev = prev[-self.history_turns:]
        return " ".join(prev + [question])

    #------------------------------------------------------------------
    # 재검색 계획 (규칙 + decide 라우팅)
    #=> 생성 없이 재검색 질의를 만든다.
    #    1) 후보 문서(최대 20개) 중 decide 라우팅으로 상위 문서를 고른다 → 그 문서 안에서 같은 질의로 다시 검색
    #    2) 관련성 상위 청크 2개에서 PRF 확장어를 뽑아 질의 + 확장어로 전체 검색
    #    3) 질의 + 1순위 문서 제목으로 전체 검색 (제목 단어가 질문에 없을 때 도움)
    #   후보가 하나도 없으면(질의 단어가 어느 문서와도 안 맞고 근거도 없을 때) 1)·3) 은 빼고 2) 만 한다.
    #
    # -in: question = 판단용 질문 (멀티턴이면 이전 대화 포함)
    # -in: query = 지금 검색 질의
    # -in: ctx = 지금까지 관련성 상위 청크 (PRF 재료·후보 문서)
    # -in: acl = 사용자 권한 그룹 (None = 필터 없음)
    #
    # -out: (plan, info) = [(질의, 문서ID 집합 또는 None), …], 기록용 dict
    # -out: error = 서버·엔진 실패 시 예외 전파
    #------------------------------------------------------------------
    def research_plan(self, question: str, query: str, ctx: list[dict], acl=None) -> tuple[list, dict]:
        terms = self.index.prf_terms([c["chunk_id"] for c in ctx[:2]], query, self.n_prf)
        cands = self.candidates(query, ctx, acl)
        if not cands:
            return [(query + " " + " ".join(terms), None)], {"route": {}, "prf": terms, "candidates": 0}
        probs = self.route(question, cands)
        top = sorted(probs, key=lambda k: -probs[k])[: self.route_top]
        # 1순위는 항상 쓰고, 2순위부터는 확률이 너무 낮으면 범위에서 뺀다
        top = top[:1] + [k for k in top[1:] if probs[k] >= self.route_min]
        title = next(d["title"] for d in cands if d["doc_id"] == top[0])
        plan = [(query, set(top)), (query + " " + " ".join(terms), None), (query + " " + title, None)]
        # 후보 문서 ID 도 남긴다: 권한 시험에서 "볼 수 없는 문서의 제목이 decide 에 갔는가" 를 확인하는 근거
        return plan, {"route": {k: round(probs[k], 3) for k in top}, "prf": terms, "candidates": len(cands),
                      "candidate_ids": [d["doc_id"] for d in cands]}
