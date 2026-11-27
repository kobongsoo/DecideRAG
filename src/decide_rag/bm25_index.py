# -*- coding: utf-8 -*-
"""3단계: BM25 검색 (Kiwi 형태소 토큰)."""
import math
from collections import Counter

import threading

from kiwipiepy import Kiwi
from rank_bm25 import BM25Okapi

# 검색에 쓸 품사: 일반/고유명사, 수사, 어근, 동사·형용사 어간, 외국어, 한자, 숫자
KEEP_TAGS = {"NNG", "NNP", "NR", "XR", "VV", "VA", "SL", "SH", "SN", "NNB"}
# 의미가 약해 점수만 흐리는 의존명사·어간
_KIWI_LOCK = threading.Lock()
STOP = {"것", "수", "등", "때", "경우", "하", "있", "되", "이", "없", "같", "위하"}


#------------------------------------------------------------------
# 문장 → 검색 토큰 (공용)
#=> 형태소 분석 후 KEEP_TAGS 품사만 남기고, 영문은 소문자로 맞춘다.
#   메모리 엔진과 OpenSearch 엔진이 같은 규칙을 쓰도록 함수로 뺐다.
#
# -in: kiwi = Kiwi 객체
# -in: text = 문장
#
# -out: tokens = 토큰 목록
# -out: error = 없음
#------------------------------------------------------------------
def ko_tokenize(kiwi, text: str) -> list[str]:
    out = []
    # HTTP 서버에서 여러 요청이 동시에 같은 Kiwi 를 쓸 수 있으므로 한 번에 하나만 분석한다 (질의당 1ms 안팎)
    with _KIWI_LOCK:
        toks = kiwi.tokenize(text)
    for t in toks:
        tag = t.tag.split("-")[0]  # VV-R 같은 불규칙 표시는 떼고 본다
        if tag in KEEP_TAGS and t.form not in STOP:
            out.append(t.form.lower())
    return out


#------------------------------------------------------------------
# 한국어 BM25 색인
#=> 청크 텍스트를 Kiwi 로 형태소 분석해 토큰 목록을 만들고 BM25Okapi 로 색인한다.
#   "경조사지원규정" 같은 붙여 쓴 말도 경조사/지원/규정 으로 쪼개져 질문 단어와 맞는다.
#
# -필드: chunks = 청크 목록 (색인 순서 = 점수 배열 순서)
# -필드: kiwi = 형태소 분석기
# -필드: bm25 = BM25Okapi 색인
# -필드: tokens = 청크별 토큰 목록 (질의 확장 PRF 에 다시 씀)
# -필드: df = 토큰별 문서 빈도 (PRF 가중치 계산용)
#------------------------------------------------------------------
class BM25Index:
    #------------------------------------------------------------------
    # 색인 만들기
    #=> 모든 청크를 토큰화하고 BM25 색인과 문서 빈도표를 만든다.
    #
    # -in: chunks = [{"chunk_id","doc_id","text",…}, …]
    # -in: kiwi = 같이 쓸 Kiwi 객체 (없으면 새로 만듦. 만들 때 1~2초 걸리므로 여러 색인이 하나를 나눠 쓴다)
    #
    # -out: 없음
    # -out: error = 청크가 비면 rank_bm25 가 ZeroDivisionError
    #------------------------------------------------------------------
    def __init__(self, chunks: list[dict], kiwi=None):
        self.chunks = chunks
        self.kiwi = kiwi or Kiwi()
        self.tokens = [self.tokenize(c["text"]) for c in chunks]
        self.bm25 = BM25Okapi(self.tokens)
        self.df = Counter(t for toks in self.tokens for t in set(toks))
        self.by_id = {c["chunk_id"]: i for i, c in enumerate(chunks)}

    #------------------------------------------------------------------
    # 문장 → 검색 토큰
    #=> 형태소 분석 후 KEEP_TAGS 품사만 남기고, 영문은 소문자로 맞춘다.
    #
    # -in: text = 문장
    #
    # -out: tokens = 토큰 목록
    # -out: error = 없음
    #------------------------------------------------------------------
    def tokenize(self, text: str) -> list[str]:
        return ko_tokenize(self.kiwi, text)

    #------------------------------------------------------------------
    # BM25 검색
    #=> 질의를 토큰화해 모든 청크 점수를 구하고 상위 k 개를 돌려준다.
    #   doc_ids 를 주면 그 문서들의 청크만 남긴다 (decide 문서 라우팅 결과로 범위 좁히기).
    #
    # -in: query = 질의 문장
    # -in: k = 돌려줄 개수
    # -in: doc_ids = 허용 문서 ID 집합 (None 이면 전체)
    # -in: exclude = 제외할 chunk_id 집합 (이미 본 청크)
    #
    # -out: hits = [(chunk dict, bm25 점수), …] 점수 내림차순, 점수 0 은 제외
    # -out: error = 없음
    #------------------------------------------------------------------
    def search(self, query: str, k: int = 10, doc_ids=None, exclude=None) -> list[tuple[dict, float]]:
        q = self.tokenize(query)
        if not q:
            return []
        scores = self.bm25.get_scores(q)
        order = sorted(range(len(scores)), key=lambda i: -scores[i])
        hits = []
        for i in order:
            if scores[i] <= 0:
                break
            c = self.chunks[i]
            if doc_ids and c["doc_id"] not in doc_ids:
                continue
            if exclude and c["chunk_id"] in exclude:
                continue
            hits.append((c, float(scores[i])))
            if len(hits) >= k:
                break
        return hits

    #------------------------------------------------------------------
    # 의사 적합성 피드백(PRF) 확장어 뽑기
    #=> 판단 모델이 "관련 있다" 고 본 청크에서 질문에 없는 핵심어를 골라 질의에 덧붙일 단어를 만든다.
    #   LLM 없이 질의를 바꾸는 방법: 정답 근처 문서가 쓰는 단어로 다시 찾게 한다.
    #    1) 청크 토큰마다 tf × idf 점수
    #    2) 질문에 이미 있는 토큰·한 글자 토큰·숫자는 뺀다
    #
    # -in: chunk_ids = 피드백으로 쓸 청크 ID 목록
    # -in: query = 원 질문 (이미 있는 단어 제외용)
    # -in: n = 뽑을 단어 수
    #
    # -out: terms = 확장 단어 목록
    # -out: error = 없음
    #------------------------------------------------------------------
    def prf_terms(self, chunk_ids: list[str], query: str, n: int = 5) -> list[str]:
        qset = set(self.tokenize(query))
        N = len(self.chunks)
        tf = Counter()
        for cid in chunk_ids:
            if cid in self.by_id:
                tf.update(self.tokens[self.by_id[cid]])
        scored = []
        for t, f in tf.items():
            if t in qset or len(t) < 2 or t.isdigit():
                continue
            scored.append((f * math.log(1 + N / (1 + self.df[t])), t))
        return [t for _, t in sorted(scored, reverse=True)[:n]]
