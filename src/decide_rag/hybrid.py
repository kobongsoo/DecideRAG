# -*- coding: utf-8 -*-
"""BM25 + 벡터 혼합 검색(RRF)과 리랭커 (설계서 4.3.1~4.3.3, 14.7).

서비스 (7단계): make_retriever 가 [vector] 설정대로 검색 엔진을 감싼다.
  BM25(OpenSearch) + 벡터(OpenSearch k-NN, 질의는 BGE-m3-ko 서버로 임베딩) → RRF → 리랭커 → decide 루프
  벡터·리랭커가 실패하면 그 단계를 건너뛰고(차단기 30초) 이유를 요청 기록에 남긴다.
구성 비교 (eval/compare_configs.py): 메모리 벡터 색인(VectorIndex)과 make_hybrid 를 쓴다.
"""
import hashlib
import json
import os
import threading
import time
from pathlib import Path

import numpy as np

QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "


#------------------------------------------------------------------
# e5 임베더 (ONNX)
#=> 문장을 e5-small-ko 로 벡터로 바꾼다. 프로토타입 vector_index.OnnxEmbedder 와 같은 방식이다.
#
# -필드: tok = 토크나이저
# -필드: sess = ONNX 세션
# -필드: max_len = 최대 토큰 수
#------------------------------------------------------------------
class E5Embedder:
    #------------------------------------------------------------------
    # 임베더 만들기
    #=> model.onnx·tokenizer.json 을 연다.
    #
    # -in: model_dir = 모델 폴더
    # -in: max_len = 최대 토큰 수 (기본 512)
    #
    # -out: 없음
    # -out: error = 파일이 없으면 onnxruntime/tokenizers 예외 전파
    #------------------------------------------------------------------
    def __init__(self, model_dir: str, max_len: int = 512):
        import onnxruntime as ort
        from tokenizers import Tokenizer
        self.max_len = max_len
        self.tok = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(os.path.join(model_dir, "model.onnx"), sess_options=so,
                                         providers=["CPUExecutionProvider"])
        self.inputs = {i.name for i in self.sess.get_inputs()}

    #------------------------------------------------------------------
    # 문장 목록 → 벡터
    #=> 배치 안 최장 길이로 패딩 → 추론 → 패딩 뺀 평균 → 길이 1 로 정규화.
    #
    # -in: texts = 문장 목록
    # -in: prefix = "query: " 또는 "passage: " (e5 규약)
    # -in: batch = 한 번에 넣을 문장 수 (기본 16)
    #
    # -out: vecs = (len(texts), 차원) float32
    # -out: error = 없음
    #------------------------------------------------------------------
    def embed(self, texts: list[str], prefix: str, batch: int = 16) -> np.ndarray:
        out = []
        for s in range(0, len(texts), batch):
            encs = [self.tok.encode(prefix + t, add_special_tokens=True) for t in texts[s:s + batch]]
            lens = [min(len(e.ids), self.max_len) for e in encs]
            w = max(lens)
            ids = np.zeros((len(encs), w), dtype=np.int64)
            att = np.zeros((len(encs), w), dtype=np.int64)
            for i, e in enumerate(encs):
                ids[i, :lens[i]] = e.ids[:lens[i]]
                att[i, :lens[i]] = e.attention_mask[:lens[i]]
            feeds = {"input_ids": ids, "attention_mask": att, "token_type_ids": np.zeros_like(ids)}
            hidden = self.sess.run(None, {k: v for k, v in feeds.items() if k in self.inputs})[0]
            # 패딩 위치는 평균에서 빼야 짧은 문장의 벡터가 흐려지지 않는다
            m = att.astype(np.float32)[:, :, None]
            v = (hidden * m).sum(1) / np.clip(m.sum(1), 1e-9, None)
            out.append(v / np.clip(np.linalg.norm(v, axis=1, keepdims=True), 1e-9, None))
        return np.vstack(out).astype(np.float32)


#------------------------------------------------------------------
# 벡터 색인 (메모리)
#=> 청크 글을 임베딩해 행렬로 두고, 질의와 내적(=코사인)이 큰 순으로 찾는다.
#   임베딩은 청크 글 해시별로 파일에 저장해 다시 쓴다 (청크 600개 임베딩은 약 1분).
#
# -필드: chunks = 청크 목록
# -필드: mat = (청크 수, 차원) 행렬
#------------------------------------------------------------------
class VectorIndex:
    #------------------------------------------------------------------
    # 색인 만들기
    #=> 캐시 파일에 있는 청크는 다시 임베딩하지 않는다.
    #
    # -in: chunks = 청크 목록 ({"chunk_id","doc_id","text",…})
    # -in: emb = E5Embedder 또는 HttpEmbedder (embed(texts, prefix) 를 가진 것)
    # -in: cache_file = 임베딩 캐시 (.npz, None 이면 캐시 안 씀)
    #
    # -out: 없음
    # -out: error = 없음
    #------------------------------------------------------------------
    def __init__(self, chunks: list[dict], emb, cache_file: Path | None = None):
        self.chunks, self.emb = chunks, emb
        keys = [hashlib.sha1(c["text"].encode("utf-8")).hexdigest() for c in chunks]
        have = {}
        if cache_file and cache_file.exists():
            z = np.load(cache_file)
            have = dict(zip(z["keys"].tolist(), z["vecs"]))
        todo = [i for i, k in enumerate(keys) if k not in have]
        if todo:
            vecs = emb.embed([chunks[i]["text"] for i in todo], PASSAGE_PREFIX)
            for i, v in zip(todo, vecs):
                have[keys[i]] = v
            if cache_file:
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                np.savez(cache_file, keys=np.array(list(have)), vecs=np.stack(list(have.values())))
        self.mat = np.stack([have[k] for k in keys])
        self.pos = {c["chunk_id"]: i for i, c in enumerate(chunks)}

    #------------------------------------------------------------------
    # 벡터 검색
    #=> 질의와 가장 가까운 청크 k 개. 문서 제한·제외 청크·권한 필터를 BM25 검색과 같은 뜻으로 적용한다.
    #
    # -in: query = 질의
    # -in: k = 개수
    # -in: doc_ids = 이 문서들 안에서만 (None = 전부)
    # -in: exclude = 뺄 chunk_id 집합
    # -in: acl = 사용자 그룹 (None = 필터 없음, 청크의 "acl" 필드와 겹쳐야 한다)
    #
    # -out: hits = [(청크, 코사인), …]
    # -out: error = 없음
    #------------------------------------------------------------------
    def search(self, query: str, k: int, doc_ids=None, exclude=None, acl=None) -> list[tuple[dict, float]]:
        q = self.emb.embed([query], QUERY_PREFIX)[0]
        sims = self.mat @ q
        out = []
        for i in np.argsort(-sims):
            c = self.chunks[i]
            if doc_ids and c["doc_id"] not in doc_ids:
                continue
            if exclude and c["chunk_id"] in exclude:
                continue
            if acl is not None and c.get("acl") and not set(c["acl"]) & set(acl):
                continue
            out.append((c, float(sims[i])))
            if len(out) >= k:
                break
        return out


#------------------------------------------------------------------
# 요청별 기록 (스레드마다)
#=> 서비스는 요청을 스레드로 동시에 처리하고, 혼합·리랭크 엔진은 요청끼리 함께 쓴다.
#   그래서 "이 요청에서 벡터·리랭커가 걸린 시간"과 "건너뛴 이유"는 스레드마다 따로 모은다.
#   run_agentic 이 질문 시작에 reset_notes(), 끝에 take_notes() 를 부른다.
#------------------------------------------------------------------
_local = threading.local()


#------------------------------------------------------------------
# 요청별 기록 비우기
#=> 질문을 시작할 때 부른다.
#
# -in: 없음
#
# -out: 없음
# -out: error = 없음
#------------------------------------------------------------------
def reset_notes():
    _local.notes = {"embed": 0.0, "vector": 0.0, "rerank": 0.0, "skipped": []}


#------------------------------------------------------------------
# 요청별 기록 꺼내기
#=> {"embed","vector","rerank" (초), "skipped": [건너뛴 이유]} 를 돌려주고 비운다.
#
# -in: 없음
#
# -out: notes = 기록 dict
# -out: error = 없음
#------------------------------------------------------------------
def take_notes() -> dict:
    n = getattr(_local, "notes", None) or {"embed": 0.0, "vector": 0.0, "rerank": 0.0, "skipped": []}
    reset_notes()
    return n


#------------------------------------------------------------------
# 요청별 기록 더하기 (안쪽 도우미)
#=> 시간은 더하고, 건너뛴 이유는 처음 한 번만 남긴다.
#
# -in: key = "embed"/"vector"/"rerank"/"skipped"
# -in: value = 초 또는 이유 문자열
#
# -out: 없음
# -out: error = 없음
#------------------------------------------------------------------
def _note(key: str, value):
    if getattr(_local, "notes", None) is None:
        reset_notes()
    if key == "skipped":
        if value not in _local.notes["skipped"]:
            _local.notes["skipped"].append(value)
    else:
        _local.notes[key] += value


#------------------------------------------------------------------
# 차단기 (외부 서버 하나)
#=> 서버가 실패하면 cooldown 초 동안은 부르지 않고 바로 건너뛴다 (꺼진 서버를 검색마다 기다리지 않게).
#
# -필드: name = 서버 이름 (기록용)
# -필드: cooldown = 차단 시간 (초)
# -필드: until = 차단이 풀리는 시각
# -필드: last_error = 마지막 오류
#------------------------------------------------------------------
class Breaker:
    #------------------------------------------------------------------
    # 만들기
    #=> 이름과 차단 시간을 받는다.
    #
    # -in: name = 서버 이름
    # -in: cooldown = 차단 시간 (초, 기본 30)
    #
    # -out: 없음
    # -out: error = 없음
    #------------------------------------------------------------------
    def __init__(self, name: str, cooldown: float = 30.0):
        self.name, self.cooldown, self.until, self.last_error = name, cooldown, 0.0, None

    #------------------------------------------------------------------
    # 지금 부를 수 있는가
    #=> 차단 중이면 False.
    #
    # -in: 없음
    #
    # -out: ok = True/False
    # -out: error = 없음
    #------------------------------------------------------------------
    def ok(self) -> bool:
        return time.time() >= self.until

    #------------------------------------------------------------------
    # 실패 기록
    #=> 차단을 시작하고 오류를 남긴다.
    #
    # -in: e = 예외
    #
    # -out: 없음
    # -out: error = 없음
    #------------------------------------------------------------------
    def fail(self, e: Exception):
        self.until = time.time() + self.cooldown
        self.last_error = f"{type(e).__name__}: {str(e)[:200]}"

    #------------------------------------------------------------------
    # 상태
    #=> /health 용.
    #
    # -in: 없음
    #
    # -out: {"down_for_sec","last_error"}
    # -out: error = 없음
    #------------------------------------------------------------------
    def status(self) -> dict:
        return {"down_for_sec": round(max(0.0, self.until - time.time()), 1), "last_error": self.last_error}


#------------------------------------------------------------------
# OpenSearch k-NN 벡터 소스 (7단계)
#=> 질의를 임베딩 서버로 벡터로 바꾸고 OpenSearch 의 청크 벡터(k-NN)에서 찾는다.
#   VectorIndex(메모리)와 같은 search / search_many 모양.
#
# -필드: engine = OpenSearchEngine (vector_dim 이 있어야 함)
# -필드: emb = HttpEmbedder
#------------------------------------------------------------------
class OpenSearchVectorSource:
    #------------------------------------------------------------------
    # 만들기
    #=> 엔진과 임베더를 받는다.
    #
    # -in: engine = OpenSearchEngine
    # -in: emb = 임베더 (embed(texts, prefix))
    #
    # -out: 없음
    # -out: error = 없음
    #------------------------------------------------------------------
    def __init__(self, engine, emb):
        self.engine, self.emb = engine, emb

    #------------------------------------------------------------------
    # 여러 질의 벡터 검색
    #=> 질의들을 한 번에 임베딩하고 k-NN 을 _msearch 한 번으로 보낸다.
    #
    # -in: reqs = [(질의, k, doc_ids, exclude), …]
    # -in: acl = 사용자 그룹
    #
    # -out: results = 질의마다 [(청크, 점수), …]
    # -out: error = 임베딩·OpenSearch 오류 전파 (HybridEngine 이 잡는다)
    #------------------------------------------------------------------
    def search_many(self, reqs: list[tuple], acl=None):
        t = time.perf_counter()
        vecs = self.emb.embed([q for q, _, _, _ in reqs], QUERY_PREFIX)
        _note("embed", time.perf_counter() - t)
        return self.engine.knn_many([(v, k, d, ex) for v, (_, k, d, ex) in zip(vecs, reqs)], acl)

    #------------------------------------------------------------------
    # 질의 하나 벡터 검색
    #=> search_many 의 한 개짜리.
    #
    # -in: query, k, doc_ids, exclude, acl = VectorIndex.search 와 같음
    #
    # -out: hits = [(청크, 점수), …]
    # -out: error = 오류 전파
    #------------------------------------------------------------------
    def search(self, query: str, k: int, doc_ids=None, exclude=None, acl=None):
        return self.search_many([(query, k, doc_ids, exclude)], acl)[0]


#------------------------------------------------------------------
# BM25 + 벡터 혼합 엔진 (RRF)
#=> 기존 검색 엔진(OpenSearch 등)을 감싼다. 청크 검색만 두 검색기의 RRF 로 바꾸고, 나머지(문서 후보·PRF·ID 조회)는 그대로 넘긴다.
#   그래서 decide 루프(run_agentic)에 검색 엔진 대신 넣으면 "BM25+벡터+decide" 가 된다.
#   벡터 쪽(임베딩·k-NN)이 실패하면 BM25 결과만 쓰고 이유를 요청 기록에 남긴다 (설계서 8.2).
#
# -필드: base = 감싼 검색 엔진
# -필드: vec = 벡터 소스 (VectorIndex 또는 OpenSearchVectorSource)
# -필드: pool = 검색기마다 받을 후보 수 (기본 20, 프로토타입과 같음)
# -필드: rrf_k = RRF 상수 (기본 60)
# -필드: breaker = 벡터 쪽 차단기
#------------------------------------------------------------------
class HybridEngine:
    #------------------------------------------------------------------
    # 만들기
    #=> 엔진·벡터 소스·RRF 값을 받는다.
    #
    # -in: base = 검색 엔진
    # -in: vec = 벡터 소스
    # -in: pool = 검색기별 후보 수
    # -in: rrf_k = RRF 상수
    # -in: cooldown = 벡터 쪽 실패 후 건너뛸 시간 (초, 기본 30)
    #
    # -out: 없음
    # -out: error = 없음
    #------------------------------------------------------------------
    def __init__(self, base, vec, pool: int = 20, rrf_k: int = 60, cooldown: float = 30.0):
        self.base, self.vec, self.pool, self.rrf_k = base, vec, pool, rrf_k
        self.name = f"hybrid({base.name})"
        self.breaker = Breaker("vector", cooldown)

    #------------------------------------------------------------------
    # 감싼 엔진의 나머지 기능은 그대로 쓴다
    #=> search_docs, get_chunks, prf_terms, docs_by_id, counts 등.
    #
    # -in: attr = 속성 이름
    #
    # -out: 감싼 엔진의 속성
    # -out: error = 없으면 AttributeError
    #------------------------------------------------------------------
    def __getattr__(self, attr):
        return getattr(self.base, attr)

    #------------------------------------------------------------------
    # 벡터 쪽 검색 (실패하면 빈 결과)
    #=> 차단 중이거나 오류가 나면 질의마다 빈 목록을 돌려주고 이유를 남긴다.
    #
    # -in: reqs = [(질의, k, doc_ids, exclude), …] (k 는 이미 pool 로 늘린 값)
    # -in: acl = 사용자 그룹
    #
    # -out: results = 질의마다 [(청크, 점수), …] (실패 시 빈 목록들)
    # -out: error = 없음 (예외를 밖으로 내지 않는다)
    #------------------------------------------------------------------
    def _vector(self, reqs: list[tuple], acl) -> list[list]:
        if not self.breaker.ok():
            _note("skipped", f"vector: 차단 중 ({self.breaker.last_error})")
            return [[] for _ in reqs]
        t = time.perf_counter()
        try:
            if hasattr(self.vec, "search_many"):
                out = self.vec.search_many(reqs, acl)
            else:
                out = [self.vec.search(q, k, d, ex, acl) for q, k, d, ex in reqs]
        except Exception as e:  # 임베딩·k-NN 장애: BM25 만으로 답한다
            self.breaker.fail(e)
            _note("skipped", f"vector: {type(e).__name__}: {str(e)[:150]}")
            return [[] for _ in reqs]
        _note("vector", time.perf_counter() - t)
        return out

    #------------------------------------------------------------------
    # 여러 질의 혼합 검색
    #=> 질의마다 BM25 max(pool,k) 개 + 벡터 max(pool,k) 개를 받아 RRF(1/(rrf_k+순위)) 합으로 상위 k 개.
    #   BM25 쪽은 감싼 엔진의 search_many 로 한 번에 보낸다. 벡터 쪽이 비면 BM25 순서 그대로가 된다.
    #
    # -in: reqs = [(질의, k, doc_ids, exclude), …]
    # -in: acl = 사용자 그룹
    #
    # -out: results = 질의마다 [(청크, RRF 점수), …]
    # -out: error = BM25 엔진 오류 전파
    #------------------------------------------------------------------
    def search_many(self, reqs: list[tuple], acl=None) -> list[list[tuple[dict, float]]]:
        wide = [(q, max(self.pool, k), d, ex) for q, k, d, ex in reqs]
        bm = self.base.search_many(wide, acl)
        vv = self._vector(wide, acl)
        out = []
        for (q, k, d, ex), b, v in zip(reqs, bm, vv):
            score, by = {}, {}
            for hits in (b, v):
                for rank, (c, _) in enumerate(hits, 1):
                    score[c["chunk_id"]] = score.get(c["chunk_id"], 0.0) + 1.0 / (self.rrf_k + rank)
                    by.setdefault(c["chunk_id"], c)
            top = sorted(score, key=lambda cid: -score[cid])[:k]
            out.append([(by[cid], score[cid]) for cid in top])
        return out

    #------------------------------------------------------------------
    # 질의 하나 혼합 검색
    #=> search_many 의 한 개짜리.
    #
    # -in: query, k, doc_ids, exclude, acl = 감싼 엔진의 search 와 같음
    #
    # -out: hits = [(청크, RRF 점수), …]
    # -out: error = 엔진 오류 전파
    #------------------------------------------------------------------
    def search(self, query: str, k: int = 10, doc_ids=None, exclude=None, acl=None):
        return self.search_many([(query, k, doc_ids, exclude)], acl)[0]


#------------------------------------------------------------------
# 크로스인코더 리랭커 (ONNX)
#=> (질문, 청크) 쌍마다 관련성 점수를 낸다. SimpleRAG 의 bge-reranker-base-int8 과 같은 모델·입력 방식.
#
# -필드: tok = 토크나이저 (질문+청크 512토큰에서 자름)
# -필드: sess = ONNX 세션
#------------------------------------------------------------------
class Reranker:
    #------------------------------------------------------------------
    # 리랭커 만들기
    #=> model_int8.onnx·tokenizer.json 을 연다.
    #
    # -in: model_dir = 모델 폴더
    # -in: max_len = 질문+청크 최대 토큰 (기본 512)
    #
    # -out: 없음
    # -out: error = 파일이 없으면 예외 전파
    #------------------------------------------------------------------
    def __init__(self, model_dir: str, max_len: int = 512):
        import onnxruntime as ort
        from tokenizers import Tokenizer
        self.tok = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
        self.tok.enable_truncation(max_length=max_len)
        self.tok.enable_padding()
        onnx = next(p for p in ("model_int8.onnx", "model.onnx") if os.path.exists(os.path.join(model_dir, p)))
        self.sess = ort.InferenceSession(os.path.join(model_dir, onnx), providers=["CPUExecutionProvider"])
        self.inputs = {i.name for i in self.sess.get_inputs()}
        self.name = Path(model_dir).name

    #------------------------------------------------------------------
    # 점수 매기기
    #=> 쌍을 한 번에 넣어 로짓을 받는다. 클수록 관련이 높다.
    #
    # -in: query = 질문
    # -in: passages = 청크 글 목록
    # -in: batch = 한 번에 넣을 쌍 수 (기본 8)
    #
    # -out: scores = passages 순서대로 점수 목록
    # -out: error = 추론 실패 시 onnxruntime 예외 전파
    #------------------------------------------------------------------
    def score(self, query: str, passages: list[str], batch: int = 8) -> list[float]:
        out = []
        for s in range(0, len(passages), batch):
            enc = self.tok.encode_batch([(query, p) for p in passages[s:s + batch]])
            feeds = {"input_ids": np.array([e.ids for e in enc], dtype=np.int64),
                     "attention_mask": np.array([e.attention_mask for e in enc], dtype=np.int64)}
            if "token_type_ids" in self.inputs:
                feeds["token_type_ids"] = np.zeros_like(feeds["input_ids"])
            out += self.sess.run(None, {k: v for k, v in feeds.items() if k in self.inputs})[0].reshape(-1).tolist()
        return out


#------------------------------------------------------------------
# HTTP 리랭커 (vLLM /v1/rerank)
#=> A30 서버에 띄운 한국어 리랭커(bge-reranker-v2-m3-ko)를 부른다. Reranker 와 같은 score() 모양.
#
# -필드: url = /v1/rerank 주소
# -필드: model = 서버의 모델 이름
# -필드: session = HTTP 세션
#------------------------------------------------------------------
class HttpReranker:
    #------------------------------------------------------------------
    # 만들기
    #=> 서버 주소와 모델 이름을 받는다.
    #
    # -in: url = 서버 주소 (예: http://10.10.4.10:18003)
    # -in: model = 모델 이름 (기본 bge-reranker-v2-m3-ko)
    # -in: timeout = 응답 대기 (초, 기본 30)
    #
    # -out: 없음
    # -out: error = 없음
    #------------------------------------------------------------------
    def __init__(self, url: str, model: str = "bge-reranker-v2-m3-ko", timeout: float = 30):
        import requests
        self.url = url.rstrip("/") + "/v1/rerank"
        self.model, self.timeout, self.name = model, timeout, model
        self.session = requests.Session()

    #------------------------------------------------------------------
    # 점수 매기기
    #=> 한 번에 보내고, 결과(관련성 순)를 입력 순서로 되돌린다.
    #
    # -in: query = 질문
    # -in: passages = 청크 글 목록
    #
    # -out: scores = passages 순서대로 relevance_score 목록
    # -out: error = 서버 오류 시 requests 예외 전파
    #------------------------------------------------------------------
    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        r = self.session.post(self.url, json={"model": self.model, "query": query, "documents": passages},
                              timeout=self.timeout)
        r.raise_for_status()
        out = [0.0] * len(passages)
        for x in r.json()["results"]:
            out[x["index"]] = x["relevance_score"]
        return out


#------------------------------------------------------------------
# HTTP 임베더 (OpenAI 호환 /v1/embeddings, 예: A30 서버의 vLLM bge-m3)
#=> E5Embedder 와 같은 embed() 모양. bge-m3 는 질의·문서 접두어를 쓰지 않으므로 prefix 는 무시한다.
#
# -필드: url = /v1/embeddings 주소
# -필드: model = 서버의 모델 이름
# -필드: session = HTTP 세션
#------------------------------------------------------------------
class HttpEmbedder:
    #------------------------------------------------------------------
    # 만들기
    #=> 서버 주소와 모델 이름을 받는다.
    #
    # -in: url = 서버 주소 (예: http://10.10.4.10:18002)
    # -in: model = 모델 이름 (기본 bge-m3)
    # -in: timeout = 응답 대기 (초, 기본 60)
    #
    # -out: 없음
    # -out: error = 없음
    #------------------------------------------------------------------
    def __init__(self, url: str, model: str = "bge-m3", timeout: float = 60):
        import requests
        self.url = url.rstrip("/") + "/v1/embeddings"
        self.model, self.timeout, self.name = model, timeout, model
        self.session = requests.Session()

    #------------------------------------------------------------------
    # 문장 목록 → 벡터
    #=> batch 개씩 보내 받은 벡터를 길이 1 로 정규화한다 (내적 = 코사인이 되게).
    #
    # -in: texts = 문장 목록
    # -in: prefix = 쓰지 않음 (E5Embedder 와 모양을 맞추려고 받는다)
    # -in: batch = 한 번에 보낼 문장 수 (기본 32)
    #
    # -out: vecs = (len(texts), 차원) float32
    # -out: error = 서버 오류 시 requests 예외 전파
    #------------------------------------------------------------------
    def embed(self, texts: list[str], prefix: str = "", batch: int = 32) -> np.ndarray:
        out = []
        for s in range(0, len(texts), batch):
            r = self.session.post(self.url, json={"model": self.model, "input": texts[s:s + batch]}, timeout=self.timeout)
            r.raise_for_status()
            # 서버가 순서를 섞어 줄 수 있으므로 index 로 되돌린다
            data = sorted(r.json()["data"], key=lambda x: x["index"])
            v = np.array([d["embedding"] for d in data], dtype=np.float32)
            out.append(v / np.clip(np.linalg.norm(v, axis=1, keepdims=True), 1e-9, None))
        return np.vstack(out)


#------------------------------------------------------------------
# 리랭크 엔진 (검색 결과를 리랭커로 다시 줄 세움)
#=> 감싼 엔진(보통 HybridEngine)에서 질의마다 max(candidates, k) 개를 받아 리랭커 점수순 상위 k 개를 돌려준다.
#   decide 루프에 넣으면 "BM25+벡터+리랭커+decide" 가 된다 (decide 는 리랭커가 고른 후보를 판단한다).
#   리랭커가 실패하면 감싼 엔진의 순서(RRF)를 그대로 쓰고 이유를 요청 기록에 남긴다 (설계서 8.2).
#
# -필드: base = 감싼 엔진
# -필드: reranker = Reranker 또는 HttpReranker
# -필드: candidates = 리랭커에 넣을 후보 수
# -필드: breaker = 리랭커 차단기
#------------------------------------------------------------------
class RerankEngine:
    #------------------------------------------------------------------
    # 만들기
    #=> 감싼 엔진과 리랭커를 받는다.
    #
    # -in: base = 엔진
    # -in: reranker = score(query, passages) 를 가진 리랭커
    # -in: candidates = 후보 수 (기본 20)
    # -in: cooldown = 리랭커 실패 후 건너뛸 시간 (초, 기본 30)
    #
    # -out: 없음
    # -out: error = 없음
    #------------------------------------------------------------------
    def __init__(self, base, reranker, candidates: int = 20, cooldown: float = 30.0):
        self.base, self.reranker, self.candidates = base, reranker, candidates
        self.name = f"rerank({base.name})"
        self.breaker = Breaker("rerank", cooldown)

    #------------------------------------------------------------------
    # 감싼 엔진의 나머지 기능은 그대로 쓴다
    #=> search_docs, get_chunks, prf_terms 등.
    #
    # -in: attr = 속성 이름
    #
    # -out: 감싼 엔진의 속성
    # -out: error = 없으면 AttributeError
    #------------------------------------------------------------------
    def __getattr__(self, attr):
        return getattr(self.base, attr)

    #------------------------------------------------------------------
    # 지금까지 이 요청에서 리랭커에 쓴 시간 (비교 스크립트 호환)
    #=> 요청별 기록의 rerank 값.
    #
    # -in: 없음
    #
    # -out: 초
    # -out: error = 없음
    #------------------------------------------------------------------
    @property
    def rerank_sec(self) -> float:
        n = getattr(_local, "notes", None)
        return n["rerank"] if n else 0.0

    @rerank_sec.setter
    def rerank_sec(self, v):
        # 비교 스크립트가 질문마다 0 으로 돌리던 자리 → 요청별 기록 비우기로 대신한다
        reset_notes()

    #------------------------------------------------------------------
    # 여러 질의 검색 + 리랭크
    #=> 질의마다 후보를 받아 리랭커 점수순으로 상위 k 개. 리랭커가 안 되면 후보 순서 상위 k 개.
    #
    # -in: reqs = [(질의, k, doc_ids, exclude), …]
    # -in: acl = 사용자 그룹
    #
    # -out: results = 질의마다 [(청크, 점수), …]
    # -out: error = 감싼 엔진 오류 전파 (리랭커 오류는 잡는다)
    #------------------------------------------------------------------
    def search_many(self, reqs: list[tuple], acl=None) -> list[list[tuple[dict, float]]]:
        got = self.base.search_many([(q, max(self.candidates, k), d, ex) for q, k, d, ex in reqs], acl)
        out = []
        for (q, k, _, _), hits in zip(reqs, got):
            if not hits:
                out.append([])
                continue
            if not self.breaker.ok():
                _note("skipped", f"rerank: 차단 중 ({self.breaker.last_error})")
                out.append(hits[:k])
                continue
            t = time.perf_counter()
            try:
                sc = self.reranker.score(q, [c["text"] for c, _ in hits])
            except Exception as e:  # 리랭커 장애: 혼합 검색 순서로 답한다
                self.breaker.fail(e)
                _note("skipped", f"rerank: {type(e).__name__}: {str(e)[:150]}")
                out.append(hits[:k])
                continue
            _note("rerank", time.perf_counter() - t)
            out.append(sorted([(c, s) for (c, _), s in zip(hits, sc)], key=lambda x: -x[1])[:k])
        return out

    #------------------------------------------------------------------
    # 질의 하나 검색 + 리랭크
    #=> search_many 의 한 개짜리.
    #
    # -in: query, k, doc_ids, exclude, acl = 엔진 search 와 같음
    #
    # -out: hits = [(청크, 리랭커 점수), …]
    # -out: error = 엔진 오류 전파
    #------------------------------------------------------------------
    def search(self, query: str, k: int = 10, doc_ids=None, exclude=None, acl=None):
        return self.search_many([(query, k, doc_ids, exclude)], acl)[0]


#------------------------------------------------------------------
# 임베딩 캐시 (청크 글 해시 → 벡터)
#=> 색인(indexer)과 비교 스크립트가 같이 쓴다. 같은 글은 다시 임베딩하지 않는다.
#   파일은 모델마다 하나 (data\vectors\<모델>.npz).
#
# -필드: path = 캐시 파일
# -필드: have = {글 해시: 벡터}
# -필드: dirty = 저장할 것이 있는지
#------------------------------------------------------------------
class VectorCache:
    #------------------------------------------------------------------
    # 캐시 열기
    #=> 파일이 있으면 읽는다.
    #
    # -in: path = 캐시 파일 (.npz)
    #
    # -out: 없음
    # -out: error = 파일이 깨졌으면 예외 전파
    #------------------------------------------------------------------
    def __init__(self, path: Path):
        self.path, self.have, self.dirty = path, {}, False
        if path.exists():
            z = np.load(path)
            self.have = dict(zip(z["keys"].tolist(), z["vecs"]))

    #------------------------------------------------------------------
    # 글 목록 → 벡터 (캐시에 없는 것만 임베딩)
    #=> 글 순서대로 벡터 행렬을 돌려준다.
    #
    # -in: texts = 글 목록
    # -in: emb = 임베더
    #
    # -out: vecs = (len(texts), 차원) float32
    # -out: error = 임베딩 서버 오류 전파
    #------------------------------------------------------------------
    def vectors(self, texts: list[str], emb) -> np.ndarray:
        keys = [hashlib.sha1(t.encode("utf-8")).hexdigest() for t in texts]
        todo = [i for i, k in enumerate(keys) if k not in self.have]
        if todo:
            for i, v in zip(todo, emb.embed([texts[i] for i in todo], PASSAGE_PREFIX)):
                self.have[keys[i]] = v
            self.dirty = True
        return np.stack([self.have[k] for k in keys]) if keys else np.zeros((0, 0), np.float32)

    #------------------------------------------------------------------
    # 저장
    #=> 바뀐 것이 있을 때만 임시 파일에 쓰고 바꿔 끼운다.
    #
    # -in: 없음
    #
    # -out: 없음
    # -out: error = 쓰기 실패 시 OSError 전파
    #------------------------------------------------------------------
    def save(self):
        if not self.dirty or not self.have:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.stem + ".tmp.npz")
        np.savez(tmp, keys=np.array(list(self.have)), vecs=np.stack(list(self.have.values())))
        tmp.replace(self.path)
        self.dirty = False


#------------------------------------------------------------------
# 서비스용 임베더 만들기
#=> [vector] embed_url·embed_model 로 HttpEmbedder 를 만든다.
#
# -in: cfg = 설정 dict
#
# -out: emb = HttpEmbedder
# -out: error = 없음
#------------------------------------------------------------------
def service_embedder(cfg: dict):
    v = cfg["vector"]
    return HttpEmbedder(v["embed_url"], v.get("embed_model", "bge-m3-ko"), v.get("timeout_sec", 30))


#------------------------------------------------------------------
# 서비스용 임베딩 캐시 경로
#=> data\vectors\<embed_model>.npz
#
# -in: cfg = 설정 dict
#
# -out: path = 캐시 파일 경로
# -out: error = 없음
#------------------------------------------------------------------
def service_cache_path(cfg: dict) -> Path:
    return Path(cfg["paths"]["data_dir"]) / "vectors" / f"{cfg['vector'].get('embed_model', 'bge-m3-ko')}.npz"


#------------------------------------------------------------------
# 서비스 검색 경로 만들기 (7단계, 설계서 4.3.1~4.3.3)
#=> [vector] enabled 면 BM25 엔진을 혼합 엔진으로 감싸고, rerank_enabled 면 그 위에 리랭크 엔진을 씌운다.
#    - OpenSearch: 벡터는 OpenSearch k-NN (색인할 때 넣은 청크 벡터)
#    - 메모리 엔진(시험용): 청크 파일 + 임베딩 캐시로 메모리 VectorIndex
#
# -in: cfg = 설정 dict
# -in: base = make_engine 결과
#
# -out: engine = base / HybridEngine / RerankEngine
# -out: error = 없음 (서버 연결은 첫 검색에서, 실패하면 건너뛴다)
#------------------------------------------------------------------
def make_retriever(cfg: dict, base):
    v = cfg.get("vector", {})
    eng = base
    if v.get("enabled"):
        emb = service_embedder(cfg)
        if getattr(base, "vector_dim", None):
            src = OpenSearchVectorSource(base, emb)
        else:
            from .chunker import load_chunks
            src = VectorIndex(load_chunks(cfg), emb, service_cache_path(cfg))
        eng = HybridEngine(base, src, v.get("pool", 20), v.get("rrf_k", 60), v.get("cooldown_sec", 30))
    if v.get("rerank_enabled"):
        rr = HttpReranker(v["reranker_url"], v.get("reranker_model", "bge-reranker-v2-m3-ko"), v.get("timeout_sec", 30))
        eng = RerankEngine(eng, rr, v.get("rerank_candidates", 20), v.get("cooldown_sec", 30))
    return eng


#------------------------------------------------------------------
# 혼합 엔진 만들기
#=> 저장된 청크 파일 전부로 벡터 색인을 만들고(임베딩 캐시 사용) 기존 엔진을 감싼다.
#
# -in: cfg = 설정 dict ([vector] model_dir, embed_url, embed_model, pool, rrf_k)
# -in: base = 감싼 검색 엔진
# -in: embed = "e5" (이 PC 의 ONNX) 또는 "m3" (서버의 bge-m3, embed_url)
#
# -out: engine = HybridEngine
# -out: error = 모델·서버·청크 파일이 없으면 예외 전파
#------------------------------------------------------------------
def make_hybrid(cfg: dict, base, embed: str = "e5") -> HybridEngine:
    from .chunker import load_chunks
    v = cfg["vector"]
    chunks = load_chunks(cfg)
    # 권한 필터도 벡터 쪽에서 같게 하려고 manifest 의 문서별 acl 을 청크에 붙인다
    mf = Path(cfg["paths"]["data_dir"]) / "manifest.json"
    acl = {d: e.get("acl") for d, e in json.loads(mf.read_text(encoding="utf-8")).items()} if mf.exists() else {}
    for c in chunks:
        c.setdefault("acl", acl.get(c["doc_id"]))
    if embed == "m3":
        emb = HttpEmbedder(v["embed_url"], v.get("embed_model", "bge-m3"))
        name = emb.name
    else:
        emb = E5Embedder(v["model_dir"])
        name = Path(v["model_dir"]).name
    # 모델마다 캐시 파일을 따로 둔다 (벡터 차원·값이 다르다)
    cache = Path(cfg["paths"]["data_dir"]) / "vectors" / f"{name}.npz"
    return HybridEngine(base, VectorIndex(chunks, emb, cache), v.get("pool", 20), v.get("rrf_k", 60))
