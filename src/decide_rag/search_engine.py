# -*- coding: utf-8 -*-
"""검색 엔진 (설계서 12.1·12.2).

두 가지 구현이 같은 메서드를 가진다.
  OpenSearchEngine : 운영용. 청크 색인 + 문서 색인(라우팅 후보), 권한 필터, 증분 갱신, _msearch.
  MemoryEngine     : 작은 문서 묶음·시험용. rank_bm25 를 메모리에 올린다 (1단계와 같은 점수).

공통 메서드
  search_many(reqs)            reqs = [(질의, k, 문서범위|None, 제외 chunk_id 집합|None), …] → [[(청크, 점수)], …]
  search(q, k, doc_ids, exclude)
  search_docs(q, k)            문서 색인 BM25 → [문서 항목]
  docs_by_id(ids)              → [문서 항목]
  get_chunks(ids)              → [청크]
  prf_terms(chunk_ids, q, n)   의사 적합성 피드백 확장어
  tokenize(text)               Kiwi 토큰

acl(권한 그룹 목록)은 모든 검색 메서드가 받는다. None 이면 필터 없음. 사용자 그룹 연결은 2.5단계.
"""
import json
import math
import os
from collections import Counter, OrderedDict

from .bm25_index import BM25Index, ko_tokenize


#------------------------------------------------------------------
# 문서 색인 항목 만들기 (라우팅용)
#=> 설계서 12.1: 문서마다 "제목 + 헤딩 경로 목록 + 첫 청크 앞부분" 을 색인 글로 쓴다.
#   decide 선택지 설명(desc)은 1단계 doc_summaries 와 같은 규칙(헤딩 목록 앞 120자)이다.
#    1) 헤딩은 중복·숫자만 있는 것을 뺀다
#    2) text = 제목 + 헤딩 줄들 + 첫 청크 본문 앞 first_chars 자
#
# -in: chunks = 한 문서의 청크 목록 (순서대로)
# -in: first_chars = 첫 청크 본문에서 쓸 글자 수 (기본 300)
# -in: desc_len = decide 선택지 설명 길이 (기본 120)
#
# -out: entry = {"doc_id","title","desc","text"} (청크가 없으면 None)
# -out: error = 없음
#------------------------------------------------------------------
def doc_entry(chunks: list[dict], first_chars: int = 300, desc_len: int = 120) -> dict | None:
    if not chunks:
        return None
    heads: list[str] = []
    for c in chunks:
        for h in c["path"]:
            # 숫자로만 된 헤딩·중복은 설명에 도움이 안 된다
            if h not in heads and not h.replace(".", "").strip().isdigit():
                heads.append(h)
    first = chunks[0].get("body") or chunks[0]["text"]
    title = chunks[0]["title"]
    return {"doc_id": chunks[0]["doc_id"], "title": title, "desc": ", ".join(heads)[:desc_len],
            "text": title + "\n" + "\n".join(heads) + "\n" + first[:first_chars]}


#------------------------------------------------------------------
# 청크 목록 → 문서별 묶음
#=> 청크를 doc_id 별로 나눈다 (청크 순서 유지).
#
# -in: chunks = 청크 목록
#
# -out: groups = OrderedDict{doc_id: [청크, …]}
# -out: error = 없음
#------------------------------------------------------------------
def group_by_doc(chunks: list[dict]) -> "OrderedDict[str, list[dict]]":
    g: "OrderedDict[str, list[dict]]" = OrderedDict()
    for c in chunks:
        g.setdefault(c["doc_id"], []).append(c)
    return g


#------------------------------------------------------------------
# 메모리 검색 엔진
#=> 1단계와 같은 rank_bm25 점수로 청크를 찾고, 문서 색인도 같은 방식으로 하나 더 만든다.
#   문서가 수천 개를 넘으면 느리므로(12.2) 시험·소규모용이다.
#
# -필드: chunks = 청크 색인 (BM25Index)
# -필드: docs = 문서 색인 (BM25Index, 항목의 chunk_id = doc_id)
# -필드: doc_by_id = doc_id → 문서 항목
# -필드: chunk_by_id = chunk_id → 청크
#------------------------------------------------------------------
class MemoryEngine:
    name = "memory"

    #------------------------------------------------------------------
    # 엔진 만들기
    #=> 청크 목록으로 청크 색인과 문서 색인을 만든다. Kiwi 하나를 두 색인이 나눠 쓴다.
    #
    # -in: chunks = 전체 청크 목록 (청크마다 "acl" 이 있으면 권한 필터에 쓴다)
    # -in: first_chars = 문서 색인 글에 넣을 첫 청크 글자 수
    #
    # -out: 없음
    # -out: error = 청크가 비면 rank_bm25 가 ZeroDivisionError
    #------------------------------------------------------------------
    def __init__(self, chunks: list[dict], first_chars: int = 300):
        self.chunks = BM25Index(chunks)
        entries = []
        for cs in group_by_doc(chunks).values():
            e = doc_entry(cs, first_chars)
            if e:
                # 문서 항목의 권한은 그 문서 청크의 권한과 같다
                entries.append({**e, "acl": cs[0].get("acl") or ["all"]})
        self.doc_by_id = {e["doc_id"]: e for e in entries}
        # 문서 색인은 BM25Index 를 그대로 쓰려고 chunk_id 자리에 doc_id 를 넣는다
        self.docs = BM25Index([{**e, "chunk_id": e["doc_id"]} for e in entries], kiwi=self.chunks.kiwi)
        self.chunk_by_id = {c["chunk_id"]: c for c in chunks}

    #------------------------------------------------------------------
    # 문장 → 검색 토큰
    #=> 청크 색인과 같은 Kiwi 규칙.
    #
    # -in: text = 문장
    #
    # -out: tokens = 토큰 목록
    # -out: error = 없음
    #------------------------------------------------------------------
    def tokenize(self, text: str) -> list[str]:
        return self.chunks.tokenize(text)

    #------------------------------------------------------------------
    # 청크 검색 1회
    #=> BM25Index.search 에 권한 필터만 더한다 (청크에 acl 이 없으면 통과).
    #
    # -in: query = 질의
    # -in: k = 개수
    # -in: doc_ids = 허용 문서 ID 집합 (None = 전체)
    # -in: exclude = 제외할 chunk_id 집합
    # -in: acl = 사용자 권한 그룹 목록 (None = 필터 없음)
    #
    # -out: hits = [(청크, 점수), …]
    # -out: error = 없음
    #------------------------------------------------------------------
    def search(self, query: str, k: int = 10, doc_ids=None, exclude=None, acl=None) -> list[tuple[dict, float]]:
        if not acl:
            return self.chunks.search(query, k, doc_ids, exclude)
        allowed = set(acl)
        # 권한으로 걸러질 몫을 고려해 넉넉히 받은 뒤 자른다
        hits = self.chunks.search(query, len(self.chunk_by_id), doc_ids, exclude)
        return [h for h in hits if allowed & set(h[0].get("acl") or ["all"])][:k]

    #------------------------------------------------------------------
    # 청크 검색 여러 번
    #=> OpenSearch 의 _msearch 와 같은 모양으로 맞춘 것. 메모리에서는 차례로 부른다.
    #
    # -in: reqs = [(질의, k, 문서범위, 제외집합), …]
    # -in: acl = 권한 그룹 목록
    #
    # -out: results = 요청마다 hits 목록
    # -out: error = 없음
    #------------------------------------------------------------------
    def search_many(self, reqs: list[tuple], acl=None) -> list[list[tuple[dict, float]]]:
        return [self.search(q, k, d, ex, acl) for q, k, d, ex in reqs]

    #------------------------------------------------------------------
    # 문서 검색 (라우팅 후보)
    #=> 문서 색인 BM25 상위 k 개 문서 항목.
    #
    # -in: query = 질의
    # -in: k = 개수
    # -in: acl = 권한 그룹 목록 (None = 필터 없음)
    #
    # -out: entries = [{"doc_id","title","desc","text"}, …] 점수순
    # -out: error = 없음
    #------------------------------------------------------------------
    def search_docs(self, query: str, k: int = 20, acl=None) -> list[dict]:
        hits = [self.doc_by_id[h[0]["doc_id"]] for h in self.docs.search(query, len(self.doc_by_id))]
        if acl:
            allowed = set(acl)
            # 볼 수 없는 문서는 제목도 decide 에 보내지 않는다 (설계서 12.4)
            hits = [d for d in hits if allowed & set(d.get("acl") or ["all"])]
        return hits[:k]

    #------------------------------------------------------------------
    # 문서 항목 가져오기
    #=> doc_id 목록 순서대로 항목을 돌려준다 (없는 ID 는 뺀다).
    #
    # -in: ids = doc_id 목록
    #
    # -out: entries = 문서 항목 목록
    # -out: error = 없음
    #------------------------------------------------------------------
    def docs_by_id(self, ids: list[str]) -> list[dict]:
        return [self.doc_by_id[i] for i in ids if i in self.doc_by_id]

    #------------------------------------------------------------------
    # 청크 가져오기
    #=> chunk_id 목록 순서대로 청크를 돌려준다 (없는 ID 는 뺀다).
    #
    # -in: ids = chunk_id 목록
    #
    # -out: chunks = 청크 목록
    # -out: error = 없음
    #------------------------------------------------------------------
    def get_chunks(self, ids: list[str]) -> list[dict]:
        return [self.chunk_by_id[i] for i in ids if i in self.chunk_by_id]

    #------------------------------------------------------------------
    # PRF 확장어
    #=> BM25Index.prf_terms 그대로.
    #
    # -in: chunk_ids = 피드백 청크 ID
    # -in: query = 원 질문
    # -in: n = 단어 수
    #
    # -out: terms = 확장 단어 목록
    # -out: error = 없음
    #------------------------------------------------------------------
    def prf_terms(self, chunk_ids: list[str], query: str, n: int = 5) -> list[str]:
        return self.chunks.prf_terms(chunk_ids, query, n)


#------------------------------------------------------------------
# OpenSearch 검색 엔진
#=> 청크 색인(<prefix>_chunks)과 문서 색인(<prefix>_docs)을 쓴다.
#   토큰화는 이쪽(Kiwi)에서 하고, OpenSearch 에는 공백으로 이은 토큰을 넣는다 (whitespace 분석기, BM25 k1=1.5 b=0.75).
#
# -필드: url = OpenSearch 주소
# -필드: chunks_index / docs_index = 색인 이름
# -필드: s = HTTP 세션 (인증은 환경변수 OPENSEARCH_USER / OPENSEARCH_PASSWORD 가 있을 때만)
# -필드: kiwi_index = 토큰화용 BM25Index 규칙 (Kiwi)
#------------------------------------------------------------------
class OpenSearchEngine:
    name = "opensearch"

    #------------------------------------------------------------------
    # 엔진 만들기
    #=> 설정에서 주소·색인 이름을 읽고 HTTP 세션을 연다. 색인은 만들지 않는다 (ensure_indices).
    #
    # -in: sc = cfg["search"] (url, index_prefix, timeout_sec, verify_tls)
    #
    # -out: 없음
    # -out: error = 없음 (연결 오류는 첫 요청에서 난다)
    #------------------------------------------------------------------
    def __init__(self, sc: dict):
        import requests
        from kiwipiepy import Kiwi
        self.url = sc["url"].rstrip("/")
        prefix = sc.get("index_prefix", "decide")
        self.chunks_index, self.docs_index = f"{prefix}_chunks", f"{prefix}_docs"
        self.timeout = sc.get("timeout_sec", 10)
        # 벡터 차원 (7단계). None 이면 벡터 필드를 만들지 않는다 (BM25 만)
        self.vector_dim = sc.get("vector_dim")
        self.recreated = False
        self.s = requests.Session()
        self.s.verify = sc.get("verify_tls", True)
        user, pw = os.environ.get("OPENSEARCH_USER"), os.environ.get("OPENSEARCH_PASSWORD")
        if user and pw:
            self.s.auth = (user, pw)
        self._kiwi = Kiwi()
        # Kiwi 는 첫 분석 때 약 1.1초 준비 시간이 든다 → 첫 사용자 질문이 기다리지 않게 시작할 때 미리 한 번 돌린다
        ko_tokenize(self._kiwi, "준비")

    #------------------------------------------------------------------
    # 문장 → 검색 토큰
    #=> 메모리 엔진과 같은 Kiwi 규칙.
    #
    # -in: text = 문장
    #
    # -out: tokens = 토큰 목록
    # -out: error = 없음
    #------------------------------------------------------------------
    def tokenize(self, text: str) -> list[str]:
        return ko_tokenize(self._kiwi, text)

    #------------------------------------------------------------------
    # HTTP 요청 보내기
    #=> 응답 코드가 300 이상이면 본문 앞부분을 붙여 RuntimeError 를 낸다.
    #
    # -in: method = "GET"/"POST"/"PUT"/"DELETE"
    # -in: path = "/색인/_search" 같은 경로
    # -in: body = JSON 본문 (dict) 또는 None
    # -in: ndjson = _bulk·_msearch 용 줄 단위 본문 (str)
    # -in: ok404 = True 면 404 를 오류로 보지 않음
    #
    # -out: data = 응답 JSON (dict)
    # -out: error = 실패 시 RuntimeError
    #------------------------------------------------------------------
    def _req(self, method: str, path: str, body=None, ndjson: str | None = None, ok404: bool = False) -> dict:
        kw = {"timeout": self.timeout}
        if ndjson is not None:
            kw["data"] = ndjson.encode("utf-8")
            kw["headers"] = {"Content-Type": "application/x-ndjson"}
        elif body is not None:
            kw["json"] = body
        r = self.s.request(method, self.url + path, **kw)
        if r.status_code == 404 and ok404:
            return {}
        if r.status_code >= 300:
            raise RuntimeError(f"OpenSearch {method} {path} {r.status_code}: {r.text[:500]}")
        return r.json() if r.content else {}

    #------------------------------------------------------------------
    # 색인 정의
    #=> 청크 색인과 문서 색인의 설정·매핑. tokens 만 검색하고 나머지는 저장만 한다.
    #
    # -in: kind = "chunks" 또는 "docs"
    # -in: dim = 청크 벡터 차원 (None 이면 벡터 필드 없음, 7단계)
    #
    # -out: body = 색인 생성 요청 본문
    # -out: error = 없음
    #------------------------------------------------------------------
    @staticmethod
    def index_body(kind: str, dim: int | None = None) -> dict:
        props = {"doc_id": {"type": "keyword"}, "acl": {"type": "keyword"},
                 "tokens": {"type": "text", "analyzer": "whitespace", "similarity": "bm25p"},
                 "title": {"type": "keyword", "index": False}, "text": {"type": "text", "index": False}}
        if kind == "chunks":
            props.update({"chunk_id": {"type": "keyword"}, "path": {"type": "keyword", "index": False},
                          "page": {"type": "integer", "index": False}, "body": {"type": "text", "index": False}})
        else:
            props.update({"desc": {"type": "text", "index": False}})
        settings = {"number_of_shards": 1, "number_of_replicas": 0,
                    "similarity": {"bm25p": {"type": "BM25", "k1": 1.5, "b": 0.75}}}
        if kind == "chunks" and dim:
            # 청크 벡터 (7단계, 설계서 12.2.2): lucene HNSW, 코사인. 권한 필터를 k-NN 안에서 건다
            settings["knn"] = True
            props["vec"] = {"type": "knn_vector", "dimension": dim,
                            "method": {"name": "hnsw", "engine": "lucene", "space_type": "cosinesimil"}}
        return {"settings": settings, "mappings": {"properties": props}}

    #------------------------------------------------------------------
    # 색인 준비
    #=> 두 색인이 없으면 만든다. recreate=True 면 지우고 새로 만든다 (전체 재색인).
    #   벡터를 쓰는데 기존 청크 색인에 맞는 벡터 필드가 없으면 스스로 다시 만들고 self.recreated 를 True 로 둔다.
    #
    # -in: recreate = 기존 색인 삭제 여부
    #
    # -out: 없음
    # -out: error = OpenSearch 오류 시 RuntimeError
    #------------------------------------------------------------------
    def ensure_indices(self, recreate: bool = False):
        # 벡터를 쓰는데 청크 색인에 벡터 필드가 없거나 차원이 다르면 두 색인을 다시 만든다
        #   (매핑은 바꿀 수 없다. 다시 만든 뒤 indexer 가 저장된 청크 파일로 다시 채운다)
        if not recreate and self.vector_dim and self._exists(self.chunks_index):
            m = self._req("GET", f"/{self.chunks_index}/_mapping")
            vec = next(iter(m.values()))["mappings"].get("properties", {}).get("vec", {})
            if vec.get("dimension") != self.vector_dim:
                recreate = True
        self.recreated = recreate
        for kind, name in (("chunks", self.chunks_index), ("docs", self.docs_index)):
            if recreate:
                self._req("DELETE", f"/{name}", ok404=True)
            if not self._exists(name):
                self._req("PUT", f"/{name}", self.index_body(kind, self.vector_dim))

    #------------------------------------------------------------------
    # 색인 존재 확인
    #=> HEAD 요청은 본문이 없으므로 상태 코드로 판단한다.
    #
    # -in: name = 색인 이름
    #
    # -out: exists = True/False
    # -out: error = 연결 오류는 예외 전파
    #------------------------------------------------------------------
    def _exists(self, name: str) -> bool:
        return self.s.head(f"{self.url}/{name}", timeout=self.timeout).status_code == 200

    #------------------------------------------------------------------
    # 문서 하나 교체 (증분 색인)
    #=> 그 문서의 옛 청크를 지우고 새 청크와 문서 항목을 넣는다.
    #    1) 청크 색인에서 doc_id 로 delete_by_query
    #    2) 새 청크를 _bulk 로 넣음 (_id = chunk_id)
    #    3) 문서 색인에 항목을 넣음 (_id = doc_id), 청크가 없으면 문서 항목도 지움
    #
    # -in: chunks = 이 문서의 새 청크 목록 (비어 있을 수 있음)
    # -in: doc_id = 문서 ID
    # -in: acl = 이 문서를 볼 수 있는 그룹 목록
    # -in: first_chars = 문서 색인 글에 넣을 첫 청크 글자 수
    # -in: vectors = 청크 순서대로 벡터 (None 이면 벡터 없이, 7단계)
    #
    # -out: n = 넣은 청크 수
    # -out: error = OpenSearch 오류 시 RuntimeError
    #------------------------------------------------------------------
    def replace_doc(self, doc_id: str, chunks: list[dict], acl: list[str], first_chars: int = 300,
                    vectors=None) -> int:
        self.delete_doc(doc_id)
        lines = []
        for i, c in enumerate(chunks):
            lines.append(json.dumps({"index": {"_index": self.chunks_index, "_id": c["chunk_id"]}}))
            src = {**c, "acl": acl, "tokens": " ".join(self.tokenize(c["text"]))}
            if vectors is not None:
                # 소수 6자리로 줄여 보낸다 (bulk 크기를 줄이려고, 코사인에는 영향이 거의 없다)
                src["vec"] = [round(float(x), 6) for x in vectors[i]]
            lines.append(json.dumps(src, ensure_ascii=False))
        e = doc_entry(chunks, first_chars)
        if e:
            lines.append(json.dumps({"index": {"_index": self.docs_index, "_id": doc_id}}))
            lines.append(json.dumps({**e, "acl": acl, "tokens": " ".join(self.tokenize(e["text"]))},
                                    ensure_ascii=False))
        if lines:
            r = self._req("POST", "/_bulk", ndjson="\n".join(lines) + "\n")
            if r.get("errors"):
                bad = [it for it in r["items"] if list(it.values())[0].get("error")][:2]
                raise RuntimeError(f"bulk 오류: {json.dumps(bad, ensure_ascii=False)[:500]}")
        return len(chunks)

    #------------------------------------------------------------------
    # 문서 하나 삭제
    #=> 청크 색인과 문서 색인에서 그 문서를 지운다.
    #
    # -in: doc_id = 문서 ID
    #
    # -out: 없음
    # -out: error = OpenSearch 오류 시 RuntimeError
    #------------------------------------------------------------------
    def delete_doc(self, doc_id: str):
        self._req("POST", f"/{self.chunks_index}/_delete_by_query?refresh=true&conflicts=proceed",
                  {"query": {"term": {"doc_id": doc_id}}})
        self._req("DELETE", f"/{self.docs_index}/_doc/{doc_id}?refresh=true", ok404=True)

    #------------------------------------------------------------------
    # 문서 하나의 권한만 바꾸기
    #=> 내용은 그대로 두고 그 문서의 청크·문서 항목의 acl 필드만 갱신한다 (파싱·토큰화 없음).
    #
    # -in: doc_id = 문서 ID
    # -in: acl = 새 그룹 목록
    #
    # -out: 없음
    # -out: error = OpenSearch 오류 시 RuntimeError
    #------------------------------------------------------------------
    def update_acl(self, doc_id: str, acl: list[str]):
        script = {"source": "ctx._source.acl = params.acl", "lang": "painless", "params": {"acl": list(acl)}}
        self._req("POST", f"/{self.chunks_index}/_update_by_query?refresh=true&conflicts=proceed",
                  {"query": {"term": {"doc_id": doc_id}}, "script": script})
        self._req("POST", f"/{self.docs_index}/_update/{doc_id}?refresh=true", {"doc": {"acl": list(acl)}},
                  ok404=True)

    #------------------------------------------------------------------
    # 새로 넣은 내용 검색에 반영
    #=> 색인 뒤 refresh 를 해야 바로 검색된다.
    #
    # -in: 없음
    #
    # -out: 없음
    # -out: error = OpenSearch 오류 시 RuntimeError
    #------------------------------------------------------------------
    def refresh(self):
        self._req("POST", f"/{self.chunks_index},{self.docs_index}/_refresh")

    #------------------------------------------------------------------
    # 색인된 문서·청크 수
    #=> 증분 색인이 엔진과 어긋났는지(예: 엔진을 새로 깔았을 때) 확인하는 데 쓴다.
    #
    # -in: 없음
    #
    # -out: (문서 수, 청크 수)
    # -out: error = 색인이 없으면 (0, 0)
    #------------------------------------------------------------------
    def counts(self) -> tuple[int, int]:
        out = []
        for name in (self.docs_index, self.chunks_index):
            out.append(self._req("GET", f"/{name}/_count", ok404=True).get("count", 0))
        return out[0], out[1]

    #------------------------------------------------------------------
    # 검색 질의 본문 만들기
    #=> tokens 필드 match + (문서범위·권한) filter + (제외 청크) must_not.
    #
    # -in: query = 질의 문장
    # -in: k = 개수
    # -in: doc_ids = 허용 문서 ID 집합 또는 None
    # -in: exclude = 제외 chunk_id 집합 또는 None
    # -in: acl = 권한 그룹 목록 또는 None
    #
    # -out: body = _search 본문 (토큰이 없으면 None)
    # -out: error = 없음
    #------------------------------------------------------------------
    def _body(self, query: str, k: int, doc_ids=None, exclude=None, acl=None) -> dict | None:
        toks = self.tokenize(query)
        if not toks:
            return None
        b: dict = {"must": [{"match": {"tokens": " ".join(toks)}}]}
        flt = []
        if doc_ids:
            flt.append({"terms": {"doc_id": sorted(doc_ids)}})
        if acl:
            # 권한 없는 문서는 점수 계산 전에 걸러진다 (설계서 12.4)
            flt.append({"terms": {"acl": list(acl)}})
        if flt:
            b["filter"] = flt
        if exclude:
            b["must_not"] = [{"ids": {"values": sorted(exclude)}}]
        return {"size": k, "query": {"bool": b}, "_source": {"excludes": ["tokens", "acl", "vec"]}}

    #------------------------------------------------------------------
    # 벡터 검색 여러 번 (k-NN, _msearch 한 번)
    #=> 질의 벡터마다 가까운 청크 k 개. 문서 범위·권한·제외 청크는 k-NN 안의 filter 로 건다
    #   (lucene 엔진은 거른 뒤에 k 개를 채운다 → 권한 없는 청크 때문에 결과가 비지 않는다).
    #
    # -in: reqs = [(질의 벡터, k, 문서범위, 제외집합), …]
    # -in: acl = 권한 그룹 목록 (None = 필터 없음)
    #
    # -out: results = 요청마다 [(청크, 점수), …]
    # -out: error = 벡터 필드가 없으면 RuntimeError, OpenSearch 오류 시 RuntimeError
    #------------------------------------------------------------------
    def knn_many(self, reqs: list[tuple], acl=None) -> list[list[tuple[dict, float]]]:
        if not self.vector_dim:
            raise RuntimeError("벡터를 쓰지 않는 엔진입니다 ([vector] enabled)")
        if not reqs:
            return []
        nd = []
        for vec, k, doc_ids, exclude in reqs:
            flt, must_not = [], []
            if doc_ids:
                flt.append({"terms": {"doc_id": sorted(doc_ids)}})
            if acl:
                flt.append({"terms": {"acl": list(acl)}})
            if exclude:
                must_not.append({"ids": {"values": sorted(exclude)}})
            knn = {"vector": [float(x) for x in vec], "k": k}
            if flt or must_not:
                knn["filter"] = {"bool": {"filter": flt, "must_not": must_not}}
            nd.append(json.dumps({"index": self.chunks_index}))
            nd.append(json.dumps({"size": k, "query": {"knn": {"vec": knn}},
                                  "_source": {"excludes": ["tokens", "acl", "vec"]}}))
        resp = self._req("POST", "/_msearch", ndjson="\n".join(nd) + "\n")["responses"]
        out = []
        for r in resp:
            if "error" in r:
                raise RuntimeError(f"knn 오류: {json.dumps(r['error'], ensure_ascii=False)[:300]}")
            out.append([(h["_source"], h["_score"]) for h in r["hits"]["hits"]])
        return out

    #------------------------------------------------------------------
    # 청크 검색 여러 번 (_msearch 한 번)
    #=> 재검색 질의 여러 개를 요청 한 번으로 보낸다 (12.2.1: 따로 보내는 것의 절반 시간).
    #
    # -in: reqs = [(질의, k, 문서범위, 제외집합), …]
    # -in: acl = 권한 그룹 목록
    #
    # -out: results = 요청마다 [(청크, 점수), …]
    # -out: error = OpenSearch 오류 시 RuntimeError
    #------------------------------------------------------------------
    def search_many(self, reqs: list[tuple], acl=None) -> list[list[tuple[dict, float]]]:
        bodies = [self._body(q, k, d, ex, acl) for q, k, d, ex in reqs]
        live = [b for b in bodies if b]
        if not live:
            return [[] for _ in reqs]
        nd = "".join(json.dumps({"index": self.chunks_index}) + "\n" + json.dumps(b, ensure_ascii=False) + "\n"
                     for b in live)
        resp = iter(self._req("POST", "/_msearch", ndjson=nd)["responses"])
        out = []
        for b in bodies:
            if not b:
                out.append([])
                continue
            r = next(resp)
            if "error" in r:
                raise RuntimeError(f"msearch 오류: {json.dumps(r['error'], ensure_ascii=False)[:300]}")
            out.append([(h["_source"], h["_score"]) for h in r["hits"]["hits"]])
        return out

    #------------------------------------------------------------------
    # 청크 검색 1회
    #=> search_many 에 요청 하나를 넣는다.
    #
    # -in: query, k, doc_ids, exclude, acl = search_many 와 같음
    #
    # -out: hits = [(청크, 점수), …]
    # -out: error = OpenSearch 오류 시 RuntimeError
    #------------------------------------------------------------------
    def search(self, query: str, k: int = 10, doc_ids=None, exclude=None, acl=None) -> list[tuple[dict, float]]:
        return self.search_many([(query, k, doc_ids, exclude)], acl)[0]

    #------------------------------------------------------------------
    # 문서 검색 (라우팅 후보)
    #=> 문서 색인에서 BM25 상위 k 개 문서 항목. 권한 필터를 먼저 건다.
    #
    # -in: query = 질의
    # -in: k = 개수
    # -in: acl = 권한 그룹 목록
    #
    # -out: entries = [{"doc_id","title","desc","text"}, …] 점수순
    # -out: error = OpenSearch 오류 시 RuntimeError
    #------------------------------------------------------------------
    def search_docs(self, query: str, k: int = 20, acl=None) -> list[dict]:
        b = self._body(query, k, acl=acl)
        if not b:
            return []
        return [h["_source"] for h in self._req("POST", f"/{self.docs_index}/_search", b)["hits"]["hits"]]

    #------------------------------------------------------------------
    # ID 로 가져오기 (_mget)
    #=> 색인에서 _id 목록을 한 번에 가져와 요청 순서대로 돌려준다 (없는 ID 는 뺀다).
    #
    # -in: index = 색인 이름
    # -in: ids = _id 목록
    #
    # -out: sources = _source 목록
    # -out: error = OpenSearch 오류 시 RuntimeError
    #------------------------------------------------------------------
    def _mget(self, index: str, ids: list[str]) -> list[dict]:
        if not ids:
            return []
        r = self._req("POST", f"/{index}/_mget?_source_excludes=tokens,acl,vec", {"ids": list(ids)})
        return [d["_source"] for d in r["docs"] if d.get("found")]

    #------------------------------------------------------------------
    # 문서 항목 가져오기
    #=> 문서 색인에서 doc_id 목록으로.
    #
    # -in: ids = doc_id 목록
    #
    # -out: entries = 문서 항목 목록
    # -out: error = OpenSearch 오류 시 RuntimeError
    #------------------------------------------------------------------
    def docs_by_id(self, ids: list[str]) -> list[dict]:
        return self._mget(self.docs_index, ids)

    #------------------------------------------------------------------
    # 청크 가져오기
    #=> 청크 색인에서 chunk_id 목록으로.
    #
    # -in: ids = chunk_id 목록
    #
    # -out: chunks = 청크 목록
    # -out: error = OpenSearch 오류 시 RuntimeError
    #------------------------------------------------------------------
    def get_chunks(self, ids: list[str]) -> list[dict]:
        return self._mget(self.chunks_index, ids)

    #------------------------------------------------------------------
    # PRF 확장어
    #=> 메모리 엔진과 같은 식(tf × log(1 + N / (1 + df)))을 OpenSearch 의 단어 통계로 계산한다.
    #    1) _mtermvectors 로 피드백 청크의 단어별 tf 와 df(doc_freq), 전체 청크 수(doc_count)를 받는다
    #    2) 질문에 이미 있는 단어·한 글자·숫자는 뺀다
    #
    # -in: chunk_ids = 피드백 청크 ID
    # -in: query = 원 질문
    # -in: n = 단어 수
    #
    # -out: terms = 확장 단어 목록
    # -out: error = OpenSearch 오류 시 RuntimeError
    #------------------------------------------------------------------
    def prf_terms(self, chunk_ids: list[str], query: str, n: int = 5) -> list[str]:
        if not chunk_ids:
            return []
        r = self._req("POST", f"/{self.chunks_index}/_mtermvectors",
                      {"ids": list(chunk_ids), "parameters": {"fields": ["tokens"], "term_statistics": True,
                                                               "field_statistics": True, "positions": False,
                                                               "offsets": False}})
        qset = set(self.tokenize(query))
        tf, df, N = Counter(), {}, 0
        for d in r.get("docs", []):
            tv = d.get("term_vectors", {}).get("tokens")
            if not tv:
                continue
            N = max(N, tv["field_statistics"]["doc_count"])
            for t, st in tv["terms"].items():
                tf[t] += st["term_freq"]
                df[t] = st.get("doc_freq", 0)
        scored = [(f * math.log(1 + N / (1 + df[t])), t) for t, f in tf.items()
                  if t not in qset and len(t) >= 2 and not t.isdigit()]
        return [t for _, t in sorted(scored, reverse=True)[:n]]


#------------------------------------------------------------------
# 설정에 맞는 엔진 만들기
#=> cfg["search"]["engine"] 이 "opensearch" 면 OpenSearchEngine, "memory" 면 저장된 청크로 MemoryEngine.
#   메모리 엔진의 청크 권한은 manifest 의 문서 acl 을 붙인다.
#
# -in: cfg = 설정 dict
#
# -out: engine = 검색 엔진
# -out: error = 알 수 없는 엔진 이름이면 ValueError
#------------------------------------------------------------------
def make_engine(cfg: dict):
    sc = cfg.get("search", {"engine": "memory"})
    v = cfg.get("vector", {})
    if sc["engine"] == "opensearch":
        # 벡터를 켜면 청크 색인에 벡터 필드를 둔다 (7단계)
        return OpenSearchEngine({**sc, "vector_dim": v.get("dim") if v.get("enabled") else None})
    if sc["engine"] == "memory":
        from .chunker import load_chunks
        from .indexer import load_manifest
        acl = {d: v.get("acl") or ["all"] for d, v in load_manifest(cfg).items()}
        chunks = [{**c, "acl": acl.get(c["doc_id"], ["all"])} for c in load_chunks(cfg)]
        return MemoryEngine(chunks, cfg.get("index", {}).get("doc_first_chars", 300))
    raise ValueError(f"알 수 없는 검색 엔진: {sc['engine']}")
