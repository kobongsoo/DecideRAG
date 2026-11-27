"""어휘 불일치("축의금" ↔ 문서엔 "경조") 보완 방안 측정.

  A    : 문서 BM25 (제목+헤딩+첫청크300자)
  A+X  : A 색인 글에 Qwen 이 만든 검색어를 덧붙인 BM25 (LLM 문서 확장)
  S    : neural sparse (opensearch-neural-sparse-encoding-multilingual-v1, 문서 쪽만 모델)
  A+S, A+X+S : RRF 혼합
  B    : e5-small-ko (참고)

사용:
  python tools\\vocab_gap_bench.py expand      # Qwen 문서 확장 → data\\doc_expand.jsonl (문서당 시간 기록)
  python tools\\vocab_gap_bench.py recall      # 기존 90문항 + 어휘 불일치 75문항 재현율
  python tools\\vocab_gap_bench.py speed       # sparse 문서 인코딩 처리량, 질문 인코딩 시간
  python tools\\vocab_gap_bench.py os          # OpenSearch 문서 10만 개: BM25 + sparse 역색인 적재·검색 속도
"""
import json
import random
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from decide_rag.bm25_index import BM25Index  # noqa: E402
from doc_route_bench import KS, best_rank, doc_texts, load_queries, rrf  # noqa: E402

SPARSE_DIR = r"D:\Tools\models\opensearch-neural-sparse-multilingual-v1"
EXPAND_FILE = ROOT / "data" / "doc_expand.jsonl"
MISMATCH_FILE = ROOT / "eval" / "evalset_mismatch.jsonl"
OS = "http://127.0.0.1:9200"
PRUNE = 0.1   # 모델 카드 권장: 문서 최대 가중치의 10% 미만 토큰은 버린다
EXPAND_PROMPT = """다음은 사내 문서의 제목·목차·앞부분이다.
직원이 이 문서를 찾으려고 검색창에 칠 법한 검색어를 40개 만들어라.
- 문서에 이미 적힌 단어보다, 같은 뜻의 다른 말·구어체·줄임말·상위/하위 개념을 우선한다.
- 검색어만 쉼표로 구분해 한 줄로 출력한다. 설명은 쓰지 않는다.

[문서]
{doc}"""


#------------------------------------------------------------------
# 청크 → 문서 목록 (확장용 원문 포함)
#=> 문서 색인 글(제목+헤딩+첫청크300자)과, LLM 에 줄 좀 더 긴 원문(제목+헤딩+본문 앞 1500자)을 만든다.
#
# -in: 없음
#
# -out: docs = [{"doc_id","title","text","source"}, …]
# -out: error = 데이터 파일이 없으면 예외 전파
#------------------------------------------------------------------
def load_docs() -> list[dict]:
    chunks = [json.loads(l) for l in open(ROOT / "data" / "legacy_stage1" / "chunks.jsonl", encoding="utf-8")]
    docs = doc_texts(chunks, True)
    body: dict[str, str] = {}
    for c in chunks:
        # 문서별 본문을 순서대로 이어 붙여 앞 1500자만 쓴다
        if len(body.get(c["doc_id"], "")) < 1500:
            body[c["doc_id"]] = body.get(c["doc_id"], "") + (c.get("body") or c["text"]) + "\n"
    for d in docs:
        heads = d["text"].split("\n")[1:-1]
        d["source"] = d["title"] + "\n" + "\n".join(heads)[:800] + "\n" + body[d["doc_id"]][:1500]
    return docs


#------------------------------------------------------------------
# Qwen 문서 확장 생성
#=> 문서마다 Qwen 에게 "사용자가 칠 법한 다른 표현" 40개를 받아 저장하고, 문서당 걸린 시간을 잰다.
#    이미 만든 문서는 건너뛴다 (끊겨도 이어서).
#
# -in: 없음
#
# -out: 없음 (data\doc_expand.jsonl 에 {"doc_id","terms","sec","usage"} 저장, 요약 출력)
# -out: error = LLM 서버 오류는 clients 의 재시도 후 예외 전파
#------------------------------------------------------------------
def expand():
    from decide_rag.clients import LLMClient
    from decide_rag.config import load_config
    llm = LLMClient(load_config()["llm"])
    done = {}
    if EXPAND_FILE.exists():
        done = {json.loads(l)["doc_id"]: json.loads(l) for l in open(EXPAND_FILE, encoding="utf-8")}
    with open(EXPAND_FILE, "a", encoding="utf-8") as f:
        for d in load_docs():
            if d["doc_id"] in done:
                continue
            t = time.perf_counter()
            text, usage = llm.chat([{"role": "user", "content": EXPAND_PROMPT.format(doc=d["source"])}],
                                   max_tokens=400, temperature=0.0)
            sec = time.perf_counter() - t
            terms = [w.strip() for w in text.replace("\n", ",").split(",") if w.strip()]
            row = {"doc_id": d["doc_id"], "terms": terms, "sec": round(sec, 2), "usage": usage}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            done[d["doc_id"]] = row
            print(d["doc_id"], d["title"][:20], f"{sec:.1f}s", len(terms), "개:", ", ".join(terms[:8]))
    secs = [r["sec"] for r in done.values()]
    print(f"문서 {len(secs)}개, 문서당 평균 {statistics.mean(secs):.1f}s, 최대 {max(secs):.1f}s "
          f"→ 10만 문서 순차 {100000 * statistics.mean(secs) / 3600:.0f}시간")


#------------------------------------------------------------------
# neural sparse 인코더 묶음
#=> 문서는 모델로, 질문은 토크나이저+가중치표로 희소 벡터({토큰: 가중치})를 만든다.
#
# -필드: m = sentence-transformers SparseEncoder
#------------------------------------------------------------------
class Sparse:
    #------------------------------------------------------------------
    # 모델 불러오기
    #=> 내려받은 폴더에서 CPU 로 읽는다 (원격 코드 실행 없음).
    #
    # -in: 없음
    #
    # -out: 없음
    # -out: error = 파일이 없으면 예외 전파
    #------------------------------------------------------------------
    def __init__(self):
        from sentence_transformers.sparse_encoder import SparseEncoder
        self.m = SparseEncoder(SPARSE_DIR, device="cpu", trust_remote_code=False)

    #------------------------------------------------------------------
    # 희소 텐서 → {토큰: 가중치} 목록
    #=> prune>0 이면 행마다 최대 가중치 × prune 미만은 버린다 (모델 카드 권장 max_ratio 가지치기).
    #
    # -in: emb = encode_* 결과 (희소 텐서)
    # -in: prune = 가지치기 비율 (0 이면 안 함)
    #
    # -out: vecs = [{토큰: 가중치}, …]
    # -out: error = 없음
    #------------------------------------------------------------------
    def to_dicts(self, emb, prune: float = 0.0) -> list[dict]:
        out = []
        for row in self.m.decode(emb, top_k=None):
            d = {w: float(s) for w, s in row}
            if prune and d:
                cut = max(d.values()) * prune
                d = {w: s for w, s in d.items() if s >= cut}
            out.append(d)
        return out

    #------------------------------------------------------------------
    # 문서들 → 희소 벡터
    #=> 신경망을 돌려 문서마다 관련 토큰과 가중치를 만든다 (색인 때 1번).
    #
    # -in: texts = 문서 글 목록
    # -in: prune = 가지치기 비율
    #
    # -out: vecs = [{토큰: 가중치}, …]
    # -out: error = 예외 전파
    #------------------------------------------------------------------
    def docs(self, texts: list[str], prune: float = 0.0) -> list[dict]:
        return self.to_dicts(self.m.encode_document(texts, batch_size=8), prune)

    #------------------------------------------------------------------
    # 질문 → 희소 벡터
    #=> 신경망 없이 토크나이저와 가중치표만 쓴다.
    #
    # -in: q = 질문
    #
    # -out: vec = {토큰: 가중치}
    # -out: error = 없음
    #------------------------------------------------------------------
    def query(self, q: str) -> dict:
        return self.to_dicts(self.m.encode_query([q]))[0]


#------------------------------------------------------------------
# 희소 내적 점수
#=> 질문과 문서에 함께 있는 토큰의 가중치 곱을 더한다 (OpenSearch neural_sparse 와 같은 식).
#
# -in: q = 질문 벡터 {토큰: 가중치}
# -in: d = 문서 벡터 {토큰: 가중치}
#
# -out: score = 실수
# -out: error = 없음
#------------------------------------------------------------------
def sdot(q: dict, d: dict) -> float:
    return sum(w * d[t] for t, w in q.items() if t in d)


#------------------------------------------------------------------
# 어휘 불일치 평가 질의 읽기
#=> evalset_mismatch.jsonl 을 load_queries() 와 같은 모양으로 바꾼다.
#
# -in: 없음
#
# -out: qs = [{"id","type","query","gold"}, …]
# -out: error = 파일이 없으면 예외 전파
#------------------------------------------------------------------
def load_mismatch() -> list[dict]:
    return [{"id": r["id"], "type": r["type"], "query": r["question"], "gold": set(r["doc_ids"])}
            for r in map(json.loads, open(MISMATCH_FILE, encoding="utf-8"))]


#------------------------------------------------------------------
# 재현율 비교
#=> 기존 90문항과 어휘 불일치 75문항에서, 방식별로 정답 문서가 상위 k 에 드는 비율을 잰다.
#    1) A, A+X 는 Kiwi BM25 (A+X 는 색인 글 + Qwen 검색어)
#    2) S 는 가지치기 없음 / 0.1 두 가지
#    3) B(e5) 는 참고용
#
# -in: 없음
#
# -out: 없음 (표 출력, eval\runs\vocab_gap_bench.json 저장)
# -out: error = 확장 파일이 없으면 KeyError
#------------------------------------------------------------------
def recall():
    # doc_route_bench 를 불러올 때 프로토타입 src 가 sys.path 에 들어간다
    from vector_index import OnnxEmbedder, PASSAGE_PREFIX, QUERY_PREFIX
    from doc_route_bench import E5_DIR
    docs = load_docs()
    ids = [d["doc_id"] for d in docs]
    exp = {json.loads(l)["doc_id"]: json.loads(l)["terms"] for l in open(EXPAND_FILE, encoding="utf-8")}
    bm_a = BM25Index(docs)
    bm_x = BM25Index([{**d, "text": d["text"] + "\n" + ", ".join(exp[d["doc_id"]])} for d in docs])
    sp = Sparse()
    dvec = {"S": sp.docs([d["text"] for d in docs]), "S_p0.1": sp.docs([d["text"] for d in docs], PRUNE)}
    dvec["SX"] = sp.docs([d["text"] + "\n" + ", ".join(exp[d["doc_id"]]) for d in docs], PRUNE)
    e5 = OnnxEmbedder(E5_DIR)
    e5d = e5.embed([d["text"] for d in docs], PASSAGE_PREFIX)

    # BM25 순위: 점수 0 문서는 BM25 가 돌려주지 않으므로 "못 찾음" 으로 둔다 (실제 엔진과 같음)
    def bm_rank(bm, q):
        return [h[0]["doc_id"] for h in bm.search(q, k=len(ids))]

    result = {}
    for setname, qs in (("기존 90문항", load_queries()), ("어휘 불일치 75문항", load_mismatch())):
        ranks: dict[str, list[int]] = {}
        for q in qs:
            qv = sp.query(q["query"])
            L = {"A": bm_rank(bm_a, q["query"]), "A+X": bm_rank(bm_x, q["query"])}
            for name in ("S", "S_p0.1", "SX"):
                sc = [sdot(qv, v) for v in dvec[name]]
                L[name] = [ids[i] for i in np.argsort(-np.array(sc), kind="stable")]
            L["B"] = [ids[i] for i in np.argsort(-(e5d @ e5.embed([q["query"]], QUERY_PREFIX)[0]))]
            L["A+S"] = rrf(L["A"], L["S_p0.1"])
            L["A+X+S"] = rrf(L["A+X"], L["S_p0.1"])
            L["A+B"] = rrf(L["A"], L["B"])
            L["A+X+B"] = rrf(L["A+X"], L["B"])
            for m, lst in L.items():
                ranks.setdefault(m, []).append(best_rank(lst, q["gold"]))
        table = {m: {**{f"@{k}": round(sum(r <= k for r in rs) / len(rs), 3) for k in KS},
                     "평균순위": round(statistics.mean(rs), 2)} for m, rs in ranks.items()}
        result[setname] = table
        print(f"\n## {setname} (문서 {len(ids)}개)")
        print("| 방식 | @1 | @3 | @5 | 평균순위 |\n|---|---|---|---|---|")
        for m, t in table.items():
            print(f"| {m} | {t['@1']} | {t['@3']} | {t['@5']} | {t['평균순위']} |")
    # A001 원 사례: "축의금" 한 단어 질의에서 경조사지원규정(D01) 순위
    for q in ("축의금", "축의금 얼마 줘?"):
        qv = sp.query(q)
        r = {"A": bm_rank(bm_a, q), "A+X": bm_rank(bm_x, q),
             "S_p0.1": [ids[i] for i in np.argsort(-np.array([sdot(qv, v) for v in dvec["S_p0.1"]]))],
             "B": [ids[i] for i in np.argsort(-(e5d @ e5.embed([q], QUERY_PREFIX)[0]))]}
        print(f"'{q}' → D01 순위(없음=BM25 점수 0):", {m: (lst.index("D01") + 1 if "D01" in lst else "없음")
                                                     for m, lst in r.items()},
              "| 각 1위:", {m: (lst[0] if lst else "없음") for m, lst in r.items()})
    print("D01 확장어:", ", ".join(exp["D01"]))
    out = ROOT / "eval" / "runs" / "vocab_gap_bench.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print("저장:", out)


#------------------------------------------------------------------
# sparse 인코딩 속도
#=> 문서 글(평균 약 800자)을 반복해 문서 인코딩 처리량을 재고, 질문 인코딩 시간을 잰다.
#
# -in: 없음
#
# -out: 없음 (출력)
# -out: error = 예외 전파
#------------------------------------------------------------------
def speed():
    import torch
    docs = [d["text"] for d in load_docs()]
    sp = Sparse()
    sp.docs(docs[:2])  # 준비 시간 제외
    sample = (docs * 12)[:200]
    t = time.perf_counter()
    vecs = sp.docs(sample, PRUNE)
    dps = len(sample) / (time.perf_counter() - t)
    nz = statistics.mean(len(v) for v in vecs)
    qt = []
    for q in load_queries():
        t = time.perf_counter()
        sp.query(q["query"])
        qt.append((time.perf_counter() - t) * 1000)
    qt.sort()
    print(f"torch 스레드 {torch.get_num_threads()} | 문서 인코딩 {dps:.1f}개/초 → 10만 {100000 / dps / 3600:.1f}시간 | "
          f"가지치기 후 문서당 토큰 {nz:.0f}개 | 질문 1개 평균 {statistics.mean(qt):.2f}ms p90 {qt[int(len(qt) * 0.9)]:.2f}ms")


#------------------------------------------------------------------
# OpenSearch 색인 정의 (문서 단위: BM25 + sparse)
#=> tokens 는 Kiwi 토큰(공백 분석기·BM25), sparse 는 rank_features (토큰→가중치), acl 은 권한 필터용.
#
# -in: 없음
#
# -out: body = 색인 생성 요청 본문
# -out: error = 없음
#------------------------------------------------------------------
def os_index_body() -> dict:
    return {"settings": {"number_of_shards": 1, "number_of_replicas": 0, "refresh_interval": "-1",
                         "similarity": {"bm25p": {"type": "BM25", "k1": 1.5, "b": 0.75}}},
            "mappings": {"properties": {
                "doc_id": {"type": "keyword"}, "acl": {"type": "keyword"},
                "tokens": {"type": "text", "analyzer": "whitespace", "similarity": "bm25p"},
                "sparse": {"type": "rank_features"}}}}


#------------------------------------------------------------------
# OpenSearch 역색인 속도 측정 (문서 10만 개)
#=> 실제 청크 606개 + 문서 18개의 희소 벡터·Kiwi 토큰을 반복해 문서 10만 개를 만들고 적재·검색 시간을 잰다.
#    1) 적재: _bulk 2000건씩, 끝나고 refresh·forcemerge
#    2) 검색: 질문 165개(기존 90 + 불일치 75) × BM25 / sparse / 둘 다(_msearch), 권한 필터 있음·없음
#
# -in: 없음
#
# -out: 없음 (출력)
# -out: error = OpenSearch 응답 오류는 RuntimeError
#------------------------------------------------------------------
def os_bench(n: int = 100000, name: str = "docs_sparse_100k"):
    import requests
    s = requests.Session()
    chunks = [json.loads(l) for l in open(ROOT / "data" / "legacy_stage1" / "chunks.jsonl", encoding="utf-8")]
    docs = load_docs()
    base = [d["text"] for d in docs] + [c["text"][:900] for c in chunks]
    sp = Sparse()
    bm = BM25Index(docs)
    t = time.perf_counter()
    svec = sp.docs(base, PRUNE)
    enc_sec = time.perf_counter() - t
    # rank_features 이름에 '.' 이 있으면 객체 경로로 해석되므로 뺀다
    svec = [{k: round(v, 4) for k, v in d.items() if "." not in k and v > 0} for d in svec]
    toks = [" ".join(bm.tokenize(x)) for x in base]
    print(f"원본 {len(base)}개 인코딩 {enc_sec:.0f}s, 문서당 sparse 토큰 평균 {statistics.mean(map(len, svec)):.0f}개")

    s.delete(f"{OS}/{name}")
    r = s.put(f"{OS}/{name}", json=os_index_body())
    if r.status_code >= 300:
        raise RuntimeError(r.text[:500])
    rnd = random.Random(7)
    groups = [f"g{i:03d}" for i in range(200)]
    t = time.perf_counter()
    for start in range(0, n, 2000):
        lines = []
        for i in range(start, min(n, start + 2000)):
            j = i % len(base)
            acl = ["all"] if rnd.random() < 0.2 else rnd.sample(groups, 2)
            lines.append(json.dumps({"index": {"_index": name, "_id": f"S{i:06d}"}}))
            lines.append(json.dumps({"doc_id": f"S{i:06d}", "acl": acl, "tokens": toks[j], "sparse": svec[j]},
                                    ensure_ascii=False))
        r = s.post(f"{OS}/_bulk", data=("\n".join(lines) + "\n").encode("utf-8"),
                   headers={"Content-Type": "application/x-ndjson"})
        if r.status_code >= 300 or r.json().get("errors"):
            raise RuntimeError(r.text[:800])
    s.put(f"{OS}/{name}/_settings", json={"index": {"refresh_interval": "1s"}})
    s.post(f"{OS}/{name}/_refresh")
    load_sec = time.perf_counter() - t
    s.post(f"{OS}/{name}/_forcemerge?max_num_segments=1", timeout=600)
    st = s.get(f"{OS}/{name}/_stats/store,docs").json()["indices"][name]["primaries"]
    print(f"적재 {n:,}개 {load_sec:.0f}s ({n / load_sec:,.0f}개/초), 디스크 {st['store']['size_in_bytes'] / 1e9:.2f}GB")

    user_acl = ["all"] + groups[:10]
    qs = [q["query"] for q in load_queries()] + [q["query"] for q in load_mismatch()]

    # 질의 본문: BM25 / sparse (neural_sparse 가 안 되면 rank_feature 합으로)
    def q_bm25(q, acl):
        b = {"bool": {"must": [{"match": {"tokens": " ".join(bm.tokenize(q))}}]}}
        if acl:
            b["bool"]["filter"] = [{"terms": {"acl": acl}}]
        return {"size": 20, "_source": ["doc_id"], "query": b}

    def q_sparse(q, acl, mode):
        qv = {k: v for k, v in sp.query(q).items() if "." not in k}
        if mode == "neural_sparse":
            core = {"neural_sparse": {"sparse": {"query_tokens": qv}}}
        else:
            core = {"bool": {"should": [{"rank_feature": {"field": f"sparse.{k}", "linear": {}, "boost": v}}
                                        for k, v in qv.items()]}}
        b = {"bool": {"must": [core]}}
        if acl:
            b["bool"]["filter"] = [{"terms": {"acl": acl}}]
        return {"size": 20, "_source": ["doc_id"], "query": b}

    mode = "neural_sparse"
    r = s.post(f"{OS}/{name}/_search", json=q_sparse(qs[0], None, mode))
    if r.status_code >= 300:
        print("neural_sparse 질의 실패 → rank_feature 로 대체:", r.text[:200])
        mode = "rank_feature"

    def timed(fn):
        ts = []
        for q in qs:
            t0 = time.perf_counter()
            fn(q)
            ts.append((time.perf_counter() - t0) * 1000)
        ts.sort()
        return f"평균 {statistics.mean(ts):.1f}ms · p90 {ts[int(len(ts) * 0.9)]:.1f}ms"

    def one(body):
        r = s.post(f"{OS}/{name}/_search", json=body)
        if r.status_code >= 300:
            raise RuntimeError(r.text[:300])

    def both(q, acl):
        lines = ""
        for body in (q_bm25(q, acl), q_sparse(q, acl, mode)):
            lines += json.dumps({}) + "\n" + json.dumps(body, ensure_ascii=False) + "\n"
        r = s.post(f"{OS}/{name}/_msearch", data=lines.encode("utf-8"),
                   headers={"Content-Type": "application/x-ndjson"})
        if r.status_code >= 300:
            raise RuntimeError(r.text[:300])

    for q in qs[:20]:  # 캐시·JIT 준비
        one(q_bm25(q, None)); one(q_sparse(q, None, mode))
    print(f"sparse 질의 방식: {mode}, 질문 {len(qs)}개, 각 상위 20")
    for acl in (None, user_acl):
        tag = "권한 필터" if acl else "필터 없음"
        print(f"[{tag}] BM25 {timed(lambda q: one(q_bm25(q, acl)))} | "
              f"sparse {timed(lambda q: one(q_sparse(q, acl, mode)))} | "
              f"BM25+sparse _msearch {timed(lambda q: both(q, acl))}")
    t = time.perf_counter()
    for q in qs:
        sp.query(q)
    print(f"(참고) 질문 sparse 변환 평균 {(time.perf_counter() - t) / len(qs) * 1000:.2f}ms, 검색 시간에 포함됨")


if __name__ == "__main__":
    {"expand": expand, "recall": recall, "speed": speed, "os": os_bench}[sys.argv[1]]()
