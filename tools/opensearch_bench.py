# -*- coding: utf-8 -*-
"""OpenSearch 측정 도구 (설계서 12.2 확인용).

  python tools\\opensearch_bench.py parity            # 실제 청크 606개: OpenSearch 상위 10 ↔ rank_bm25 상위 10 비교
  python tools\\opensearch_bench.py load 1000000      # 합성 청크 N 개 적재 (색인 시간)
  python tools\\opensearch_bench.py bench             # 검색 시간: 단일 / 권한 필터 / 질문당 7회 (개별 vs _msearch)
  python tools\\opensearch_bench.py stats             # 색인 디스크 크기, OpenSearch 프로세스 메모리

OpenSearch 는 http://127.0.0.1:9200 (보안 플러그인 끔, localhost 만) 을 가정한다.
토큰은 OpenSearch 분석기 대신 Kiwi 토큰을 공백으로 이어 whitespace 필드에 넣는다 (평가해 둔 토큰화 유지).
"""
import json
import random
import statistics as st
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8")

from decide_rag.bm25_index import BM25Index
from decide_rag.chunker import load_chunks
from decide_rag.config import load_config

OS = "http://127.0.0.1:9200"
S = requests.Session()
GROUPS = [f"g{i:03d}" for i in range(200)]
USER_GROUPS = ["all"] + GROUPS[:10]   # 측정용 사용자: 전사 공개 + 부서 그룹 10개


#------------------------------------------------------------------
# 색인 설정·매핑
#=> BM25 계수를 프로토타입(rank_bm25 기본값 k1=1.5, b=0.75)과 맞추고, 검색용 tokens 만 색인한다.
#   본문(text)·제목·경로는 저장만 하고 색인하지 않는다 (검색은 Kiwi 토큰으로만).
#
# -in: 없음
#
# -out: body = 색인 생성 요청 dict
# -out: error = 없음
#------------------------------------------------------------------
def index_body() -> dict:
    return {
        "settings": {"number_of_shards": 1, "number_of_replicas": 0, "refresh_interval": "-1",
                     "similarity": {"bm25p": {"type": "BM25", "k1": 1.5, "b": 0.75}}},
        "mappings": {"properties": {
            "chunk_id": {"type": "keyword"}, "doc_id": {"type": "keyword"}, "acl": {"type": "keyword"},
            "tokens": {"type": "text", "analyzer": "whitespace", "similarity": "bm25p"},
            "text": {"type": "text", "index": False}, "title": {"type": "keyword", "index": False},
            "path": {"type": "keyword", "index": False}, "page": {"type": "integer", "index": False}}},
    }


#------------------------------------------------------------------
# 색인 새로 만들기
#=> 같은 이름이 있으면 지우고 다시 만든다 (측정용 색인만 다룬다).
#
# -in: name = 색인 이름
#
# -out: 없음
# -out: error = 생성 실패 시 RuntimeError
#------------------------------------------------------------------
def recreate(name: str):
    S.delete(f"{OS}/{name}")
    r = S.put(f"{OS}/{name}", json=index_body())
    if r.status_code >= 300:
        raise RuntimeError(r.text[:500])


#------------------------------------------------------------------
# 대량 적재
#=> 문서 dict 를 묶음(batch)으로 _bulk 에 보내고, 끝나면 refresh 해서 검색 가능하게 한다.
#
# -in: name = 색인 이름
# -in: docs = 문서 dict 반복자 (각 dict 에 "chunk_id")
# -in: batch = 한 번에 보낼 문서 수 (기본 2000)
#
# -out: (n, seconds) = 적재한 수, 걸린 시간
# -out: error = bulk 응답에 오류가 있으면 RuntimeError
#------------------------------------------------------------------
def bulk_load(name: str, docs, batch: int = 2000) -> tuple[int, float]:
    t0 = time.perf_counter()
    n, buf = 0, []
    def flush():  # 모은 묶음을 보낸다
        body = "".join(buf)
        r = S.post(f"{OS}/{name}/_bulk", data=body.encode("utf-8"),
                   headers={"Content-Type": "application/x-ndjson"}, timeout=600)
        j = r.json()
        if j.get("errors"):
            bad = next(i for i in j["items"] if "error" in i["index"])
            raise RuntimeError(json.dumps(bad, ensure_ascii=False)[:500])
        buf.clear()
    for d in docs:
        buf.append(json.dumps({"index": {"_id": d["chunk_id"]}}) + "\n" + json.dumps(d, ensure_ascii=False) + "\n")
        n += 1
        if len(buf) >= batch:
            flush()
            if n % 100000 == 0:
                print(f"  적재 {n:,} ({time.perf_counter() - t0:.0f}s)", flush=True)
    if buf:
        flush()
    S.post(f"{OS}/{name}/_refresh", timeout=600)
    S.put(f"{OS}/{name}/_settings", json={"index": {"refresh_interval": "1s"}})
    return n, time.perf_counter() - t0


#------------------------------------------------------------------
# 검색 질의 만들기
#=> Kiwi 토큰 match + (선택) 권한 필터·문서 범위 필터. 필터는 점수에 영향이 없다.
#
# -in: qtokens = 질의 Kiwi 토큰 목록
# -in: k = 상위 몇 개
# -in: acl = 사용자 그룹 목록 (None 이면 권한 필터 없음)
# -in: doc_ids = 문서 범위 (None 이면 전체)
#
# -out: body = _search 요청 dict
# -out: error = 없음
#------------------------------------------------------------------
def query_body(qtokens: list[str], k: int = 10, acl=None, doc_ids=None) -> dict:
    filt = []
    if acl:
        filt.append({"terms": {"acl": acl}})
    if doc_ids:
        filt.append({"terms": {"doc_id": sorted(doc_ids)}})
    return {"size": k, "_source": ["chunk_id"], "track_total_hits": False,
            "query": {"bool": {"must": {"match": {"tokens": " ".join(qtokens)}}, "filter": filt}}}


#------------------------------------------------------------------
# 검색 한 번
#=> _search 를 보내 (chunk_id 목록, 왕복 시간, OpenSearch 내부 took) 를 돌려준다.
#
# -in: name = 색인 이름
# -in: body = query_body 결과
#
# -out: (ids, wall_ms, took_ms)
# -out: error = HTTP 오류 시 RuntimeError
#------------------------------------------------------------------
def search(name: str, body: dict):
    t0 = time.perf_counter()
    r = S.post(f"{OS}/{name}/_search", json=body)
    wall = (time.perf_counter() - t0) * 1000
    if r.status_code >= 300:
        raise RuntimeError(r.text[:300])
    j = r.json()
    return [h["_source"]["chunk_id"] for h in j["hits"]["hits"]], wall, j["took"]


#------------------------------------------------------------------
# 여러 검색 한 번에 (_msearch)
#=> 재검색 라운드의 질의 3개를 한 요청으로 보낸다 (왕복 횟수를 줄이기 위함).
#
# -in: name = 색인 이름
# -in: bodies = query_body 목록
#
# -out: (결과 ids 목록들, wall_ms)
# -out: error = HTTP 오류 시 RuntimeError
#------------------------------------------------------------------
def msearch(name: str, bodies: list[dict]):
    nd = "".join(json.dumps({"index": name}) + "\n" + json.dumps(b, ensure_ascii=False) + "\n" for b in bodies)
    t0 = time.perf_counter()
    r = S.post(f"{OS}/_msearch", data=nd.encode("utf-8"), headers={"Content-Type": "application/x-ndjson"})
    wall = (time.perf_counter() - t0) * 1000
    if r.status_code >= 300:
        raise RuntimeError(r.text[:300])
    return [[h["_source"]["chunk_id"] for h in x["hits"]["hits"]] for x in r.json()["responses"]], wall


#------------------------------------------------------------------
# 평가셋 질문 + 실제 색인 준비
#=> 실제 청크로 BM25Index(Kiwi) 를 만들고, 평가셋 질문 100개를 토큰으로 바꾼다.
#
# -in: 없음
#
# -out: (chunks, ix, qtoks) = 청크, BM25Index, 질문별 토큰 목록
# -out: error = data\chunks.jsonl 이 없으면 FileNotFoundError
#------------------------------------------------------------------
def prepare():
    chunks = load_chunks(load_config())
    ix = BM25Index(chunks)
    items = [json.loads(l) for l in (ROOT / "eval" / "evalset_v2.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    # 멀티턴은 서비스와 같게 이전 질문을 이어 붙인 질의로 잰다
    qs = [" ".join([h["q"] for h in (it.get("history") or [])][-2:] + [it["question"]]) for it in items]
    return chunks, ix, [ix.tokenize(q) for q in qs]


#------------------------------------------------------------------
# 점수 일치 확인 (실제 청크 606개)
#=> 같은 토큰·같은 계수로 OpenSearch 상위 10 과 rank_bm25 상위 10 이 얼마나 같은지 본다.
#   idf 식이 조금 달라(rank_bm25: log((N-n+.5)/(n+.5)), Lucene: log(1+(N-n+.5)/(n+.5))) 완전히 같지는 않을 수 있다.
#
# -in: 없음
#
# -out: 없음 (화면 출력)
# -out: error = OpenSearch 연결 실패 시 예외
#------------------------------------------------------------------
def parity():
    chunks, ix, qtoks = prepare()
    recreate("chunks_real")
    n, sec = bulk_load("chunks_real", ({"chunk_id": c["chunk_id"], "doc_id": c["doc_id"], "acl": ["all"],
                                        "tokens": " ".join(t), "text": c["text"], "title": c["title"],
                                        "path": c["path"], "page": c["page"]} for c, t in zip(chunks, ix.tokens)))
    print(f"실제 청크 {n}개 적재 {sec:.1f}s")
    top1, ov, same_set = 0, [], 0
    for q in qtoks:
        a = [c["chunk_id"] for c, _ in ix.search(" ".join(q), 10)] if q else []
        # ix.search 는 다시 토큰화하므로 토큰을 이어 붙여도 같은 토큰이 나온다
        b, _, _ = search("chunks_real", query_body(q, 10))
        top1 += bool(a and b and a[0] == b[0])
        ov.append(len(set(a) & set(b)) / max(1, len(a)))
        same_set += set(a) == set(b)
    print(f"질문 {len(qtoks)}개: 1위 일치 {top1}개, 상위10 겹침 평균 {st.mean(ov):.3f}, 상위10 집합 완전 일치 {same_set}개")


#------------------------------------------------------------------
# 합성 청크 N 개 적재
#=> 실제 청크 606개의 토큰·본문을 반복해 N 개로 만든다. 문서는 청크 10개씩(N/10 개 문서).
#   권한: 문서의 20% 는 전사 공개("all"), 나머지는 부서 그룹 200개 중 2개를 무작위로 준다.
#   (같은 글이 반복되므로 디스크 압축률은 실제보다 좋게 나온다)
#
# -in: n = 청크 수
#
# -out: 없음 (화면 출력)
# -out: error = OpenSearch 오류 시 RuntimeError
#------------------------------------------------------------------
def load(n: int):
    chunks, ix, _ = prepare()
    rnd = random.Random(7)
    acl_of = {}
    def gen():  # 합성 문서를 하나씩 만든다
        for i in range(n):
            c = chunks[i % len(chunks)]
            did = f"S{i // 10:06d}"
            if did not in acl_of:
                acl_of[did] = ["all"] if rnd.random() < 0.2 else rnd.sample(GROUPS, 2)
            yield {"chunk_id": f"{did}#{i % 10:02d}", "doc_id": did, "acl": acl_of[did],
                   "tokens": " ".join(ix.tokens[i % len(chunks)]), "text": c["text"], "title": c["title"],
                   "path": c["path"], "page": c["page"]}
    recreate("chunks_1m")
    cnt, sec = bulk_load("chunks_1m", gen())
    vis = sum(1 for a in acl_of.values() if set(a) & set(USER_GROUPS)) / len(acl_of)
    print(f"합성 청크 {cnt:,}개 적재 {sec:.0f}s ({cnt / sec:,.0f}개/초), 측정 사용자에게 보이는 문서 비율 {vis:.1%}")


#------------------------------------------------------------------
# 검색 시간 측정
#=> 평가셋 질문 100개로 1M 색인에서 잰다. 먼저 한 바퀴 돌려 캐시를 데운 뒤 잰다.
#    1) 단일 검색 (필터 없음 / 권한 필터)
#    2) 질문 하나 = 1라운드 1회 + 재검색 2라운드 × 질의 3개 = 7회 (권한 필터 포함)
#       - 개별 7회 요청 vs 1라운드 1회 + _msearch 2회
#
# -in: 없음
#
# -out: 없음 (화면 출력)
# -out: error = OpenSearch 오류 시 RuntimeError
#------------------------------------------------------------------
def bench():
    _, ix, qtoks = prepare()
    qtoks = [q for q in qtoks if q]
    name = "chunks_1m"
    rnd = random.Random(3)
    pct = lambda xs, p: sorted(xs)[int(p * (len(xs) - 1))]  # 분위수
    show = lambda tag, xs: print(f"{tag:<34} 평균 {st.mean(xs):6.1f}ms  p50 {pct(xs, .5):6.1f}  p90 {pct(xs, .9):6.1f}  최대 {max(xs):6.1f}")
    for q in qtoks:  # 캐시 데우기
        search(name, query_body(q))
    for tag, acl in (("단일 검색 (필터 없음)", None), ("단일 검색 (권한 필터)", USER_GROUPS)):
        wall, took = [], []
        for q in qtoks:
            _, w, t = search(name, query_body(q, 10, acl))
            wall.append(w)
            took.append(t)
        show(tag + " 왕복", wall)
        show(tag + " 내부 took", took)
    # 재검색 질의 흉내: 원 질의 + 문서 범위 / 원 질의 + 확장어 5개 / 원 질의 + 제목
    ext = ix.tokenize("경조금 휴가 신청 지급 기준 규정")
    sep, multi = [], []
    for q in qtoks:
        docs = {f"S{rnd.randrange(100000):06d}" for _ in range(2)}
        r1 = query_body(q, 10, USER_GROUPS)
        rr = [query_body(q, 5, USER_GROUPS, docs), query_body(q + ext[:5], 5, USER_GROUPS),
              query_body(q + ext[-2:], 5, USER_GROUPS)]
        t0 = time.perf_counter()
        search(name, r1)
        for _ in range(2):
            for b in rr:
                search(name, b)
        sep.append((time.perf_counter() - t0) * 1000)
        t0 = time.perf_counter()
        search(name, r1)
        for _ in range(2):
            msearch(name, rr)
        multi.append((time.perf_counter() - t0) * 1000)
    show("질문당 7회 (개별 요청)", sep)
    show("질문당 7회 (1회 + _msearch 2회)", multi)


#------------------------------------------------------------------
# 디스크·메모리
#=> 색인별 디스크 크기와 OpenSearch(java) 프로세스 메모리(RSS)를 출력한다.
#
# -in: 없음
#
# -out: 없음 (화면 출력)
# -out: error = OpenSearch 연결 실패 시 예외
#------------------------------------------------------------------
def stats():
    for row in S.get(f"{OS}/_cat/indices?format=json&bytes=b").json():
        print(f"색인 {row['index']}: 문서 {int(row['docs.count']):,}개, 디스크 {int(row['store.size']) / 2**30:.2f}GB")
    try:
        import psutil
        for p in psutil.process_iter(["name", "cmdline", "memory_info"]):
            if p.info["name"] and p.info["name"].lower().startswith("java") and "opensearch" in " ".join(p.info["cmdline"] or []).lower():
                print(f"OpenSearch 프로세스 메모리(RSS) {p.info['memory_info'].rss / 2**30:.2f}GB")
    except ImportError:
        print("psutil 없음 — 메모리 측정 생략")
    j = S.get(f"{OS}/_nodes/stats/jvm").json()
    for n in j["nodes"].values():
        m = n["jvm"]["mem"]
        print(f"JVM 힙 사용 {m['heap_used_in_bytes'] / 2**30:.2f}GB / 최대 {m['heap_max_in_bytes'] / 2**30:.2f}GB")


#------------------------------------------------------------------
# 명령 실행
#=> parity / load N / bench / stats
#
# -in: 없음 (sys.argv)
#
# -out: 없음
# -out: error = 알 수 없는 명령이면 SystemExit
#------------------------------------------------------------------
def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "parity":
        parity()
    elif cmd == "load":
        load(int(sys.argv[2]))
    elif cmd == "bench":
        bench()
    elif cmd == "stats":
        stats()
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
