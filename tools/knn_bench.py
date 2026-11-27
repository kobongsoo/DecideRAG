# -*- coding: utf-8 -*-
"""합성 벡터로 OpenSearch k-NN 검색 시간 재기 (7단계, 설계서 12.2.2).

  python tools\\knn_bench.py --n 100000            # 색인 decide_knnbench 에 무작위 벡터 n 개 → 검색 시간
  python tools\\knn_bench.py --n 100000 --keep     # 다 재고도 색인을 남긴다 (기본은 지운다)

- 벡터는 무작위 단위 벡터(1024차원)다. 실제 문서 벡터와 분포가 달라 HNSW 정확도는 재지 않는다. 시간·크기만 본다.
- 청크마다 acl 그룹을 무작위로 1개(all 70%, 부서 30%) 넣고, 서비스와 같은 k-NN + 권한 filter 로 찾는다.
- 결과: eval\\runs\\knn_bench_<n>.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8")

INDEX = "decide_knnbench"


#------------------------------------------------------------------
# 분위수
#=> 정렬한 값에서 p 위치(내림)의 값.
#
# -in: xs = 숫자 목록
# -in: p = 0~1
#
# -out: v = 분위수
# -out: error = 빈 목록이면 IndexError
#------------------------------------------------------------------
def q(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))]


#------------------------------------------------------------------
# 무작위 단위 벡터
#=> 정규분포에서 뽑아 길이 1 로 맞춘다 (코사인 검색용).
#
# -in: rng = numpy 난수 생성기
# -in: n = 개수
# -in: dim = 차원
#
# -out: vecs = (n, dim) float32
# -out: error = 없음
#------------------------------------------------------------------
def unit(rng, n, dim):
    v = rng.standard_normal((n, dim)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


#------------------------------------------------------------------
# 시험 실행
#=> 색인 만들기 → bulk 로 n 개 넣기 → 강제 병합(세그먼트 1개) → 검색 시간 측정.
#    1) 서비스와 같은 매핑 (lucene HNSW, cosinesimil)
#    2) 검색: 권한 필터(acl=all 또는 hr) 를 k-NN 안에 넣은 질의, k=20 (리랭커 후보 수)
#    3) 같은 조건으로 _msearch 3개 묶음(재검색 1회 분량)도 잰다
#
# -in: url = OpenSearch 주소
# -in: n = 벡터 수
# -in: dim = 차원 (기본 1024)
# -in: batch = bulk 한 번에 넣을 수
# -in: keep = True 면 끝나고 색인을 지우지 않는다
#
# -out: result = 측정 dict (eval\runs\knn_bench_<n>.json 에도 씀)
# -out: error = OpenSearch 오류 시 RuntimeError
#------------------------------------------------------------------
def run(url: str, n: int, dim: int = 1024, batch: int = 500, keep: bool = False) -> dict:
    s = requests.Session()
    s.delete(f"{url}/{INDEX}")
    body = {"settings": {"index": {"knn": True, "number_of_shards": 1, "number_of_replicas": 0,
                                   "refresh_interval": "-1"}},
            "mappings": {"properties": {
                "acl": {"type": "keyword"},
                "vec": {"type": "knn_vector", "dimension": dim,
                        "method": {"name": "hnsw", "engine": "lucene", "space_type": "cosinesimil"}}}}}
    r = s.put(f"{url}/{INDEX}", json=body)
    if r.status_code >= 400:
        raise RuntimeError(r.text[:300])
    rng = np.random.default_rng(7)
    t0 = time.perf_counter()
    for start in range(0, n, batch):
        m = min(batch, n - start)
        vecs = unit(rng, m, dim)
        groups = np.where(rng.random(m) < 0.7, "all", "hr")
        lines = []
        for i in range(m):
            lines.append(json.dumps({"index": {"_id": str(start + i)}}))
            # 소수 5자리로 줄여 bulk 크기를 줄인다 (서비스는 6자리)
            lines.append(json.dumps({"acl": [groups[i]], "vec": [round(float(x), 5) for x in vecs[i]]}))
        r = s.post(f"{url}/{INDEX}/_bulk", data="\n".join(lines) + "\n",
                   headers={"Content-Type": "application/x-ndjson"}, timeout=600)
        if r.status_code >= 400 or r.json().get("errors"):
            raise RuntimeError(r.text[:300])
        if (start // batch) % 20 == 0:
            print(f"  넣음 {start + m}/{n} ({time.perf_counter() - t0:.0f}s)", flush=True)
    t_bulk = time.perf_counter() - t0
    t1 = time.perf_counter()
    s.post(f"{url}/{INDEX}/_refresh", timeout=3600)
    t_refresh = time.perf_counter() - t1
    t1 = time.perf_counter()
    s.post(f"{url}/{INDEX}/_forcemerge?max_num_segments=1", timeout=7200)
    t_merge = time.perf_counter() - t1
    size = s.get(f"{url}/_cat/indices/{INDEX}?format=json&bytes=b").json()[0]["store.size"]
    # 검색: 처음 몇 번은 그래프를 메모리에 올리므로 데우기 후 잰다
    qs = unit(rng, 120, dim)

    def knn(v, acl):
        return {"size": 20, "_source": False,
                "query": {"knn": {"vec": {"vector": [float(x) for x in v], "k": 20,
                                          "filter": {"bool": {"filter": [{"terms": {"acl": acl}}]}}}}}}
    for v in qs[:20]:
        s.post(f"{url}/{INDEX}/_search", json=knn(v, ["all"]))
    lat = {"all": [], "hr_only": [], "msearch3": []}
    for v in qs[20:]:
        t = time.perf_counter()
        s.post(f"{url}/{INDEX}/_search", json=knn(v, ["all", "hr"]))
        lat["all"].append(time.perf_counter() - t)
        t = time.perf_counter()
        s.post(f"{url}/{INDEX}/_search", json=knn(v, ["hr"]))   # 30% 만 볼 수 있는 사용자
        lat["hr_only"].append(time.perf_counter() - t)
        nd = "".join(json.dumps({"index": INDEX}) + "\n" + json.dumps(knn(v, ["all"])) + "\n" for _ in range(3))
        t = time.perf_counter()
        s.post(f"{url}/_msearch", data=nd, headers={"Content-Type": "application/x-ndjson"})
        lat["msearch3"].append(time.perf_counter() - t)
    out = {"n": n, "dim": dim, "bulk_sec": round(t_bulk, 1), "refresh_sec": round(t_refresh, 1),
           "forcemerge_sec": round(t_merge, 1), "store_bytes": int(size),
           "latency_ms": {k: {"p50": round(q(v, .5) * 1000, 1), "p90": round(q(v, .9) * 1000, 1),
                              "max": round(max(v) * 1000, 1)} for k, v in lat.items()}}
    (ROOT / "eval" / "runs" / f"knn_bench_{n}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                                                encoding="utf-8")
    if not keep:
        s.delete(f"{url}/{INDEX}")
    return out


#------------------------------------------------------------------
# 명령 실행
#=> --url, --n, --dim, --keep 을 받아 run 을 부르고 결과를 출력한다.
#
# -in: 없음 (sys.argv)
#
# -out: 없음
# -out: error = 오류 전파
#------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:9200")
    ap.add_argument("--n", type=int, default=100000)
    ap.add_argument("--dim", type=int, default=1024)
    ap.add_argument("--keep", action="store_true")
    a = ap.parse_args()
    print(json.dumps(run(a.url, a.n, a.dim, keep=a.keep), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
