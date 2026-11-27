"""2단계 완료 기준 확인: 합성 100만 청크에서 질문당 검색 p90 (설계서 14장, 50ms 이하).

서비스 코드(OpenSearchEngine)를 그대로 쓰고, 색인만 측정용 색인으로 바꾼다.
  청크 색인 : chunks_1m        (tools\\opensearch_bench.py load 로 만든 합성 100만 청크)
  문서 색인 : docs_sparse_100k (tools\\vocab_gap_bench.py os 로 만든 합성 문서 10만 개, tokens·acl 필드)

질문 하나에서 에이전트가 하는 검색을 3라운드 끝까지 간다고 보고 흉내 낸다 (최악 경우).
  1라운드 : search_many 1개 질의 (k=10)
  재검색   : prf_terms + search_many 3개 질의 (k=5×3, 제외 청크 포함) × 2라운드, search_docs(후보 20)는 처음 한 번
  → 청크 검색 7번 + PRF 2번 + 문서 검색 1번

  python tools\\engine_bench_1m.py
"""
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from decide_rag.config import load_config  # noqa: E402
from decide_rag.search_engine import OpenSearchEngine  # noqa: E402

USER_ACL = ["all"] + [f"g{i:03d}" for i in range(10)]   # opensearch_bench.py 와 같은 사용자 권한


#------------------------------------------------------------------
# 평가 질의 읽기
#=> 평가셋 100문항. 멀티턴은 서비스와 같게 앞 질문 2개를 붙인다.
#
# -in: 없음
#
# -out: qs = 질의 문자열 목록
# -out: error = 파일이 없으면 예외 전파
#------------------------------------------------------------------
def queries() -> list[str]:
    out = []
    for l in open(ROOT / "eval" / "evalset_v2.jsonl", encoding="utf-8"):
        it = json.loads(l)
        out.append(" ".join([h["q"] for h in it.get("history") or []][-2:] + [it["question"]]))
    return out


#------------------------------------------------------------------
# 질문 하나의 검색 전부
#=> 에이전트와 같은 순서·같은 요청 모양으로 엔진을 부르고, 단계별 시간을 돌려준다.
#
# -in: eng = OpenSearchEngine
# -in: q = 질의
# -in: acl = 권한 그룹 (None = 필터 없음)
# -in: rounds = 라운드 수 (기본 3 = 최악 경우, 1 이면 첫 검색만)
#
# -out: t = {"round1","research","total"} 밀리초
# -out: error = OpenSearch 오류 시 RuntimeError
#------------------------------------------------------------------
def one_question(eng, q: str, acl, rounds: int = 3) -> dict:
    t0 = time.perf_counter()
    seen: set[str] = set()
    hits = eng.search_many([(q, 10, None, seen)], acl)[0]
    seen |= {c["chunk_id"] for c, _ in hits}
    t1 = time.perf_counter()
    ctx = [c for c, _ in hits[:5]]
    for r in range(rounds - 1):
        terms = eng.prf_terms([c["chunk_id"] for c in ctx[:2]], q, 5)
        if r == 0:  # DecideJudge 는 문서 후보 검색을 질문당 한 번만 한다 (질의가 같으므로)
            eng.search_docs(q, 20, acl)
        docs = {c["doc_id"] for c in ctx[:2]} or None
        title = ctx[0]["title"] if ctx else ""
        plan = [(q, docs), (q + " " + " ".join(terms), None), (q + " " + title, None)]
        res = eng.search_many([(pq, 5 * 3, d, set(seen)) for pq, d in plan], acl)
        for hs in res:
            seen |= {c["chunk_id"] for c, _ in hs[:5]}
    t2 = time.perf_counter()
    return {"round1": (t1 - t0) * 1000, "research": (t2 - t1) * 1000, "total": (t2 - t0) * 1000}


#------------------------------------------------------------------
# 측정 실행
#=> 준비(캐시) 20문항 뒤, 100문항 × (필터 없음 / 권한 필터) 시간을 잰다.
#   마지막 줄은 OpenSearch 판단 평가(eval/runs/stage2_os_judgeonly.jsonl)의 문항별 실제 라운드 수를 쓴다.
#
# -in: 없음
#
# -out: 없음 (표 출력)
# -out: error = 측정용 색인이 없으면 RuntimeError
#------------------------------------------------------------------
def main():
    sys.stdout.reconfigure(encoding="utf-8")
    eng = OpenSearchEngine(load_config()["search"])
    eng.chunks_index, eng.docs_index = "chunks_1m", "docs_sparse_100k"
    n_chunks = eng._req("GET", f"/{eng.chunks_index}/_count")["count"]
    n_docs = eng._req("GET", f"/{eng.docs_index}/_count")["count"]
    qs = queries()
    for q in qs[:20]:
        one_question(eng, q, None)
    print(f"청크 {n_chunks:,} · 문서 {n_docs:,} · 질문 {len(qs)}개 (3라운드 최악 경우: 청크 검색 7·PRF 2·문서 검색 1)")
    print("| 조건 | 1라운드 평균 | 재검색 2라운드 평균 | 질문당 합계 평균 | p50 | p90 | 최대 |")
    print("|---|---|---|---|---|---|---|")
    real = {}
    rf = ROOT / "eval" / "runs" / "stage2_os_judgeonly.jsonl"
    if rf.exists():
        real = {json.loads(l)["id"]: len(json.loads(l)["rounds"]) for l in open(rf, encoding="utf-8")}
    ids = [json.loads(l)["id"] for l in open(ROOT / "eval" / "evalset_v2.jsonl", encoding="utf-8")]
    runs = [("필터 없음, 3라운드", None, [3] * len(qs)), ("권한 필터, 3라운드", USER_ACL, [3] * len(qs))]
    if real:
        runs.append(("권한 필터, 실제 라운드 분포", USER_ACL, [real.get(i, 3) for i in ids]))
    for tag, acl, rounds in runs:
        rs = [one_question(eng, q, acl, n) for q, n in zip(qs, rounds)]
        tot = sorted(r["total"] for r in rs)
        print(f"| {tag} | {statistics.mean(r['round1'] for r in rs):.1f}ms | "
              f"{statistics.mean(r['research'] for r in rs):.1f}ms | {statistics.mean(tot):.1f}ms | "
              f"{tot[len(tot) // 2]:.1f}ms | {tot[int(len(tot) * 0.9)]:.1f}ms | {tot[-1]:.1f}ms |")


if __name__ == "__main__":
    main()
