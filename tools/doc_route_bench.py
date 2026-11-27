"""문서 후보 선택(설계서 12.1) 비교 측정: 문서 BM25 vs e5-small-ko vs 정적 임베딩(69m) vs 혼합(RRF).

사용:
  python tools\\doc_route_bench.py recall          # A·B·C 재현율 (평가셋 100문항)
  python tools\\doc_route_bench.py speed           # D: 임베딩 처리량 → 문서 10만·청크 100만 환산
"""
import json
import statistics
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, r"D:\Project\Q&A\AgenticRAG\src")

from decide_rag.bm25_index import BM25Index  # noqa: E402

E5_DIR = r"D:\Project\SimpleRAG\dist-rust\models\e5-small-ko"
STATIC_DIR = r"D:\Tools\models\static-retrieval-multilingual-69m-v1"
FIRST_CHARS = 300   # 문서 색인 글에 넣을 첫 청크 앞부분 글자 수 (설계서 12.1)
KS = (1, 3, 5)
RRF_K = 60


#------------------------------------------------------------------
# 정적 임베딩 모델 (model2vec / sentence-transformers StaticEmbedding 방식)
#=> 신경망 계산 없이 "토큰별 벡터표"에서 벡터를 꺼내 평균만 낸다. 그래서 매우 빠르다.
#    1) tokenizer.json 으로 문장을 토큰 ID 로 바꾼다 (특수 토큰 없이)
#    2) 각 토큰 ID 의 벡터(384차원)를 표에서 꺼내 평균
#    3) 코사인 비교를 위해 길이 1 로 정규화
#
# -필드: tok = 토크나이저
# -필드: emb = 토큰 벡터표 (어휘 179,936 × 384)
#------------------------------------------------------------------
class StaticEmbedder:
    #------------------------------------------------------------------
    # 모델 불러오기
    #=> safetensors 의 벡터표와 tokenizer.json 을 읽는다 (torch 불필요).
    #
    # -in: model_dir = 모델 폴더
    #
    # -out: 없음
    # -out: error = 파일이 없으면 예외 전파
    #------------------------------------------------------------------
    def __init__(self, model_dir: str):
        from safetensors.numpy import load_file
        from tokenizers import Tokenizer
        self.name = Path(model_dir).name
        self.tok = Tokenizer.from_file(str(Path(model_dir) / "tokenizer.json"))
        self.emb = load_file(str(Path(model_dir) / "model.safetensors"))["embeddings"]

    #------------------------------------------------------------------
    # 문장들 → 벡터
    #=> 토큰 벡터 평균 후 정규화. prefix 는 e5 와 호출 모양을 맞추려는 것이고 이 모델은 쓰지 않는다.
    #
    # -in: texts = 문장 목록
    # -in: prefix = 무시 (모델 설정의 query/document 프롬프트가 빈 문자열)
    #
    # -out: vecs = (문장 수 × 384) float32, 행마다 길이 1
    # -out: error = 없음 (토큰이 하나도 없으면 0 벡터)
    #------------------------------------------------------------------
    def embed(self, texts: list[str], prefix: str = "") -> np.ndarray:
        out = np.zeros((len(texts), self.emb.shape[1]), np.float32)
        for i, e in enumerate(self.tok.encode_batch(texts, add_special_tokens=False)):
            if e.ids:
                v = self.emb[e.ids].mean(0)
                out[i] = v / max(np.linalg.norm(v), 1e-9)
        return out


#------------------------------------------------------------------
# 청크 → 문서 색인 글
#=> 설계서 12.1 의 문서 단위 색인 글을 만든다.
#    1) 제목 + 헤딩 경로 목록 (중복·숫자만 있는 헤딩 제외)
#    2) with_first=True 면 첫 청크 본문 앞 FIRST_CHARS 자를 덧붙인다
#
# -in: chunks = data\chunks.jsonl 행 목록
# -in: with_first = 첫 청크 앞부분 포함 여부
#
# -out: docs = [{"doc_id","title","text"}, …] (문서 등장 순서)
# -out: error = 없음
#------------------------------------------------------------------
def doc_texts(chunks: list[dict], with_first: bool) -> list[dict]:
    docs: "OrderedDict[str, dict]" = OrderedDict()
    for c in chunks:
        d = docs.setdefault(c["doc_id"], {"doc_id": c["doc_id"], "title": c["title"], "heads": [],
                                          "first": c.get("body") or c["text"]})
        for h in c["path"]:
            if h not in d["heads"] and not h.replace(".", "").strip().isdigit():
                d["heads"].append(h)
    out = []
    for d in docs.values():
        text = d["title"] + "\n" + "\n".join(d["heads"])
        if with_first:
            text += "\n" + d["first"][:FIRST_CHARS]
        out.append({"doc_id": d["doc_id"], "chunk_id": d["doc_id"], "title": d["title"], "text": text})
    return out


#------------------------------------------------------------------
# 평가 질의 만들기
#=> 답할 수 있는 문항만 쓰고, 멀티턴이면 앞 질문(최대 2개)을 붙여 에이전트와 같은 질의로 만든다.
#
# -in: 없음
#
# -out: qs = [{"id","type","query","gold"(정답 문서 ID 집합)}, …]
# -out: error = 평가셋 파일이 없으면 예외 전파
#------------------------------------------------------------------
def load_queries() -> list[dict]:
    qs = []
    for line in open(ROOT / "eval" / "evalset_v2.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if not r["doc_ids"]:
            continue  # 문서에 없는 질문은 정답 문서가 없으니 뺀다
        prev = [h["q"] for h in r["history"]][-2:]
        qs.append({"id": r["id"], "type": r["type"], "query": " ".join(prev + [r["question"]]),
                   "gold": set(r["doc_ids"])})
    return qs


#------------------------------------------------------------------
# 순위 목록 → 정답 문서의 최고 순위
#=> 정답 문서 중 가장 앞에 나온 순위(1부터). 없으면 큰 수.
#
# -in: ranked = 문서 ID 순위 목록
# -in: gold = 정답 문서 ID 집합
#
# -out: rank = 1 이상 정수 (못 찾으면 999)
# -out: error = 없음
#------------------------------------------------------------------
def best_rank(ranked: list[str], gold: set) -> int:
    for i, d in enumerate(ranked, 1):
        if d in gold:
            return i
    return 999


#------------------------------------------------------------------
# RRF 혼합
#=> 여러 순위 목록을 1/(RRF_K+순위) 합으로 다시 줄 세운다 (프로토타입 혼합 검색과 같은 식).
#
# -in: lists = 문서 ID 순위 목록들
#
# -out: ranked = 합친 순위 목록
# -out: error = 없음
#------------------------------------------------------------------
def rrf(*lists: list[str]) -> list[str]:
    sc: dict[str, float] = {}
    for lst in lists:
        for i, d in enumerate(lst, 1):
            sc[d] = sc.get(d, 0.0) + 1.0 / (RRF_K + i)
    return sorted(sc, key=lambda d: -sc[d])


#------------------------------------------------------------------
# A·B·C 재현율 측정
#=> 문서 색인 글 두 가지(제목+헤딩 / +첫 청크)마다 방식별로 정답 문서가 상위 k 에 드는 비율을 잰다.
#    1) A: 문서 BM25 (Kiwi 토큰, 프로토타입과 같은 BM25Okapi)
#    2) B: e5-small-ko 코사인,  C: 정적 69m 코사인
#    3) A+B, A+C: RRF 혼합
#    4) BM25 가 놓친 문항(순위 > k)을 벡터가 구한 수도 센다
#
# -in: 없음
#
# -out: 없음 (표를 출력하고 eval\runs\doc_route_bench.json 에 저장)
# -out: error = 모델·데이터 파일이 없으면 예외 전파
#------------------------------------------------------------------
def recall():
    from vector_index import OnnxEmbedder, PASSAGE_PREFIX, QUERY_PREFIX
    chunks = [json.loads(l) for l in open(ROOT / "data" / "legacy_stage1" / "chunks.jsonl", encoding="utf-8")]
    qs = load_queries()
    e5 = OnnxEmbedder(E5_DIR)
    st = StaticEmbedder(STATIC_DIR)
    result = {"n_questions": len(qs), "variants": {}}
    for variant, with_first in (("제목+헤딩", False), ("제목+헤딩+첫청크300자", True)):
        docs = doc_texts(chunks, with_first)
        ids = [d["doc_id"] for d in docs]
        bm = BM25Index(docs)
        dv = {"B_e5": e5.embed([d["text"] for d in docs], PASSAGE_PREFIX),
              "C_static": st.embed([d["text"] for d in docs])}
        ranks = {m: [] for m in ("A_bm25", "B_e5", "C_static", "A+B", "A+C")}
        zero_bm25 = 0
        for q in qs:
            hits = bm.search(q["query"], k=len(ids))
            a = [h[0]["doc_id"] for h in hits]
            # BM25 점수 0 인 문서는 결과에 없다 → 공유 단어가 전혀 없는 경우
            if not (set(a) & q["gold"]):
                zero_bm25 += 1
            # 점수 0 문서는 BM25 가 돌려주지 않으므로 "못 찾음" 으로 둔다 (실제 엔진과 같음)
            vecs = {"B_e5": e5.embed([q["query"]], QUERY_PREFIX)[0], "C_static": st.embed([q["query"]])[0]}
            lists = {"A_bm25": a}
            for m, qv in vecs.items():
                lists[m] = [ids[i] for i in np.argsort(-(dv[m] @ qv))]
            lists["A+B"] = rrf(lists["A_bm25"], lists["B_e5"])
            lists["A+C"] = rrf(lists["A_bm25"], lists["C_static"])
            for m, lst in lists.items():
                ranks[m].append(best_rank(lst, q["gold"]))
        table = {}
        for m, rs in ranks.items():
            table[m] = {f"@{k}": round(sum(r <= k for r in rs) / len(rs), 3) for k in KS}
            table[m]["평균순위"] = round(statistics.mean(rs), 2)
        rescue = {}
        for k in KS:
            miss = [i for i, r in enumerate(ranks["A_bm25"]) if r > k]
            rescue[f"@{k}"] = {"BM25놓침": len(miss),
                               "e5가구함": sum(ranks["B_e5"][i] <= k for i in miss),
                               "static가구함": sum(ranks["C_static"][i] <= k for i in miss)}
        result["variants"][variant] = {"recall": table, "bm25_공유단어없음": zero_bm25, "rescue": rescue}
        print(f"\n## {variant}  (문서 {len(ids)}개, 질문 {len(qs)}개, BM25 공유단어 없음 {zero_bm25}건)")
        print("| 방식 | @1 | @3 | @5 | 평균순위 |")
        print("|---|---|---|---|---|")
        for m, t in table.items():
            print(f"| {m} | {t['@1']} | {t['@3']} | {t['@5']} | {t['평균순위']} |")
        print("BM25 가 놓친 문항 중 벡터가 구한 수:", json.dumps(rescue, ensure_ascii=False))
    out = ROOT / "eval" / "runs" / "doc_route_bench.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n저장:", out)


#------------------------------------------------------------------
# 처리량 1회 측정
#=> 주어진 글 목록을 임베딩하는 데 걸린 시간을 재서 초당 개수를 돌려준다.
#
# -in: fn = texts → 벡터 함수
# -in: texts = 글 목록
#
# -out: per_sec = 초당 처리 개수
# -out: error = 예외 전파
#------------------------------------------------------------------
def throughput(fn, texts: list[str]) -> float:
    t = time.perf_counter()
    fn(texts)
    return len(texts) / (time.perf_counter() - t)


#------------------------------------------------------------------
# D: 임베딩 속도 측정
#=> 실제 청크·문서 글을 반복해 처리량과 질문 1개 임베딩 시간을 잰다.
#    1) 문서 글(제목+헤딩+첫청크300자), 청크 글(본문 700자 안팎)
#    2) e5 는 느리므로 표본 수를 줄여 재고, 문서 10만·청크 100만은 처리량으로 환산
#    3) 질문 1개 임베딩: 평가셋 질의 100개 각각의 시간 평균·p90
#
# -in: 없음
#
# -out: 없음 (출력)
# -out: error = 예외 전파
#------------------------------------------------------------------
def speed():
    from vector_index import OnnxEmbedder, PASSAGE_PREFIX, QUERY_PREFIX
    chunks = [json.loads(l) for l in open(ROOT / "data" / "legacy_stage1" / "chunks.jsonl", encoding="utf-8")]
    docs = [d["text"] for d in doc_texts(chunks, True)]
    ctext = [c["text"] for c in chunks]
    print(f"문서 글 평균 {statistics.mean(map(len, docs)):.0f}자, 청크 글 평균 {statistics.mean(map(len, ctext)):.0f}자")
    e5 = OnnxEmbedder(E5_DIR)
    st = StaticEmbedder(STATIC_DIR)
    plans = [("e5-small-ko", lambda t: e5.embed(t, PASSAGE_PREFIX), 600, 606),
             ("static-69m", st.embed, 20000, 20000)]
    qs = [q["query"] for q in load_queries()]
    for name, fn, n_doc, n_chunk in plans:
        fn(docs[:4])  # 첫 호출 준비 시간 제외
        dps = throughput(fn, (docs * (n_doc // len(docs) + 1))[:n_doc])
        cps = throughput(fn, (ctext * (n_chunk // len(ctext) + 1))[:n_chunk])
        qfn = (lambda q: e5.embed([q], QUERY_PREFIX)) if name.startswith("e5") else (lambda q: st.embed([q]))
        qt = []
        for q in qs:
            t = time.perf_counter()
            qfn(q)
            qt.append((time.perf_counter() - t) * 1000)
        qt.sort()
        print(f"[{name}] 문서 {dps:,.0f}개/초 → 10만 {100000 / dps / 60:.1f}분 | "
              f"청크 {cps:,.0f}개/초 → 100만 {1000000 / cps / 60:.1f}분 | "
              f"질문 1개 평균 {statistics.mean(qt):.2f}ms p90 {qt[int(len(qt) * 0.9)]:.2f}ms "
              f"(표본 문서 {n_doc}, 청크 {n_chunk})")


if __name__ == "__main__":
    {"recall": recall, "speed": speed}[sys.argv[1] if len(sys.argv) > 1 else "recall"]()
