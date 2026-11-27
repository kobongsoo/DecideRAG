# -*- coding: utf-8 -*-
"""Answer Correctness 채점 (4단계 답 고치기 비교용). 판정자는 Claude, 의미 유사도는 e5.

  python eval\\ac.py packets --runs base a1 a2 …            # 판정 안 된 답만 eval\\ac\\packets_<회차>_<NN>.jsonl 로
  python eval\\ac.py check <packets> <verdicts>              # 판정 파일 형식 검사
  python eval\\ac.py score --runs base a1 a2 … --base base   # 점수·짝지은 부트스트랩 95% 구간 → eval\\runs\\ac_summary.json

- 실행 이름 x 는 eval\\runs\\ans_x.jsonl 이다.
- 같은 (질문, 답 문장, 정답) 은 한 번만 판정한다. 프로토타입 판정(설계서 11장)도 같으면 다시 쓴다.
- 판정 지침: eval\\ac_judge_guide.md (프로토타입 지침과 같은 규칙, 경로만 다름).
- RAGAS 정의 (프로토타입 answer_correctness.py 와 같음):
    사실 F1 = TP / (TP + 0.5·(FP + FN)),  AC = 0.75 × 사실 F1 + 0.25 × 코사인(답, 정답)
"""
import argparse
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.stdout.reconfigure(encoding="utf-8")

from decide_rag.answer import judge_question  # noqa: E402

AC_DIR = HERE / "ac"
PROTO = Path(r"D:\Project\Q&A\AgenticRAG")
PROTO_AC = PROTO / "out" / "v2" / "ac"
E5_DIR = r"D:\Project\SimpleRAG\dist-rust\models\e5-small-ko"
W_FACT, W_SIM = 0.75, 0.25


#------------------------------------------------------------------
# jsonl 읽기
#=> 빈 줄을 건너뛰고 dict 목록으로.
#
# -in: p = 경로
#
# -out: rows = dict 목록
# -out: error = 파일이 없으면 FileNotFoundError
#------------------------------------------------------------------
def read_jsonl(p) -> list[dict]:
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


#------------------------------------------------------------------
# 판정 묶음의 비교 키
#=> 같은 질문·같은 답 문장·같은 정답이면 같은 판정을 쓴다 (공백 끝만 정리).
#
# -in: question = 판정용 질문 (멀티턴이면 이전 대화 포함)
# -in: answer = 답
# -in: reference = 정답
#
# -out: key = 튜플
# -out: error = 없음
#------------------------------------------------------------------
def pkey(question: str, answer: str, reference: str) -> tuple:
    return question.strip(), answer.strip(), reference.strip()


#------------------------------------------------------------------
# 이미 있는 판정 모으기
#=> 이 프로젝트 eval\ac 와 프로토타입 out\v2\ac 의 packets·verdicts 를 짝지어 키 → 판정 표를 만든다.
#   프로토타입의 중복 판정분(dup)은 원래 판정과 같은 키라 앞의 것을 쓴다.
#
# -in: 없음
#
# -out: cache = {키: 판정 dict}
# -out: error = 없음 (폴더가 없으면 빈 표)
#------------------------------------------------------------------
def load_cache() -> dict:
    cache = {}
    for d in (AC_DIR, PROTO_AC):
        if not d.exists():
            continue
        packets = {p["pid"]: p for f in sorted(d.glob("packets_*.jsonl")) for p in read_jsonl(f)}
        for f in sorted(d.glob("verdicts_*.jsonl")):
            for v in read_jsonl(f):
                p = packets.get(v["pid"])
                if p:
                    cache.setdefault(pkey(p["question"], p["answer"], p["reference"]), v)
    return cache


#------------------------------------------------------------------
# 실행 결과에서 채점할 답 모으기
#=> 답할 수 있는 문항만, 오류 없는 줄만 (질문·답·정답) 으로 모은다.
#
# -in: runs = 실행 이름 목록
#
# -out: rows = [(실행 이름, 문항 dict, 결과 dict, 키)]
# -out: error = 실행 파일이 없으면 FileNotFoundError
#------------------------------------------------------------------
def collect(runs: list[str]) -> list[tuple]:
    items = {x["id"]: x for x in read_jsonl(HERE / "evalset_v2.jsonl")}
    rows = []
    for name in runs:
        for r in read_jsonl(HERE / "runs" / f"ans_{name}.jsonl"):
            it = items[r["id"]]
            if not it["answerable"] or "error" in r:
                continue
            q = judge_question(it["question"], it.get("history") or [])
            rows.append((name, it, r, pkey(q, r["answer"], it["reference"])))
    return rows


#------------------------------------------------------------------
# 판정 묶음 만들기
#=> 아직 판정이 없는 (질문, 답, 정답) 만 골라 새 회차 파일로 나눠 쓴다.
#    1) 어느 실행의 답인지는 넣지 않는다 (판정자가 변형을 모르게)
#    2) 순서를 섞고 pid 는 "<회차>-<번호>"
#
# -in: runs = 실행 이름 목록
# -in: batches = 파일 수
#
# -out: 없음 (eval\ac\packets_<회차>_<NN>.jsonl)
# -out: error = 실행 파일이 없으면 FileNotFoundError
#------------------------------------------------------------------
def make_packets(runs: list[str], batches: int):
    cache = load_cache()
    todo = {}
    for _, _, _, k in collect(runs):
        if k not in cache:
            todo.setdefault(k, None)
    AC_DIR.mkdir(exist_ok=True)
    rnd = random.Random(len(cache))
    keys = list(todo)
    rnd.shuffle(keys)
    if not keys:
        print("새로 판정할 답 없음")
        return
    # 회차 번호는 이미 있는 파일 다음 번호
    rnd_no = 1 + max([int(f.stem.split("_")[1]) for f in AC_DIR.glob("packets_*_*.jsonl")] or [0])
    per = -(-len(keys) // batches)
    for b in range(batches):
        part = keys[b * per:(b + 1) * per]
        if not part:
            continue
        lines = [json.dumps({"pid": f"r{rnd_no}-{b * per + i + 1:04d}", "question": q, "answer": a, "reference": ref},
                            ensure_ascii=False) for i, (q, a, ref) in enumerate(part)]
        (AC_DIR / f"packets_{rnd_no}_{b + 1:02d}.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"판정 있음 {len(cache)}개 재사용 · 새로 판정 {len(keys)}개 → {AC_DIR} packets_{rnd_no}_*.jsonl")


#------------------------------------------------------------------
# 판정 한 줄 검사
#=> tp/fp/fn 이 {"statement": 문자열} 목록인지, 정답 쪽 사실(tp+fn)이 하나 이상인지 본다.
#
# -in: v = 판정 dict
#
# -out: errs = 문제 설명 목록
# -out: error = 없음
#------------------------------------------------------------------
def check_verdict(v: dict) -> list[str]:
    errs = []
    for k in ("tp", "fp", "fn"):
        xs = v.get(k)
        if not isinstance(xs, list) or not all(isinstance(x, dict) and isinstance(x.get("statement"), str)
                                               and x["statement"].strip() for x in xs):
            errs.append(f"{k} 는 {{\"statement\": 문장}} 목록이어야 함")
    if not errs and not (v["tp"] or v["fn"]):
        errs.append("tp 와 fn 이 모두 비었음 (정답의 사실은 둘 중 하나에 들어가야 함)")
    return errs


#------------------------------------------------------------------
# 판정 파일 검사 실행
#=> 묶음 파일과 대조해 빠진 pid·형식 오류를 출력한다.
#
# -in: packets_path = 묶음 파일
# -in: verdicts_path = 판정 파일
#
# -out: ok = 문제 없으면 True
# -out: error = 파일이 없으면 FileNotFoundError
#------------------------------------------------------------------
def check_files(packets_path: str, verdicts_path: str) -> bool:
    pids = {p["pid"] for p in read_jsonl(packets_path)}
    bad, seen = 0, set()
    for i, line in enumerate(Path(verdicts_path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            v = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"{i}행 JSON 오류: {e}")
            bad += 1
            continue
        if v.get("pid") not in pids:
            print(f"{i}행 모르는 pid: {v.get('pid')}")
            bad += 1
            continue
        seen.add(v["pid"])
        for e in check_verdict(v):
            print(f"{v['pid']}: {e}")
            bad += 1
    missing = sorted(pids - seen)
    if missing:
        print("빠진 pid:", ", ".join(missing))
        bad += len(missing)
    print(f"판정 {len(seen)}/{len(pids)} · 문제 {bad} 건")
    return bad == 0


#------------------------------------------------------------------
# 사실 F1
#=> RAGAS 공식 TP / (TP + 0.5·(FP+FN)). 분모가 0 이면 0.
#
# -in: v = 판정 dict
#
# -out: f1 = 0~1
# -out: error = 없음
#------------------------------------------------------------------
def fact_f1(v: dict) -> float:
    tp, fp, fn = len(v["tp"]), len(v["fp"]), len(v["fn"])
    den = tp + 0.5 * (fp + fn)
    return tp / den if den else 0.0


#------------------------------------------------------------------
# 짝지은 부트스트랩 95% 구간
#=> 같은 문항끼리의 점수 차를 복원 추출로 n_boot 번 다시 뽑아 평균 차의 2.5%·97.5% 분위를 구한다.
#
# -in: diffs = 문항별 차이 목록
# -in: n_boot = 반복 수 (기본 10000)
# -in: seed = 난수 씨앗 (기본 7, 다시 돌려도 같은 구간)
#
# -out: (mean, lo, hi) = 평균 차, 95% 구간 아래·위
# -out: error = 빈 목록이면 ZeroDivisionError
#------------------------------------------------------------------
def boot_ci(diffs: list[float], n_boot: int = 10000, seed: int = 7) -> tuple[float, float, float]:
    rnd = random.Random(seed)
    n = len(diffs)
    means = sorted(sum(diffs[rnd.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return sum(diffs) / n, means[int(0.025 * n_boot)], means[int(0.975 * n_boot) - 1]


#------------------------------------------------------------------
# 분위수
#=> 정렬한 값에서 p 위치(내림)의 값 (run_eval 과 같은 방식).
#
# -in: xs = 숫자 목록
# -in: p = 0~1
#
# -out: v = 분위수 (빈 목록이면 None)
# -out: error = 없음
#------------------------------------------------------------------
def quantile(xs: list[float], p: float):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else None


#------------------------------------------------------------------
# 점수 내기
#=> 실행마다 AC 평균(전체·유형별), 기준 실행과의 짝지은 차이·95% 구간, 거절·시간·고치기 횟수를 낸다.
#    1) 판정이 없는 답이 하나라도 있으면 그 수를 알리고 멈춘다 (빠진 채로 평균 내지 않는다)
#    2) 의미 유사도는 답·정답을 e5 로 임베딩한 코사인 (대칭 비교라 둘 다 "query: ")
#    3) 거절 비율은 답할 수 있는 90문항 / 문서에 없는 10문항을 따로 센다
#
# -in: runs = 실행 이름 목록
# -in: base = 기준 실행 이름
#
# -out: summary = 실행별 지표 dict (eval\runs\ac_summary.json 에도 씀)
# -out: error = 판정이 빠지면 SystemExit, e5 모델이 없으면 예외 전파
#------------------------------------------------------------------
def score(runs: list[str], base: str) -> dict:
    cache = load_cache()
    rows = collect(runs)
    missing = {k for *_, k in rows if k not in cache}
    if missing:
        sys.exit(f"판정 없는 답 {len(missing)}개 — 먼저 packets 를 만들고 판정하세요")
    sys.path.insert(0, str(PROTO / "src"))
    from vector_index import QUERY_PREFIX, OnnxEmbedder
    keys = sorted({k for *_, k in rows})
    emb = OnnxEmbedder(E5_DIR)
    a = emb.embed([k[1] for k in keys], QUERY_PREFIX)
    r = emb.embed([k[2] for k in keys], QUERY_PREFIX)
    sim = dict(zip(keys, (a * r).sum(1).tolist()))
    ac = {(name, it["id"]): W_FACT * fact_f1(cache[k]) + W_SIM * sim[k] for name, it, _, k in rows}
    items = {x["id"]: x for x in read_jsonl(HERE / "evalset_v2.jsonl")}
    out = {}
    for name in runs:
        recs = [x for x in read_jsonl(HERE / "runs" / f"ans_{name}.jsonl") if "error" not in x]
        ids = [x["id"] for x in recs if items[x["id"]]["answerable"]]
        s = {"n": len(ids), "ac": sum(ac[(name, i)] for i in ids) / len(ids)}
        for t in ("single", "multi", "reasoning"):
            ti = [i for i in ids if items[i]["type"] == t]
            s[f"ac_{t}"] = sum(ac[(name, i)] for i in ti) / len(ti) if ti else None
        ans = [x for x in recs if items[x["id"]]["answerable"]]
        una = [x for x in recs if not items[x["id"]]["answerable"]]
        s["refused_answerable"] = sum(x["refused"] for x in ans) / len(ans) if ans else None
        s["refused_unanswerable"] = sum(x["refused"] for x in una) / len(una) if una else None
        ts = [x["timing"]["answer"] for x in recs]
        s["answer_sec_mean"], s["answer_sec_p90"] = sum(ts) / len(ts), quantile(ts, 0.9)
        s["thinking_n"] = sum(bool(x.get("thinking")) for x in recs)
        s["fixes"] = {}
        for x in recs:
            for f in x.get("fixes", []):
                s["fixes"][f["fix"]] = s["fixes"].get(f["fix"], 0) + 1
        if name != base:
            common = [i for i in ids if (base, i) in ac]
            s["vs_base"] = dict(zip(("diff", "lo", "hi"), boot_ci([ac[(name, i)] - ac[(base, i)] for i in common])))
            s["vs_base"]["changed_answers"] = sum(
                1 for i in common if next(k for n, it, _, k in rows if n == name and it["id"] == i)
                != next(k for n, it, _, k in rows if n == base and it["id"] == i))
            for t in ("single", "multi", "reasoning"):
                ct = [i for i in common if items[i]["type"] == t]
                s["vs_base"][f"diff_{t}"] = sum(ac[(name, i)] - ac[(base, i)] for i in ct) / len(ct) if ct else None
        out[name] = s
    (HERE / "runs" / "ac_summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    # 문항별 점수도 남긴다 (어느 문항에서 오르내렸는지 보려고)
    (HERE / "runs" / "ac_items.json").write_text(json.dumps({f"{n}|{i}": round(v, 4) for (n, i), v in ac.items()},
                                                            ensure_ascii=False, indent=0), encoding="utf-8")
    for name, s in out.items():
        vb = s.get("vs_base")
        print(f"{name:8s} AC {s['ac']:.3f} (단순 {s['ac_single']:.3f} · 멀티 {s['ac_multi']:.3f} · 추론 {s['ac_reasoning']:.3f})"
              f" 오거절 {s['refused_answerable']:.3f} 무답거절 {s['refused_unanswerable']:.2f}"
              f" 답변 {s['answer_sec_mean']:.2f}s p90 {s['answer_sec_p90']:.2f}s 생각 {s['thinking_n']} 고치기 {s['fixes']}"
              + (f"\n         vs {base}: {vb['diff']:+.3f} [{vb['lo']:+.3f}, {vb['hi']:+.3f}] 바뀐 답 {vb['changed_answers']}"
                 if vb else ""))
    return out


#------------------------------------------------------------------
# 명령 실행
#=> packets / check / score 하위 명령.
#
# -in: 없음 (sys.argv)
#
# -out: 없음
# -out: error = check 에서 문제가 있으면 종료 코드 1
#------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["packets", "check", "score"])
    ap.add_argument("files", nargs="*")
    ap.add_argument("--runs", nargs="+", default=[])
    ap.add_argument("--base", default="base")
    ap.add_argument("--batches", type=int, default=4)
    a = ap.parse_args()
    if a.cmd == "check":
        sys.exit(0 if check_files(a.files[0], a.files[1]) else 1)
    if a.cmd == "packets":
        make_packets(a.runs, a.batches)
    else:
        score(a.runs, a.base)


if __name__ == "__main__":
    main()
