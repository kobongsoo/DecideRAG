# -*- coding: utf-8 -*-
"""사람 검수 평가셋 (6단계, 설계서 11장 "실제 사용자 질문 50문항 이상을 사람이 검수").

  python tools\\review_set.py export --out eval\\review\\review_sheet.csv     # trace 로그 → 검수표 (엑셀로 열림)
  python tools\\review_set.py import eval\\review\\review_sheet.csv          # 검수한 표 → eval\\evalset_human.jsonl
  python run.py eval --evalset eval\\evalset_human.jsonl --out eval\\runs\\human.jsonl

검수표 열 (앞 6개는 export 가 채우고, 뒤는 검수자가 채운다):
  번호, 시각, 질문, 이전대화, 시스템답, 시스템근거 | 채택(Y/N), 유형(single/multi/reasoning/unanswerable),
  정답, 근거인용문(여러 개면 줄바꿈), 일상어(Y/N), 메모
- 사용자 ID 는 검수표에 넣지 않는다 (검수에 필요 없음).
- 근거인용문은 문서 본문에서 그대로 복사한다. import 가 색인된 청크에서 찾아 chunk_id 를 붙인다. 못 찾으면 그 문항은 넣지 않고 알려 준다.
"""
import argparse
import csv
import glob
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8")

COLS = ["번호", "시각", "질문", "이전대화", "시스템답", "시스템근거",
        "채택(Y/N)", "유형", "정답", "근거인용문", "일상어(Y/N)", "메모"]
TYPES = {"single", "multi", "reasoning", "unanswerable"}


#------------------------------------------------------------------
# 비교용 정규화
#=> 공백·표 기호·따옴표를 지운다 (평가의 근거 포함률과 같은 규칙, eval\run_eval.py norm).
#
# -in: s = 문자열
#
# -out: norm = 정규화한 문자열
# -out: error = 없음
#------------------------------------------------------------------
def norm(s: str) -> str:
    return re.sub(r"[\s|\"'“”‘’·]", "", s or "")


#------------------------------------------------------------------
# trace 로그 → 검수표
#=> 실제 사용자의 질문만 골라 검수표 CSV 로 쓴다.
#    1) 부하 시험·CLI·시험 사용자(제외 접두)와 오류 줄은 뺀다
#    2) 같은 질문(이전 대화 포함)은 한 번만
#    3) 엑셀에서 한글이 깨지지 않게 UTF-8 BOM 으로 쓴다
#
# -in: log_dir = trace 로그 폴더
# -in: out = 검수표 경로
# -in: exclude = 뺄 사용자 ID 접두 목록
# -in: limit = 최대 문항 수 (0 = 전부)
#
# -out: n = 쓴 문항 수
# -out: error = 쓰기 실패 시 OSError 전파
#------------------------------------------------------------------
def export(log_dir: str, out: str, exclude: list[str], limit: int = 0) -> int:
    seen, rows = set(), []
    for f in sorted(glob.glob(str(Path(log_dir) / "trace_*.jsonl"))):
        for line in Path(f).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            uid = str(r.get("user_id") or "")
            if r.get("source") != "api" or r.get("error") or any(uid.startswith(p) for p in exclude):
                continue
            hist = r.get("history") or []
            key = (r["question"].strip(), json.dumps(hist, ensure_ascii=False))
            if key in seen:
                continue
            seen.add(key)
            rows.append({"번호": len(rows) + 1, "시각": r.get("time"), "질문": r["question"],
                         "이전대화": json.dumps(hist, ensure_ascii=False) if hist else "",
                         "시스템답": r.get("answer") or "", "시스템근거": "\n".join(r.get("context") or [])})
            if limit and len(rows) >= limit:
                break
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8-sig", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=COLS)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in COLS})
    return len(rows)


#------------------------------------------------------------------
# 검수표 → 평가셋
#=> 채택(Y) 줄만 평가셋 형식으로 바꾼다. 형식이 틀리거나 근거를 못 찾은 줄은 넣지 않고 이유를 알린다.
#    1) 유형은 single/multi/reasoning/unanswerable 중 하나
#    2) 답할 수 있는 문항은 정답과 근거인용문이 있어야 한다
#    3) 근거인용문은 색인된 청크 본문에서 찾아 chunk_id·doc_id 를 붙인다 (data\chunks)
#
# -in: sheet = 검수한 CSV
# -in: out = 평가셋 jsonl 경로
#
# -out: (n_ok, problems) = 넣은 문항 수, [(번호, 이유)]
# -out: error = 파일이 없으면 FileNotFoundError
#------------------------------------------------------------------
def import_sheet(sheet: str, out: str) -> tuple[int, list]:
    from decide_rag.config import load_config
    from decide_rag.chunker import load_chunks
    chunks = load_chunks(load_config())
    normed = [(c, norm(c["text"])) for c in chunks]
    items, problems = [], []
    with open(sheet, encoding="utf-8-sig", newline="") as fp:
        for r in csv.DictReader(fp):
            if (r.get("채택(Y/N)") or "").strip().upper() != "Y":
                continue
            no = r.get("번호")
            typ = (r.get("유형") or "").strip()
            if typ not in TYPES:
                problems.append((no, f"유형 '{typ}' 은 {sorted(TYPES)} 중 하나여야 함"))
                continue
            answerable = typ != "unanswerable"
            ref = (r.get("정답") or "").strip()
            quotes = [q.strip() for q in (r.get("근거인용문") or "").splitlines() if q.strip()]
            if answerable and (not ref or not quotes):
                problems.append((no, "답할 수 있는 문항은 정답과 근거인용문이 필요함"))
                continue
            ev, bad = [], []
            for q in quotes:
                nq = norm(q)
                hit = next((c for c, t in normed if nq and nq in t), None)
                if hit:
                    ev.append({"chunk_id": hit["chunk_id"], "quote": q})
                else:
                    bad.append(q[:30])
            if bad:
                problems.append((no, f"근거인용문을 색인에서 못 찾음: {bad}"))
                continue
            try:
                hist = json.loads(r["이전대화"]) if (r.get("이전대화") or "").strip() else []
            except json.JSONDecodeError:
                problems.append((no, "이전대화가 JSON 이 아님"))
                continue
            items.append({"id": f"H{len(items) + 1:03d}", "type": typ,
                          "colloquial": (r.get("일상어(Y/N)") or "").strip().upper() == "Y",
                          "history": hist, "question": r["질문"].strip(),
                          "reference": ref if answerable else "문서에 해당 내용이 없음",
                          "evidence": ev, "answerable": answerable,
                          "doc_ids": sorted({e["chunk_id"].split("#")[0] for e in ev}), "reviewed": True,
                          "source_no": no})
    Path(out).write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in items), encoding="utf-8")
    return len(items), problems


#------------------------------------------------------------------
# 명령 실행
#=> export / import 하위 명령.
#
# -in: 없음 (sys.argv)
#
# -out: 없음
# -out: error = import 에서 문제가 있으면 목록을 출력 (종료 코드 0, 넣을 수 있는 것은 넣는다)
#------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["export", "import"])
    ap.add_argument("sheet", nargs="?")
    ap.add_argument("--out")
    ap.add_argument("--log-dir", default=str(ROOT / "logs"))
    ap.add_argument("--exclude", default="load,dev,cli,test", help="뺄 사용자 ID 접두 (쉼표)")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    if a.cmd == "export":
        out = a.out or str(ROOT / "eval" / "review" / "review_sheet.csv")
        n = export(a.log_dir, out, [x for x in a.exclude.split(",") if x], a.limit)
        print(f"검수표 {n}문항 → {out}")
    else:
        if not a.sheet:
            ap.error("import 에는 검수표 경로가 필요합니다")
        out = a.out or str(ROOT / "eval" / "evalset_human.jsonl")
        n, probs = import_sheet(a.sheet, out)
        for no, why in probs:
            print(f"  번호 {no}: {why}")
        print(f"평가셋 {n}문항 → {out} (넣지 못한 줄 {len(probs)})")


if __name__ == "__main__":
    main()
