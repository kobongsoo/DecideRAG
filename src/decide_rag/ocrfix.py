# -*- coding: utf-8 -*-
"""PDF 숫자 빠짐 보정 (5단계 A6, 설계서 4.1·10장).

어떤 PDF 는 글자층에서 숫자가 빠져 있다 ("제8조제1항" → "제 조제 항", "1인당 3만원" → "인당 만원").
DLL 은 글자층이 있는 쪽을 OCR 하지 않으므로, 숫자가 빠진 쪽만 골라 DLL 의 쪽 OCR(ocr_file)로 읽고,
글자층 블록과 OCR 글을 글자(한글·영문) 단위로 맞춘 뒤 숫자가 있는 틈만 OCR 글로 바꾼다.
띄어쓰기·문장은 글자층 것을 그대로 둔다.
"""
import difflib
import json
import re
from pathlib import Path

# 숫자가 빠진 흔적: "제 조", "조제 항", "회 만원", " 만원 " (앞에 숫자가 있어야 할 자리)
MISSING_RE = re.compile(r"제\s+조|조제\s+항|\s회\s+만원|\s+만원\s|제\s*조\s*제\s*항")
LETTER_RE = re.compile(r"[가-힣A-Za-z]")
DIGIT_RE = re.compile(r"\d")


#------------------------------------------------------------------
# 숫자가 빠진 쪽 찾기
#=> 블록 글에 숫자 빠짐 흔적이 있는 쪽 번호를 모은다. 이 쪽들만 OCR 한다 (OCR 은 쪽당 수 초라 비싸다).
#
# -in: blocks = 파싱 블록 목록 ({"type","text","page"})
#
# -out: pages = 쪽 번호 목록 (1부터, 오름차순)
# -out: error = 없음
#------------------------------------------------------------------
def missing_pages(blocks: list[dict]) -> list[int]:
    return sorted({b["page"] for b in blocks if b.get("page") and MISSING_RE.search(b.get("text") or "")})


#------------------------------------------------------------------
# 쪽 OCR (캐시 사용)
#=> 고른 쪽만 OCR 해서 쪽마다 줄 글 목록을 돌려준다. 결과는 파일 내용 해시별 캐시 파일에 둔다.
#    1) 캐시에 있는 쪽은 다시 OCR 하지 않는다 (색인을 다시 만들어도 10분씩 들지 않게)
#    2) OCR 엔진 이름이 다르면 캐시를 버린다 (모델이 바뀌면 글자가 달라질 수 있다)
#
# -in: parser = DocParser (0.3.0 의 ocr_file 사용)
# -in: path = PDF 경로
# -in: pages = OCR 할 쪽 번호 목록
# -in: cache_file = 캐시 파일 경로 (None 이면 캐시 안 씀)
#
# -out: (rows, info) = {쪽: [줄 글, …]}, {"ocr_pages": 새로 OCR 한 쪽 수, "cached": 캐시에서 쓴 쪽 수, "engine"}
# -out: error = OCR 실패 시 DocParseError 전파
#------------------------------------------------------------------
def ocr_rows(parser, path: str, pages: list[int], cache_file: Path | None) -> tuple[dict, dict]:
    cache = {}
    if cache_file and cache_file.exists():
        cache = json.loads(cache_file.read_text(encoding="utf-8"))
    engine = parser.version()
    got = cache.get("pages", {}) if cache.get("engine") == engine else {}
    todo = [p for p in pages if str(p) not in got]
    if todo:
        r = parser.ocr_file(path, todo)
        for p, v in r["pages"].items():
            # 줄 상자 중 글만 남긴다
            got[str(p)] = [x.get("t", "") if isinstance(x, dict) else str(x) for x in v.get("rows", [])]
        if cache_file:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            tmp = cache_file.with_suffix(".tmp")
            tmp.write_text(json.dumps({"engine": engine, "pages": got}, ensure_ascii=False), encoding="utf-8")
            tmp.replace(cache_file)
    return {int(p): got[str(p)] for p in pages if str(p) in got}, {
        "ocr_pages": len(todo), "cached": len(pages) - len(todo), "engine": engine}


#------------------------------------------------------------------
# 블록 하나에 숫자 채우기
#=> 글자층 글과 OCR 글을 한글·영문 글자만 남겨 맞춘 뒤, 맞은 글자 사이 틈에 OCR 쪽 숫자가 있으면 그 틈을 OCR 글로 바꾼다.
#    1) difflib 로 글자를 맞춘다 (OCR 이 글자 하나를 잘못 읽어도 나머지는 맞는다)
#    2) 맞은 글자가 글자층 글자의 90% 미만이거나, OCR 쪽 범위가 너무 넓게 흩어지면 바꾸지 않는다
#    3) 이웃한 두 글자가 OCR 에서도 바로 이웃(사이에 다른 글자 없음)이고 그 틈에 숫자가 있을 때만 틈을 바꾼다
#    4) 첫 글자 앞 틈(같은 줄 안)에 숫자가 있으면 앞에 붙인다 (예: "인당" → "1인당")
#
# -in: text = 글자층 블록 글
# -in: ocr = 그 쪽(과 다음 쪽) OCR 글 (줄은 "\n")
#
# -out: new = 숫자를 채운 글 (바꿀 것이 없거나 못 맞추면 None)
# -out: error = 없음
#------------------------------------------------------------------
def fill_digits(text: str, ocr: str) -> str | None:
    tl = [i for i, ch in enumerate(text) if LETTER_RE.match(ch)]
    ol = [i for i, ch in enumerate(ocr) if LETTER_RE.match(ch)]
    if len(tl) < 6 or not ol:
        return None
    nt = "".join(text[i] for i in tl)
    no = "".join(ocr[i] for i in ol)
    # 빠른 길: 그대로 들어 있으면 그 자리, 아니면 difflib 로 글자 짝을 찾는다
    k = no.find(nt)
    if k >= 0:
        pairs = {i: k + i for i in range(len(nt))}
    else:
        sm = difflib.SequenceMatcher(None, nt, no, autojunk=False)
        pairs = {}
        for a, b, n in sm.get_matching_blocks():
            for d in range(n):
                pairs[a + d] = b + d
        if len(pairs) < 0.9 * len(nt):
            return None
        span = max(pairs.values()) - min(pairs.values()) + 1
        # 맞은 글자가 OCR 쪽 여기저기에 흩어져 있으면 엉뚱한 곳과 맞춘 것이다
        if span > 1.3 * len(nt):
            return None
    out, changed = [], False
    # 첫 글자 앞: 글자층의 앞부분은 그대로, OCR 같은 줄의 숫자 머리를 붙인다
    out.append(text[:tl[0]])
    if 0 in pairs:
        j = ol[pairs[0]]
        prev = ol[pairs[0] - 1] + 1 if pairs[0] > 0 else 0
        head = ocr[prev:j].split("\n")[-1].lstrip()
        if DIGIT_RE.search(head):
            out.append(head)
            changed = True
    out.append(text[tl[0]])
    for i in range(1, len(tl)):
        gap_t = text[tl[i - 1] + 1:tl[i]]
        a, b = pairs.get(i - 1), pairs.get(i)
        if a is not None and b == a + 1:
            gap_o = ocr[ol[a] + 1:ol[b]]
            if DIGIT_RE.search(gap_o):
                # 줄바꿈은 공백으로 (블록 안에서는 한 문단)
                out.append(gap_o.replace("\n", " "))
                changed = True
            else:
                out.append(gap_t)
        else:
            out.append(gap_t)
        out.append(text[tl[i]])
    out.append(text[tl[-1] + 1:])
    return "".join(out) if changed else None


#------------------------------------------------------------------
# 문서 블록 보정
#=> 숫자 빠진 블록마다 그 쪽(+ 다음 쪽, 블록이 쪽을 넘어갈 때) OCR 글로 fill_digits 를 한다.
#   표 블록은 행(rows) 구조가 따로 있어 건드리지 않는다.
#
# -in: blocks = 파싱 블록 목록 (바꾼 사본을 돌려준다, 원본은 그대로)
# -in: rows = ocr_rows() 의 {쪽: [줄 글]}
#
# -out: (blocks, stats) = 보정한 블록 목록, {"missing": 숫자 빠진 블록 수, "fixed": 채운 수}
# -out: error = 없음
#------------------------------------------------------------------
def fix_blocks(blocks: list[dict], rows: dict) -> tuple[list[dict], dict]:
    out, miss, fixed = [], 0, 0
    for b in blocks:
        t = b.get("text") or ""
        if b.get("type") != "table" and b.get("page") in rows and MISSING_RE.search(t):
            miss += 1
            ocr = "\n".join(rows[b["page"]] + rows.get(b["page"] + 1, []))
            new = fill_digits(t, ocr)
            if new:
                b = {**b, "text": new, "source": "ocr_fix"}
                fixed += 1
        out.append(b)
    return out, {"missing": miss, "fixed": fixed}
