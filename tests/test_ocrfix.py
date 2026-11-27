# -*- coding: utf-8 -*-
"""PDF 숫자 빠짐 보정 시험 (5단계 A6).

  python tests\\test_ocrfix.py           # 글자 맞추기 단위 시험 + 실제 PDF 2쪽 OCR (DLL 0.3.0 필요, 캐시는 임시 폴더)
"""
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.stdout.reconfigure(encoding="utf-8")

from decide_rag.config import load_config  # noqa: E402
from decide_rag.extract import open_parser  # noqa: E402
from decide_rag.ocrfix import fill_digits, fix_blocks, missing_pages, ocr_rows  # noqa: E402

PDF = r"D:\Project\Q&A\dataset\doc\37_청탁금지법_Q&A_사례집[1].pdf"


#------------------------------------------------------------------
# 전체 시험
#=> 숫자 채우기 규칙을 작은 예로 확인하고, 실제 PDF 의 숫자 빠진 쪽 2쪽을 OCR 해 보정까지 해 본다.
#
# -in: 없음
#
# -out: 없음 (실패가 있으면 종료 코드 1)
# -out: error = DLL·PDF 가 없으면 예외 전파
#------------------------------------------------------------------
def main():
    ok = []

    # 결과 한 줄 기록
    def check(name, cond):
        ok.append(cond)
        print(("통과" if cond else "실패"), name)

    ocr = "청탁금지법에 따르면,누구에게든 1인당 3만원 이하의 식사만 살 수 있는\n그렇지 않습니다.1회 100만원 이하"
    check("앞 숫자·가운데 숫자 채우기, 글자층 띄어쓰기 유지",
          fill_digits("청탁금지법에 따르면 누구에게든 인당 만원 이하의 식사만 살 수 있는", ocr)
          == "청탁금지법에 따르면 누구에게든 1인당 3만원 이하의 식사만 살 수 있는")
    check("숫자 없는 틈은 글자층 그대로", fill_digits("누구에게든 식사만 살 수 있는", ocr) is None)
    check("OCR 이 글자 하나 잘못 읽어도 나머지로 맞춤",
          fill_digits("그 공직자등의 배우자 이므로 제 조제 항 제 항 참조",
                      "그 공직자등의 배우지'이므로(제8조제1항,제2항 참조),") == "그 공직자등의 배우자 이므로 제8조제1항 제2항 참조")
    check("엉뚱한 곳(글자 흩어짐)은 바꾸지 않음",
          fill_digits("회 만원 이하 금품등의 경우 직무", "금 1 품 2 등 3 의 경우 직 무 회 만 원 이 하") is None)
    check("짧은 글(6글자 미만)은 맞추지 않음", fill_digits("제 조", "제8조") is None)
    blocks = [{"type": "paragraph", "text": "인당 만원 이하의 식사만 살 수 있는", "page": 3},
              {"type": "table", "text": "| 회 만원 |", "page": 3},
              {"type": "paragraph", "text": "보통 문장", "page": 4}]
    check("숫자 빠진 쪽 고르기 (표 포함)", missing_pages(blocks) == [3])
    nb, st = fix_blocks(blocks, {3: ["누구에게든 1인당 3만원 이하의 식사만 살 수 있는"]})
    check("표 블록은 건드리지 않음", nb[1]["text"] == "| 회 만원 |" and st == {"missing": 1, "fixed": 1})

    # 실제 OCR: 숫자 빠진 쪽 중 2쪽 (9, 12)
    cfg = load_config()
    p = open_parser(cfg)
    try:
        d = p.parse(PDF)
        bl = [b for b in d["blocks"] if (b.get("text") or "").strip()]
        pages = [x for x in missing_pages(bl) if x in (9, 12)]
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "c.json"
            t = time.perf_counter()
            rows, info = ocr_rows(p, PDF, pages, cache)
            dt = time.perf_counter() - t
            check(f"실제 OCR 2쪽 ({dt:.1f}초)", info["ocr_pages"] == 2 and set(rows) == {9, 12})
            rows2, info2 = ocr_rows(p, PDF, pages, cache)
            check("두 번째는 캐시에서 (OCR 0쪽)", info2["ocr_pages"] == 0 and rows2 == rows)
        nb, st = fix_blocks([b for b in bl if b.get("page") in (9, 12)], rows)
        print("   보정", st)
        check("실제 쪽 보정: 숫자 빠진 블록의 절반 이상을 채움", st["fixed"] >= st["missing"] / 2)
        check("'1인당 3만원' 복구", any("1인당 3만원" in b["text"] for b in nb))
    finally:
        p.close()
    print(f"{sum(ok)}/{len(ok)} 통과")
    sys.exit(0 if all(ok) else 1)


if __name__ == "__main__":
    main()
