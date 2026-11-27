# -*- coding: utf-8 -*-
"""1단계: 텍스트 추출 (struct_docparse.dll) + 문서 ID·내용 해시. 5단계: PDF 숫자 빠짐 OCR 보정 (ocrfix)."""
import hashlib
import re
import sys
from pathlib import Path

# DLL 이 읽는 형식 중 이 데이터셋에 있는 것
SUPPORTED = {".doc", ".docx", ".pdf", ".pptx", ".ppt", ".hwp", ".hwpx", ".xlsx", ".txt", ".md"}
# 검색에 방해만 되는 블록 (머리말·꼬리말·쪽번호·목차)
SKIP_TYPES = {"header", "footer", "page_number", "toc"}
# 추출 방식 판. 바꾸면 내용이 같은 문서도 다음 build 때 다시 추출한다 (manifest 의 "extract" 와 비교)
#   1 = 0.2.0 DLL, 보정 없음 / 2 = 0.3.0 DLL + 숫자 빠짐 OCR 보정 (5단계 A6)
EXTRACT_VERSION = 2


#------------------------------------------------------------------
# 추출 방식 표시
#=> 판 번호와 보정 켬/끔을 묶은 문자열. manifest 에 문서마다 적어 두고, 다르면 그 문서를 다시 추출한다.
#
# -in: cfg = 설정 dict
#
# -out: sig = 예: "2:fix=1"
# -out: error = 없음
#------------------------------------------------------------------
def extract_signature(cfg: dict) -> str:
    return f"{EXTRACT_VERSION}:fix={int(bool(cfg['docparse'].get('fix_missing_digits')))}"


#------------------------------------------------------------------
# 파일 이름 → 문서 제목
#=> "03.경조사지원규정_12.01.01.doc" 처럼 앞 번호와 뒤 날짜가 붙은 이름에서 제목만 남긴다.
#    1) 앞의 "03." / "37_" 같은 번호 제거
#    2) 뒤의 "_12.01.01" 날짜, "[1]" 꼬리 제거
#    3) 남은 밑줄은 공백으로
#
# -in: path = 문서 경로
#
# -out: title = 사람이 읽는 제목 (예: "경조사지원규정")
# -out: error = 없음
#------------------------------------------------------------------
def title_of(path: Path) -> str:
    s = path.stem
    s = re.sub(r"^\d+[._]\s*", "", s)
    s = re.sub(r"\[\d+\]$", "", s)
    s = re.sub(r"_[\d.]+$", "", s)
    return s.replace("_", " ").strip()


#------------------------------------------------------------------
# 색인할 파일 목록
#=> 문서 폴더(하위 폴더 포함)에서 지원 형식 파일만 경로순으로 모은다. 임시 파일(~, .)은 뺀다.
#
# -in: doc_dir = 문서 폴더
#
# -out: files = [Path, …] 경로순
# -out: error = 폴더가 없으면 빈 목록
#------------------------------------------------------------------
def list_files(doc_dir: str) -> list[Path]:
    root = Path(doc_dir)
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*")
                  if p.is_file() and p.suffix.lower() in SUPPORTED and not p.name.startswith(("~", ".")))


#------------------------------------------------------------------
# 문서 폴더 기준 상대 경로 (ID 의 재료)
#=> 대소문자·구분자 차이로 ID 가 바뀌지 않도록 소문자 + '/' 로 맞춘다.
#
# -in: path = 파일 경로
# -in: doc_dir = 문서 폴더
#
# -out: rel = "하위폴더/파일명.확장자" (소문자)
# -out: error = 폴더 밖 경로면 ValueError
#------------------------------------------------------------------
def rel_key(path: Path, doc_dir: str) -> str:
    return path.resolve().relative_to(Path(doc_dir).resolve()).as_posix().lower()


#------------------------------------------------------------------
# 문서 ID 만들기 (경로 해시)
#=> 예전처럼 파일 순서로 D01, D02… 를 붙이면 문서가 하나 추가될 때 뒤 번호가 다 밀린다.
#   그래서 상대 경로의 SHA-1 앞 12자리로 정한다. 같은 경로면 언제나 같은 ID 다.
#
# -in: rel = rel_key() 결과
#
# -out: doc_id = "d" + 16진수 12자리 (예: "d3f9a0c1b2e4d")
# -out: error = 없음
#------------------------------------------------------------------
def doc_id_of(rel: str) -> str:
    return "d" + hashlib.sha1(rel.encode("utf-8")).hexdigest()[:12]


#------------------------------------------------------------------
# 파일 내용 해시
#=> 증분 색인에서 "바뀐 문서인가" 를 판단하는 기준. 수정 시각보다 정확하다.
#
# -in: path = 파일 경로
#
# -out: sha256 = 16진수 문자열
# -out: error = 읽기 실패 시 OSError 전파
#------------------------------------------------------------------
def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


#------------------------------------------------------------------
# 파서 열기
#=> 저장소의 ctypes 바인딩으로 struct_docparse.dll 을 연다. 다 쓰면 close() 해야 한다.
#
# -in: cfg = 설정 dict (docparse.dll, binding_dir, ocr_mode, ocr_threads)
#
# -out: parser = DocParser
# -out: error = DLL 을 못 열면 예외 전파
#------------------------------------------------------------------
def open_parser(cfg: dict):
    dp = cfg["docparse"]
    if dp["binding_dir"] not in sys.path:
        sys.path.insert(0, dp["binding_dir"])
    from struct_docparse import DocParser  # 저장소의 ctypes 바인딩
    return DocParser(dp["dll"], {"ocr": {"mode": dp.get("ocr_mode", "inline"), "threads": dp.get("ocr_threads", 4)}})


#------------------------------------------------------------------
# 문서 하나 파싱
#=> DLL 로 블록 목록을 받고, 머리말·쪽번호·목차·빈 블록을 뺀다.
#    PDF 이고 [docparse] fix_missing_digits 가 켜져 있으면 숫자 빠진 쪽만 OCR 해 블록에 숫자를 채운다 (ocrfix)
#    OCR 결과는 data\ocr\<내용 해시>.json 에 두고 다시 쓴다
#
# -in: parser = open_parser() 결과
# -in: path = 파일 경로
# -in: doc_id = 문서 ID
# -in: cfg = 설정 dict (없으면 보정 안 함)
# -in: sha = 파일 내용 해시 (OCR 캐시 파일 이름, 없으면 캐시 안 씀)
# -in: log = 진행 출력 함수 (OCR 은 수 분 걸릴 수 있어 시작을 알린다, 기본 None)
#
# -out: rec = {"doc_id","title","path","format","blocks"(+ "ocr_fix")} (실패하면 "error" 키, blocks 없음)
# -out: error = 예외를 밖으로 내지 않고 rec["error"] 에 기록. OCR 보정 실패는 rec["ocr_fix"]["error"] 에 남기고 보정 없이 진행
#------------------------------------------------------------------
def parse_file(parser, path: Path, doc_id: str, cfg: dict | None = None, sha: str | None = None,
               log=None) -> dict:
    rec = {"doc_id": doc_id, "title": title_of(path), "path": str(path)}
    try:
        d = parser.parse(str(path))
        # 본문이 아닌 블록과 빈 블록은 검색 품질만 떨어뜨리므로 버린다
        rec["blocks"] = [b for b in d.get("blocks", [])
                         if b.get("type") not in SKIP_TYPES and (b.get("text") or "").strip()]
        rec["format"] = d.get("format")
    except Exception as e:  # 문서 하나 실패는 기록만 하고 계속
        rec["error"] = f"{type(e).__name__}: {e}"
        return rec
    if cfg and cfg["docparse"].get("fix_missing_digits") and rec["format"] == "pdf":
        from .ocrfix import fix_blocks, missing_pages, ocr_rows
        pages = missing_pages(rec["blocks"])
        if pages:
            cache = Path(cfg["paths"]["data_dir"]) / "ocr" / f"{sha}.json" if sha else None
            if log:
                log(f"  숫자 빠진 쪽 {len(pages)}쪽 OCR 보정 (캐시 없는 쪽은 쪽당 수 초) {path.name}")
            try:
                rows, info = ocr_rows(parser, str(path), pages, cache)
                rec["blocks"], stats = fix_blocks(rec["blocks"], rows)
                rec["ocr_fix"] = {"pages": len(pages), **info, **stats}
            except Exception as e:  # OCR 이 안 돼도 글자층 글로는 색인한다
                rec["ocr_fix"] = {"pages": len(pages), "error": f"{type(e).__name__}: {e}"}
    return rec
