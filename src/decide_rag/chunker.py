# -*- coding: utf-8 -*-
"""2단계: 헤딩 기준 청킹."""
import json
import re
from pathlib import Path

# 청탁금지법 사례집처럼 "사례 15 …" 로 시작하는 줄은 새 단락(헤딩)으로 본다
CASE_RE = re.compile(r"^\s*(사례|Q\s*\d|질의)\s*\d*")


#------------------------------------------------------------------
# 긴 표를 행 단위로 나누기
#=> 표 하나가 max_chars 보다 길면 첫 행(머리글)을 조각마다 반복해 붙여 나눈다.
#   머리글이 있어야 "휴가(일) 5" 같은 숫자의 뜻이 조각 안에서도 남는다.
#
# -in: text = 마크다운 표 텍스트 ("| a | b |" 줄들)
# -in: max_chars = 조각 최대 글자 수
#
# -out: parts = 표 조각 문자열 목록
# -out: error = 없음
#------------------------------------------------------------------
def split_table(text: str, max_chars: int) -> list[str]:
    lines = [l for l in text.splitlines() if l.strip()]
    if len(text) <= max_chars or len(lines) < 3:
        return [text]
    head, parts, cur = lines[0], [], []
    for l in lines[1:]:
        if cur and len(head) + sum(len(x) + 1 for x in cur) + len(l) > max_chars:
            parts.append("\n".join([head] + cur))
            cur = []
        cur.append(l)
    if cur:
        parts.append("\n".join([head] + cur))
    return parts


#------------------------------------------------------------------
# 긴 문단을 문장 경계로 나누기
#=> 한 블록이 max_chars 보다 길면 마침표·줄바꿈 근처에서 잘라 여러 조각으로 만든다.
#
# -in: text = 블록 텍스트
# -in: max_chars = 조각 최대 글자 수
#
# -out: parts = 조각 목록
# -out: error = 없음
#------------------------------------------------------------------
def split_long(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    sents = re.split(r"(?<=[.다요음함\n])\s+", text)
    parts, cur = [], ""
    for s in sents:
        if cur and len(cur) + len(s) + 1 > max_chars:
            parts.append(cur)
            cur = ""
        cur = (cur + " " + s).strip()
        # 문장 하나가 너무 길면 글자 수로 강제로 자른다
        while len(cur) > max_chars:
            parts.append(cur[:max_chars])
            cur = cur[max_chars:]
    if cur:
        parts.append(cur)
    return parts


#------------------------------------------------------------------
# 문서 하나를 청크로 나누기
#=> 블록을 차례로 이어 붙이다가 헤딩을 만나거나 길이가 넘치면 끊는다.
#    1) 헤딩(level)으로 "제목 > 장 > 조" 경로를 관리한다 → 청크 앞에 붙여 BM25 가 제목 단어로도 찾게 함
#    2) 본문이 min_chars 보다 짧으면 헤딩에서 끊지 않고 이어 붙인다 (짧은 조항 여러 개를 한 청크로)
#    3) max_chars 를 넘기면 끊고, 직전 블록이 짧으면 다음 청크 앞에 겹쳐 넣는다 (문맥 끊김 완화)
#    4) 긴 표·문단은 split_table / split_long 으로 먼저 나눈다
#
# -in: doc = {"doc_id","title","blocks"}
# -in: max_chars = 청크 본문 최대 글자 수
# -in: min_chars = 헤딩에서 끊기 위한 최소 본문 길이
#
# -out: chunks = [{"chunk_id","doc_id","title","path","page","body","text"}, …]
# -out: error = 없음 (블록이 없으면 빈 목록)
#------------------------------------------------------------------
def chunk_doc(doc: dict, max_chars: int = 700, min_chars: int = 150) -> list[dict]:
    chunks: list[dict] = []
    path: list[tuple[int, str]] = []      # (level, heading) 스택
    cur: list[str] = []
    cur_path: list[str] = []
    cur_page = None

    # 현재까지 모은 줄을 청크 하나로 내보낸다
    def flush():
        nonlocal cur, cur_page
        body = "\n".join(cur).strip()
        if body:
            prefix = " > ".join([doc["title"]] + cur_path)
            chunks.append({"chunk_id": f"{doc['doc_id']}#{len(chunks):03d}", "doc_id": doc["doc_id"],
                           "title": doc["title"], "path": list(cur_path), "page": cur_page,
                           "body": body, "text": f"[{prefix}]\n{body}"})
        cur, cur_page = [], None

    for b in doc.get("blocks", []):
        text = re.sub(r"[ \t]+", " ", b["text"]).strip()
        is_head = b["type"] == "heading" or (b["type"] == "paragraph" and CASE_RE.match(text) and len(text) < 60)
        if is_head:
            level = b.get("level") or 3
            # 본문이 충분히 쌓였을 때만 헤딩에서 끊는다
            if sum(len(x) for x in cur) >= min_chars:
                flush()
            while path and path[-1][0] >= level:
                path.pop()
            path.append((level, text[:40]))
            if not cur:
                cur_path = [h for _, h in path]
            cur.append(text)
            cur_page = cur_page or b.get("page")
            continue
        pieces = split_table(text, max_chars) if b["type"] == "table" else split_long(text, max_chars)
        for piece in pieces:
            if cur and sum(len(x) + 1 for x in cur) + len(piece) > max_chars:
                last = cur[-1]
                flush()
                # 짧은 직전 줄은 다음 청크에도 넣어 문맥을 이어 준다
                if len(last) < 200 and len(last) + len(piece) <= max_chars:
                    cur.append(last)
                cur_path = [h for _, h in path]
            if not cur:
                cur_path = [h for _, h in path]
            cur.append(piece)
            cur_page = cur_page or b.get("page")
    flush()
    return chunks


#------------------------------------------------------------------
# 문서별 청크 파일 경로
#=> 증분 색인을 위해 청크를 문서마다 따로 저장한다: data\chunks\<문서ID>.jsonl
#
# -in: cfg = 설정 dict
# -in: doc_id = 문서 ID
#
# -out: path = 파일 경로
# -out: error = 없음
#------------------------------------------------------------------
def chunk_file(cfg: dict, doc_id: str) -> Path:
    return Path(cfg["paths"]["data_dir"]) / "chunks" / f"{doc_id}.jsonl"


#------------------------------------------------------------------
# 문서 하나 청킹 후 저장
#=> chunk_doc 결과를 data\chunks\<문서ID>.jsonl 로 덮어쓴다.
#
# -in: doc = parse_file 결과 (오류 문서면 빈 목록 저장)
# -in: cfg = 설정 dict (chunk.max_chars, chunk.min_chars)
#
# -out: chunks = 이 문서의 청크 목록
# -out: error = 쓰기 실패 시 OSError 전파
#------------------------------------------------------------------
def chunk_and_save(doc: dict, cfg: dict) -> list[dict]:
    c = cfg["chunk"]
    chunks = [] if "error" in doc else chunk_doc(doc, c["max_chars"], c["min_chars"])
    p = chunk_file(cfg, doc["doc_id"])
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in chunks), encoding="utf-8")
    return chunks


#------------------------------------------------------------------
# 저장된 청크 모두 읽기
#=> data\chunks\*.jsonl 을 문서 ID 순으로 읽는다 (메모리 엔진·평가용).
#   운영 규모(100만 청크)에서는 OpenSearch 가 청크를 갖고 있으므로 이 함수를 쓰지 않는다.
#
# -in: cfg = 설정 dict
#
# -out: chunks = 청크 목록
# -out: error = 폴더가 없으면 빈 목록
#------------------------------------------------------------------
def load_chunks(cfg: dict) -> list[dict]:
    d = Path(cfg["paths"]["data_dir"]) / "chunks"
    out = []
    for p in sorted(d.glob("*.jsonl")):
        out.extend(json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip())
    return out
