"""증분 색인 통합 시험 (OpenSearch + struct_docparse.dll 필요).

실제 문서 폴더·색인은 건드리지 않는다: 임시 폴더와 시험용 색인(decide_test_*)을 쓴다.
  python tests\\test_incremental.py
"""
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from decide_rag.config import load_config  # noqa: E402
from decide_rag.extract import doc_id_of  # noqa: E402
from decide_rag.indexer import build, load_manifest  # noqa: E402
from decide_rag.search_engine import OpenSearchEngine  # noqa: E402


#------------------------------------------------------------------
# 시험용 설정 만들기
#=> 문서 폴더·data 폴더를 임시 폴더로, 색인 이름을 decide_test 로 바꾼다.
#
# -in: tmp = 임시 폴더
#
# -out: cfg = 설정 dict
# -out: error = config.toml 이 없으면 예외 전파
#------------------------------------------------------------------
def test_cfg(tmp: Path) -> dict:
    cfg = load_config()
    cfg["paths"]["doc_dir"] = str(tmp / "docs")
    cfg["paths"]["data_dir"] = str(tmp / "data")
    (tmp / "docs").mkdir()
    (tmp / "data").mkdir()
    cfg["search"] = {**cfg["search"], "engine": "opensearch", "index_prefix": "decide_test"}
    return cfg


#------------------------------------------------------------------
# 조건 확인
#=> 틀리면 무엇이 틀렸는지 보여 주고 AssertionError.
#
# -in: ok = 조건
# -in: msg = 설명
#
# -out: 없음
# -out: error = 조건이 거짓이면 AssertionError
#------------------------------------------------------------------
def check(ok: bool, msg: str):
    print(("통과 " if ok else "실패 ") + msg)
    assert ok, msg


#------------------------------------------------------------------
# 시험 본체
#=> 추가 → 그대로 → 내용 변경 → 삭제 → 이름 변경 → 엔진 비움 후 재적재 순서로 확인한다.
#
# -in: 없음
#
# -out: 없음
# -out: error = 하나라도 틀리면 AssertionError
#------------------------------------------------------------------
def main():
    sys.stdout.reconfigure(encoding="utf-8")
    tmp = Path(tempfile.mkdtemp(prefix="decide_inc_"))
    quiet = lambda *_: None  # noqa: E731  (진행 출력 끔)
    try:
        cfg = test_cfg(tmp)
        eng = OpenSearchEngine(cfg["search"])
        docs = tmp / "docs"
        (docs / "a.md").write_text("# 휴가 규정\n연차 휴가는 15일이다.\n", encoding="utf-8")
        (docs / "b.md").write_text("# 출장 규정\n국내 출장 일비는 2만원이다.\n", encoding="utf-8")
        (docs / "sub").mkdir()
        (docs / "sub" / "c.md").write_text("# 경조 규정\n본인 결혼 시 50만원.\n", encoding="utf-8")

        s = build(cfg, full=True, log=quiet)
        check(s["new"] == 3 and s["engine_docs"] == 3, f"처음 색인 3개 {s}")
        ida = doc_id_of("a.md")
        check(ida in load_manifest(cfg), "문서 ID = 상대 경로 해시")

        s = build(cfg, log=quiet)
        check(s["same"] == 3 and s["new"] + s["changed"] + s["deleted"] == 0, f"다시 돌리면 할 일 없음 {s}")

        (docs / "a.md").write_text("# 휴가 규정\n연차 휴가는 20일이다.\n", encoding="utf-8")
        s = build(cfg, log=quiet)
        hits = eng.search("연차 휴가 며칠", 3)
        check(s["changed"] == 1 and s["same"] == 2, f"내용 바뀐 문서만 다시 {s}")
        check(bool(hits) and "20일" in hits[0][0]["body"] and all("15일" not in h[0]["body"] for h in hits),
              "바뀐 내용으로 검색되고 옛 청크는 없음")

        (docs / "b.md").unlink()
        s = build(cfg, log=quiet)
        # "일비" 는 Kiwi 에서 "일" 토큰을 만들어 다른 문서("20일")와도 맞으므로, 지운 문서의 본문이 없는지로 본다
        gone = all("출장 일비" not in h[0]["body"] for h in eng.search("출장 일비", 10))
        check(s["deleted"] == 1 and s["engine_docs"] == 2 and gone, f"지운 문서는 엔진에서도 사라짐 {s}")

        (docs / "sub" / "c.md").rename(docs / "sub" / "c2.md")
        s = build(cfg, log=quiet)
        check(s["new"] == 1 and s["deleted"] == 1 and s["engine_docs"] == 2, f"이름 변경 = 삭제 + 추가 {s}")

        eng.ensure_indices(recreate=True)
        s = build(cfg, log=quiet)
        check(s["reloaded"] == 2 and s["engine_docs"] == 2, f"엔진이 비면 청크 파일로 다시 넣음 {s}")

        check(bool(eng.search_docs("결혼 경조", 5)), "문서 색인(라우팅 후보) 검색")
        check(bool(eng.prf_terms([eng.search("결혼", 1)[0][0]["chunk_id"]], "결혼", 3)), "PRF 확장어")
        print("모두 통과")
    finally:
        try:
            for name in ("decide_test_chunks", "decide_test_docs"):
                eng.s.delete(f"{eng.url}/{name}")
        except Exception:
            pass
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
