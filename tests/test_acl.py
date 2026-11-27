"""권한(설계서 12.4)·HTTP API 시험 — OpenSearch, struct_docparse.dll, decide, Qwen 서버 필요.

실제 색인은 건드리지 않는다: 임시 data 폴더와 시험용 색인(decide_acltest_*)을 쓴다. 문서는 실제 18개.
  권한 규칙: 37_* (김영란법·청탁금지법 사례집) → legal,  03.경조사지원규정* → hr
  1) 단위: acl_for, user_acl
  2) 권한 문항: 정답이 제한 문서에만 있는 평가 문항 28개
     - 권한 없는 사용자(공개 그룹만): 라운드의 새 청크·근거·라우팅 후보·라우팅 결과 어디에도 제한 문서가 없어야 함 (누출 0)
                                    + 답이 "찾을 수 없습니다" 인 비율
     - 권한 있는 사용자(legal, hr): 근거 포함률
  3) API: 토큰 없음 401, 본문의 groups 무시, 관리자 아님 403, /health, trace 에 사용자 ID
  python tests\\test_acl.py            (결과: eval\\runs\\acl_test.json)
"""
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

from decide_rag.acl import acl_for, user_acl  # noqa: E402
from decide_rag.agent import run_agentic  # noqa: E402
from decide_rag.cli import load_system  # noqa: E402
from decide_rag.config import load_config  # noqa: E402
from decide_rag.indexer import build, load_manifest  # noqa: E402
from run_eval import evidence_ratio, is_refusal, read_jsonl  # noqa: E402

RULES = '[[rule]]\npattern = "37_*"\ngroups = ["legal"]\n\n[[rule]]\npattern = "03.경조사지원규정*"\ngroups = ["hr"]\n'
TOKEN = "test-gateway-token"   # 이 시험 안에서만 쓰는 가짜 토큰


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
    print(("통과 " if ok else "실패 ") + msg, flush=True)
    assert ok, msg


#------------------------------------------------------------------
# 시험용 설정
#=> data·logs 를 임시 폴더로, 색인 이름을 decide_acltest 로, 권한 규칙을 RULES 로 바꾼다.
#
# -in: tmp = 임시 폴더
#
# -out: cfg = 설정 dict
# -out: error = config.toml 이 없으면 예외 전파
#------------------------------------------------------------------
def test_cfg(tmp: Path) -> dict:
    cfg = load_config()
    for k in ("data_dir", "log_dir"):
        cfg["paths"][k] = str(tmp / k)
        (tmp / k).mkdir()
    (tmp / "acl.toml").write_text(RULES, encoding="utf-8")
    cfg["index"] = {**cfg["index"], "acl_file": str(tmp / "acl.toml")}
    cfg["search"] = {**cfg["search"], "engine": "opensearch", "index_prefix": "decide_acltest"}
    cfg["service"] = {**cfg["service"], "dev_mode": False, "trace": True}
    return cfg


#------------------------------------------------------------------
# trace 에서 건드린 문서 ID 모으기
#=> 새로 본 청크, 근거, 라우팅 후보, 라우팅 결과에 나온 문서 ID 전부.
#
# -in: tr = run_agentic 결과
#
# -out: docs = 문서 ID 집합
# -out: error = 없음
#------------------------------------------------------------------
def touched_docs(tr: dict) -> set:
    docs = {c.split("#")[0] for c in tr["context"]}
    for r in tr["rounds"]:
        docs |= {c.split("#")[0] for c in r["new"]}
        info = r.get("plan_info") or {}
        docs |= set(info.get("candidate_ids") or []) | set(info.get("route") or {})
        for _, scope in r["queries"]:
            docs |= set(scope or [])
    return docs


#------------------------------------------------------------------
# 시험 본체
#=> 단위 → 색인 → 권한 문항(권한 없음·있음) → API 순서.
#
# -in: 없음
#
# -out: 없음 (결과를 eval\runs\acl_test.json 에 저장)
# -out: error = 누출·API 규칙 위반 시 AssertionError
#------------------------------------------------------------------
def main():
    sys.stdout.reconfigure(encoding="utf-8")
    # 1) 단위
    rules = [{"pattern": "37_*", "groups": ["legal"]}, {"pattern": "*q&a*", "groups": ["qa"]}]
    check(acl_for("37_청탁금지법_q&a_사례집[1].pdf", rules, ["all"]) == ["legal", "qa"], "여러 규칙은 그룹을 합친다")
    check(acl_for("출장여비규정.doc", rules, ["all"]) == ["all"], "맞는 규칙이 없으면 기본 그룹")
    check(user_acl(["hr", " ", ""], "all") == ["all", "hr"], "사용자 그룹 + 공개 그룹, 빈 값 제거")

    tmp = Path(tempfile.mkdtemp(prefix="decide_acl_"))
    eng = None
    try:
        cfg = test_cfg(tmp)
        s = build(cfg, full=True, log=lambda *_: None)
        m = load_manifest(cfg)
        restricted = {d for d, v in m.items() if v["acl"] != ["all"]}
        check(s["engine_docs"] == 18 and len(restricted) == 3, f"색인 18개, 제한 문서 3개 {s}")
        sysm = load_system(cfg)
        eng = sysm["index"]

        # 평가셋의 옛 문서 ID(D01…) → 제목 → 새 문서 ID
        legacy = {}
        for l in open(ROOT / "data" / "legacy_stage1" / "chunks.jsonl", encoding="utf-8"):
            c = json.loads(l)
            legacy[c["doc_id"]] = c["title"]
        by_title = {v["title"]: d for d, v in m.items()}
        items = [it for it in read_jsonl(ROOT / "eval" / "evalset_v2.jsonl")
                 if it["doc_ids"] and {by_title[legacy[x]] for x in it["doc_ids"]} <= restricted]
        print(f"권한 문항 {len(items)}개 (정답 문서가 모두 제한 문서)", flush=True)

        # 2) 권한 문항
        res = {"n": len(items), "no_access": [], "access": []}
        for it in items:
            for who, groups in (("no_access", user_acl([], "all")), ("access", user_acl(["legal", "hr"], "all"))):
                tr = run_agentic(it["question"], eng, sysm["judge"], sysm["llm"], cfg, it.get("history") or [],
                                 acl=groups)
                leak = sorted(touched_docs(tr) & restricted) if who == "no_access" else []
                res[who].append({"id": it["id"], "leak": leak, "refused": is_refusal(tr["answer"]),
                                 "evidence_ratio": evidence_ratio(it, eng.get_chunks(tr["context"])),
                                 "answer": tr["answer"][:200]})
            print(it["id"], "누출", res["no_access"][-1]["leak"], "거절", res["no_access"][-1]["refused"],
                  "| 권한 있음 근거", res["access"][-1]["evidence_ratio"], flush=True)
        leaks = sum(bool(r["leak"]) for r in res["no_access"])
        refused = sum(r["refused"] for r in res["no_access"])
        ev = [r["evidence_ratio"] for r in res["access"] if r["evidence_ratio"] is not None]
        res["summary"] = {"leak_questions": leaks, "no_access_refused": refused,
                          "access_evidence_ratio": round(sum(ev) / len(ev), 3) if ev else None,
                          "access_refused": sum(r["refused"] for r in res["access"])}
        print("요약", json.dumps(res["summary"], ensure_ascii=False), flush=True)
        check(leaks == 0, f"권한 없는 사용자에게 제한 문서 누출 0건 (문항 {len(items)}개)")

        # 3) API
        from fastapi.testclient import TestClient
        from decide_rag.service import create_app
        os.environ[cfg["service"]["gateway_token_env"]] = TOKEN
        c = TestClient(create_app(cfg))
        h = {"X-Gateway-Token": TOKEN, "X-User-Id": "u1", "X-User-Groups": ""}
        q = items[0]["question"]
        check(c.post("/ask", json={"question": q}).status_code == 401, "토큰 없으면 401")
        check(c.post("/ask", json={"question": q}, headers={**h, "X-Gateway-Token": "x"}).status_code == 401,
              "토큰 틀리면 401")
        r = c.post("/ask", json={"question": q, "groups": ["legal", "hr"], "debug": True}, headers=h)
        body = r.json()
        srcs = {x["chunk_id"].split("#")[0] for x in body.get("sources", [])}
        rounds_docs = touched_docs({"context": [], "rounds": body["trace"]["rounds"]}) if r.status_code == 200 else set()
        check(r.status_code == 200 and not (srcs | rounds_docs) & restricted, "본문의 groups 는 무시 (헤더 권한만 씀)")
        r = c.post("/ask", json={"question": q}, headers={**h, "X-User-Groups": "hr,legal"})
        check(r.status_code == 200 and bool({x["chunk_id"].split("#")[0] for x in r.json()["sources"]} & restricted),
              "헤더 그룹이 있으면 제한 문서를 근거로 씀")
        check(c.post("/index/rebuild", json={}, headers=h).status_code == 403, "관리자 아니면 색인 명령 403")
        hj = c.get("/health").json()
        check(hj["engine"]["docs"] == 18 and hj["auth"] == "gateway", f"/health {hj}")
        lines = [json.loads(l) for p in Path(cfg["paths"]["log_dir"]).glob("trace_*.jsonl")
                 for l in p.read_text(encoding="utf-8").splitlines()]
        check(len(lines) == 2 and all(x["user_id"] == "u1" for x in lines), "trace 에 질문마다 사용자 ID 기록")
        del os.environ[cfg["service"]["gateway_token_env"]]

        out = ROOT / "eval" / "runs" / "acl_test.json"
        out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        print("모두 통과 →", out)
    finally:
        if eng is not None:
            for name in (eng.chunks_index, eng.docs_index):
                eng.s.delete(f"{eng.url}/{name}")
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
