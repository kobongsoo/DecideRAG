# -*- coding: utf-8 -*-
"""증분 색인 (설계서 12.3, 14장 2단계).

data\\manifest.json 에 문서마다 {상대경로, 내용 해시, 제목, 청크 수, 오류} 를 둔다.
build 를 다시 돌리면
  - 새 파일·내용이 바뀐 파일만 파싱 → 청킹 → 엔진에 교체
  - 사라진 파일은 엔진·청크 파일·파싱 결과에서 지움
  - 그대로인 파일은 건드리지 않음
  - 내용은 같은데 권한 규칙(acl.toml) 결과가 바뀐 파일은 파싱 없이 acl 필드만 갱신
엔진(OpenSearch)이 비어 있는데 manifest 가 있으면(엔진을 새로 깔았을 때) 저장된 청크 파일로 다시 넣는다 (파싱 생략).
"""
import json
import time
from pathlib import Path

from .acl import acl_for, load_rules
from .chunker import chunk_and_save, chunk_file
from .extract import doc_id_of, extract_signature, file_sha256, list_files, open_parser, parse_file, rel_key


#------------------------------------------------------------------
# manifest 경로
#=> data\manifest.json
#
# -in: cfg = 설정 dict
#
# -out: path = 파일 경로
# -out: error = 없음
#------------------------------------------------------------------
def manifest_path(cfg: dict) -> Path:
    return Path(cfg["paths"]["data_dir"]) / "manifest.json"


#------------------------------------------------------------------
# manifest 읽기
#=> 없으면 빈 dict (처음 색인).
#
# -in: cfg = 설정 dict
#
# -out: manifest = {doc_id: {"rel","path","sha256","title","n_chunks","error","indexed_at","extract"(추출 방식),"ocr_fix"(숫자 보정 결과)}}
# -out: error = JSON 이 깨졌으면 예외 전파
#------------------------------------------------------------------
def load_manifest(cfg: dict) -> dict:
    p = manifest_path(cfg)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


#------------------------------------------------------------------
# manifest 저장
#=> 중간에 끊겨도 파일이 깨지지 않도록 임시 파일에 쓴 뒤 바꿔 끼운다.
#
# -in: cfg = 설정 dict
# -in: m = manifest dict
#
# -out: 없음
# -out: error = 쓰기 실패 시 OSError 전파
#------------------------------------------------------------------
def save_manifest(cfg: dict, m: dict):
    p = manifest_path(cfg)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(m, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(p)


#------------------------------------------------------------------
# 임베딩 표시
#=> 벡터를 쓰면 임베딩 모델 이름, 아니면 None. manifest 에 문서마다 적어 두고 바뀌면 다시 색인한다 (7단계).
#
# -in: cfg = 설정 dict
#
# -out: sig = 모델 이름 또는 None
# -out: error = 없음
#------------------------------------------------------------------
def embed_signature(cfg: dict):
    v = cfg.get("vector", {})
    return v.get("embed_model", "bge-m3-ko") if v.get("enabled") else None


#------------------------------------------------------------------
# 할 일 나누기
#=> 지금 폴더의 파일과 manifest 를 비교해 새로/바뀜/권한만 바뀜/그대로/사라짐 으로 나눈다.
#    1) 문서 ID = 상대 경로 해시 → 파일을 옮기거나 이름을 바꾸면 "사라짐 + 새로" 가 된다
#    2) 같은 ID 인데 내용 해시가 다르거나 추출 방식(extract_signature)·임베딩 모델(embed_signature)이 바뀌었으면 "바뀜"
#    3) 내용은 같고 권한 규칙 결과(acl)만 다르면 "권한만 바뀜"
#
# -in: cfg = 설정 dict
# -in: m = manifest
# -in: full = True 면 모두 "바뀜" 으로 본다 (전체 재색인)
#
# -out: plan = {"new":[(doc_id, path, rel, sha, acl)], "changed":[…], "acl":[(doc_id, acl)], "same":[doc_id], "deleted":[doc_id]}
# -out: error = 파일 읽기 실패 시 OSError, 권한 규칙 오류 시 ValueError 전파
#------------------------------------------------------------------
def plan_changes(cfg: dict, m: dict, full: bool = False) -> dict:
    doc_dir = cfg["paths"]["doc_dir"]
    rules = load_rules(cfg)
    default = cfg.get("index", {}).get("default_acl", ["all"])
    plan = {"new": [], "changed": [], "acl": [], "same": [], "deleted": []}
    sig = extract_signature(cfg)
    emb = embed_signature(cfg)
    seen = set()
    for p in list_files(doc_dir):
        rel = rel_key(p, doc_dir)
        did = doc_id_of(rel)
        seen.add(did)
        sha = file_sha256(p)
        acl = acl_for(rel, rules, default)
        if did not in m:
            plan["new"].append((did, p, rel, sha, acl))
        elif (full or m[did]["sha256"] != sha or m[did].get("extract") != sig
              or m[did].get("embed") != emb):
            # 임베딩 모델을 켜거나 바꾸면 그 문서 벡터를 다시 만들어야 하므로 "바뀜" 으로 본다
            plan["changed"].append((did, p, rel, sha, acl))
        elif m[did].get("acl") != acl:
            plan["acl"].append((did, acl))
        else:
            plan["same"].append(did)
    plan["deleted"] = [d for d in m if d not in seen]
    return plan


#------------------------------------------------------------------
# 저장된 청크 파일 읽기 (문서 하나)
#=> data\chunks\<문서ID>.jsonl. 없으면 빈 목록.
#
# -in: cfg = 설정 dict
# -in: doc_id = 문서 ID
#
# -out: chunks = 청크 목록
# -out: error = 없음
#------------------------------------------------------------------
def read_doc_chunks(cfg: dict, doc_id: str) -> list[dict]:
    p = chunk_file(cfg, doc_id)
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


#------------------------------------------------------------------
# 증분 색인 실행
#=> 바뀐 문서만 파싱·청킹하고, 엔진이 OpenSearch 면 그 문서만 교체한다.
#    1) plan_changes 로 할 일을 나눈다
#    2) 사라진 문서: 엔진·청크 파일·파싱 결과·manifest 에서 지운다
#    3) 권한만 바뀐 문서: 엔진의 acl 필드만 갱신
#    4) 새/바뀐 문서: 파싱 → data\parsed 저장 → 청킹 → 엔진 교체 → manifest 갱신 (문서마다 저장)
#    5) 엔진이 비어 있으면 "그대로" 문서도 청크 파일로 다시 넣는다
#    6) [vector] enabled 면 청크마다 임베딩(캐시 data\vectors\<모델>.npz)을 만들어 함께 넣는다 (7단계).
#       벡터 필드가 없는 옛 색인이면 엔진이 색인을 다시 만들고, 저장된 청크로 다시 채운다
#
# -in: cfg = 설정 dict
# -in: full = True 면 엔진 색인을 지우고 모든 문서를 다시 파싱·색인
# -in: log = 진행 출력 함수 (기본 print)
#
# -out: summary = {"new","changed","acl_changed","same","deleted","reloaded","chunks","errors","seconds"}
#   (임베딩 서버 오류는 예외로 멈춘다. 끝낸 문서는 manifest·캐시에 남아 다음 build 가 이어서 한다)
# -out: error = 파서·엔진 오류는 예외 전파 (문서 하나의 파싱 실패는 manifest 의 error 로 기록하고 계속)
#------------------------------------------------------------------
def build(cfg: dict, full: bool = False, log=print) -> dict:
    from .search_engine import make_engine
    t0 = time.perf_counter()
    m = {} if full else load_manifest(cfg)
    plan = plan_changes(cfg, m, full)
    use_os = cfg.get("search", {}).get("engine") == "opensearch"
    engine = make_engine(cfg) if use_os else None
    first = cfg.get("index", {}).get("doc_first_chars", 300)
    parsed_dir = Path(cfg["paths"]["data_dir"]) / "parsed"
    parsed_dir.mkdir(parents=True, exist_ok=True)
    reloaded = 0
    # 벡터를 켜면 청크마다 임베딩을 만들어 엔진에 함께 넣는다 (같은 글은 캐시에서, 설계서 4.3.1)
    emb, vcache = None, None
    if embed_signature(cfg):
        from .hybrid import VectorCache, service_cache_path, service_embedder
        emb, vcache = service_embedder(cfg), VectorCache(service_cache_path(cfg))

    # 청크 목록 → 벡터 (벡터를 안 쓰면 None)
    def vecs(chunks):
        return vcache.vectors([c["text"] for c in chunks], emb) if emb and chunks else None

    if engine:
        engine.ensure_indices(recreate=full)
        if engine.recreated and not full:
            log("벡터 필드가 맞지 않아 색인을 새로 만들었습니다 (저장된 청크로 다시 채웁니다)")
        # 엔진이 비었는데 manifest 가 있다 → 파싱 없이 청크 파일로 다시 넣는다
        n_docs, _ = engine.counts()
        if n_docs == 0 and plan["same"]:
            for did in plan["same"]:
                ch = read_doc_chunks(cfg, did)
                engine.replace_doc(did, ch, m[did].get("acl") or ["all"], first, vecs(ch))
                reloaded += 1
            log(f"엔진이 비어 있어 저장된 청크로 {reloaded}개 문서를 다시 넣었습니다")

    for did in plan["deleted"]:
        if engine:
            engine.delete_doc(did)
        chunk_file(cfg, did).unlink(missing_ok=True)
        (parsed_dir / f"{did}.json").unlink(missing_ok=True)
        log(f"삭제 {did} {m[did].get('rel')}")
        m.pop(did, None)
    for did, acl in plan["acl"]:
        if engine:
            engine.update_acl(did, acl)
        log(f"권한 변경 {did} {m[did].get('rel')} {m[did].get('acl')} → {acl}")
        m[did]["acl"] = acl
    save_manifest(cfg, m)

    todo = [("새로", x) for x in plan["new"]] + [("바뀜", x) for x in plan["changed"]]
    n_chunks, errors = 0, 0
    parser = open_parser(cfg) if todo else None
    try:
        for i, (kind, (did, path, rel, sha, acl)) in enumerate(todo, 1):
            rec = parse_file(parser, path, did, cfg, sha, log)
            (parsed_dir / f"{did}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
            chunks = chunk_and_save(rec, cfg)
            v = vecs(chunks)
            if engine:
                engine.replace_doc(did, chunks, acl, first, v if engine.vector_dim else None)
            n_chunks += len(chunks)
            errors += "error" in rec
            m[did] = {"rel": rel, "path": str(path), "sha256": sha, "title": rec["title"], "acl": acl,
                      "n_chunks": len(chunks), "error": rec.get("error"), "extract": extract_signature(cfg),
                      "embed": embed_signature(cfg),
                      "ocr_fix": rec.get("ocr_fix"),
                      "indexed_at": time.strftime("%Y-%m-%d %H:%M:%S")}
            # 문서마다 저장해 두면 중간에 끊겨도 다음 build 가 이어서 한다
            save_manifest(cfg, m)
            # 임베딩 캐시도 가끔 저장한다 (끊겨도 다시 임베딩하지 않게)
            if vcache and i % 50 == 0:
                vcache.save()
            fx = rec.get("ocr_fix")
            fx = f" 숫자 보정 {fx.get('fixed')}/{fx.get('missing')} 블록" if fx and "fixed" in fx else (f" 보정 실패 {fx['error']}" if fx else "")
            log(f"[{i}/{len(todo)}] {kind} {did} {rec['title']} 청크 {len(chunks)} {rec.get('error') or ''}{fx}")
    finally:
        if parser:
            parser.close()
        if vcache:
            vcache.save()
    if engine:
        engine.refresh()
    s = {"new": len(plan["new"]), "changed": len(plan["changed"]), "acl_changed": len(plan["acl"]),
         "same": len(plan["same"]),
         "deleted": len(plan["deleted"]), "reloaded": reloaded, "chunks": n_chunks, "errors": errors,
         "seconds": round(time.perf_counter() - t0, 1)}
    if engine:
        s["engine_docs"], s["engine_chunks"] = engine.counts()
    return s
