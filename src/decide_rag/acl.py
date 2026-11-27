# -*- coding: utf-8 -*-
"""문서 권한 (설계서 12.4).

문서별 열람 그룹은 acl.toml 의 경로 규칙으로 정한다 (권한 원천 시스템이 정해지면 이 모듈만 바꾼다 — 13장 열린 질문 1).

  [[rule]]
  pattern = "37_*"            # 문서 폴더 기준 상대 경로 (소문자, '/'), fnmatch 와일드카드
  groups  = ["legal"]

- 맞는 규칙이 여럿이면 그룹을 합친다. 하나도 없으면 [index] default_acl (기본 ["all"]).
- 사용자는 인증 게이트웨이가 준 그룹 + 공개 그룹("all")을 가진다.
"""
import fnmatch
import tomllib
from pathlib import Path


#------------------------------------------------------------------
# 권한 규칙 읽기
#=> [index] acl_file (기본: 프로젝트 폴더의 acl.toml). 파일이 없으면 규칙 없음 = 모든 문서 기본 그룹.
#
# -in: cfg = 설정 dict
#
# -out: rules = [{"pattern": 소문자 패턴, "groups": [그룹, …]}, …]
# -out: error = 파일 형식이 틀리면 tomllib 예외 / 규칙에 pattern·groups 가 없으면 ValueError
#------------------------------------------------------------------
def load_rules(cfg: dict) -> list[dict]:
    from .config import ROOT
    p = Path(cfg.get("index", {}).get("acl_file", "acl.toml"))
    if not p.is_absolute():
        p = ROOT / p
    if not p.exists():
        return []
    rules = []
    for r in tomllib.loads(p.read_text(encoding="utf-8")).get("rule", []):
        if not r.get("pattern") or not r.get("groups"):
            raise ValueError(f"acl 규칙에 pattern·groups 가 필요합니다: {r}")
        rules.append({"pattern": r["pattern"].lower(), "groups": [str(g) for g in r["groups"]]})
    return rules


#------------------------------------------------------------------
# 문서 하나의 열람 그룹
#=> 상대 경로에 맞는 규칙의 그룹을 모두 합친다. 맞는 규칙이 없으면 기본 그룹.
#
# -in: rel = 문서 폴더 기준 상대 경로 (extract.rel_key 결과, 소문자)
# -in: rules = load_rules 결과
# -in: default = 맞는 규칙이 없을 때 그룹 (예: ["all"])
#
# -out: acl = 정렬된 그룹 목록
# -out: error = 없음
#------------------------------------------------------------------
def acl_for(rel: str, rules: list[dict], default: list[str]) -> list[str]:
    groups: set[str] = set()
    for r in rules:
        if fnmatch.fnmatchcase(rel, r["pattern"]):
            groups.update(r["groups"])
    return sorted(groups) if groups else sorted(default)


#------------------------------------------------------------------
# 사용자의 검색 그룹
#=> 게이트웨이가 준 그룹에 공개 그룹을 더한다. 공개 문서는 누구나 볼 수 있어야 하기 때문.
#
# -in: groups = 게이트웨이가 준 그룹 목록
# -in: public = 공개 그룹 이름 (기본 "all")
#
# -out: acl = 정렬된 그룹 목록 (빈 값·공백 제거)
# -out: error = 없음
#------------------------------------------------------------------
def user_acl(groups: list[str], public: str = "all") -> list[str]:
    return sorted({g.strip() for g in groups if g and g.strip()} | {public})
