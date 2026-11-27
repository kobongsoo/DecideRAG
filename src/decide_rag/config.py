# -*- coding: utf-8 -*-
"""config.toml 읽기."""
import os
import tomllib
from pathlib import Path

# src/decide_rag/config.py → 프로젝트 폴더(D:\Project\DecideRag)
ROOT = Path(__file__).resolve().parents[2]


#------------------------------------------------------------------
# 설정 파일 읽기
#=> config.toml 을 dict 로 읽고, 상대 경로(data_dir, log_dir)를 프로젝트 폴더 기준 절대 경로로 바꾼다.
#    1) tomllib 으로 파싱
#    2) data_dir·log_dir 이 상대 경로면 ROOT 기준으로 바꾸고 폴더를 만든다
#    3) decide API 키를 환경변수 → 키 파일 순으로 찾아 cfg["decide"]["api_key"] 에 넣는다
#
# -in: path = 설정 파일 경로 (기본 ROOT/config.toml)
#
# -out: cfg = 설정 dict
# -out: error = 파일이 없으면 FileNotFoundError, 문법 오류면 tomllib.TOMLDecodeError
#------------------------------------------------------------------
def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else ROOT / "config.toml"
    cfg = tomllib.loads(path.read_text(encoding="utf-8"))
    for k in ("data_dir", "log_dir"):
        p = Path(cfg["paths"].get(k, k.split("_")[0]))
        # 실행 위치와 상관없이 같은 폴더를 쓰도록 프로젝트 기준으로 고정
        if not p.is_absolute():
            p = ROOT / p
        p.mkdir(parents=True, exist_ok=True)
        cfg["paths"][k] = str(p)
    # 키는 설정 파일에 적지 않으므로 환경변수나 별도 파일에서 읽는다 (값은 어디에도 출력하지 않는다)
    key = os.environ.get("DECIDE_API_KEY", "").strip()
    kf = cfg["decide"].get("api_key_file")
    if not key and kf and Path(kf).exists():
        key = Path(kf).read_text(encoding="utf-8").splitlines()[0].strip()
    cfg["decide"]["api_key"] = key
    return cfg
