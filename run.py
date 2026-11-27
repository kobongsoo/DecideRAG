# -*- coding: utf-8 -*-
"""DecideRAG 실행기. src 폴더를 import 경로에 넣고 decide_rag.cli 를 부른다.

  python run.py build
  python run.py ask "결혼하면 축의금이랑 휴가 얼마나 줘?"
  python run.py eval
  python run.py summary
"""
import sys
from pathlib import Path

# 설치 없이 바로 돌리도록 src 를 경로에 넣는다
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from decide_rag.cli import main

if __name__ == "__main__":
    main()
