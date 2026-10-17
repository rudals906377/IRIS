#!/usr/bin/env bash
# IRIS 생성 서버 실행(리눅스·맥). 먼저 README.md의 설치를 마칠 것.
cd "$(dirname "$0")"
[ -f venv/bin/activate ] && source venv/bin/activate
python server.py --preload --port 8765 "$@"
