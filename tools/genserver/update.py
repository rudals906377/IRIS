"""생성 서버 파일을 깃허브의 최신 버전으로 바꾼다:  python update.py
(venv·모델·cloudflared.exe 는 그대로 둔다. 서버를 끈 상태에서 실행하고, 끝나면 다시 python server.py --preload)
"""
from __future__ import annotations

import os
import sys
import urllib.request

BRANCH = os.environ.get("IRIS_BRANCH", "claude/virtual-try-on-realtime-7uwjlm")
RAW = f"https://raw.githubusercontent.com/rudals906377/iris/{BRANCH}/tools/genserver/"
FILES = ["server.py", "pipelines.py", "masks.py", "check.py", "update.py", "requirements.txt", "README.md", "run.bat", "setup.bat", "tunnel.bat", "run.sh", "setup.sh"]


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    bad = 0
    for name in FILES:
        try:
            with urllib.request.urlopen(RAW + name, timeout=30) as r:
                data = r.read()
            with open(os.path.join(here, name), "wb") as f:
                f.write(data)
            print(f"  [OK] {name} ({len(data)} bytes)")
        except Exception as e:  # noqa: BLE001
            print(f"  [문제] {name}: {e}")
            bad += 1
    print()
    print("갱신 끝. 다음:  python server.py --preload" if bad == 0 else f"문제 {bad}개 — 인터넷 연결을 확인하고 다시 실행")
    return min(1, bad)


if __name__ == "__main__":
    sys.exit(main())
