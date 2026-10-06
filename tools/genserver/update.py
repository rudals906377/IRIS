"""생성 서버 파일과 AnyDoor 폴더(tools/anydoor)를 깃허브의 최신 버전으로 바꾼다:  python update.py
(venv·모델·cloudflared.exe·AnyDoor 저장소와 가중치는 그대로 둔다. 서버를 끈 상태에서 실행하고, 끝나면 다시 python server.py --preload)
"""
from __future__ import annotations

import os
import sys
import urllib.request

BRANCH = os.environ.get("IRIS_BRANCH", "claude/virtual-try-on-realtime-7uwjlm")
RAW = f"https://raw.githubusercontent.com/rudals906377/iris/{BRANCH}/tools/"
FILES = ["server.py", "pipelines.py", "masks.py", "anydoor_client.py", "check.py", "setup.py", "tunnel.py", "update.py", "requirements.txt", "README.md", "run.bat", "setup.bat", "tunnel.bat", "run.sh", "setup.sh"]
ANYDOOR_FILES = ["anydoor_server.py", "setup.py", "run.py", "setup.bat", "setup_paths.py", "requirements-anydoor.txt", "run.bat", "README.md"]


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    anydoor = os.path.join(os.path.dirname(here), "anydoor")
    os.makedirs(anydoor, exist_ok=True)
    bad = 0
    for sub, folder, names in (("genserver/", here, FILES), ("anydoor/", anydoor, ANYDOOR_FILES)):
        for name in names:
            try:
                with urllib.request.urlopen(RAW + sub + name, timeout=30) as r:
                    data = r.read()
                with open(os.path.join(folder, name), "wb") as f:
                    f.write(data)
                print(f"  [OK] {sub}{name} ({len(data)} bytes)")
            except Exception as e:  # noqa: BLE001
                print(f"  [문제] {sub}{name}: {e}")
                bad += 1
    # 이미 AnyDoor 를 설치한 PC: 저장소 안의 서버 파일도 새 것으로
    inst = os.path.join(anydoor, "AnyDoor", "anydoor_server.py")
    if os.path.isdir(os.path.dirname(inst)):
        try:
            with open(os.path.join(anydoor, "anydoor_server.py"), "rb") as f:
                open(inst, "wb").write(f.read())
            print("  [OK] anydoor/AnyDoor/anydoor_server.py 갱신")
        except Exception as e:  # noqa: BLE001
            print(f"  [문제] AnyDoor 서버 파일 복사: {e}")
    print()
    print("갱신 끝. 다음:  python server.py --preload" if bad == 0 else f"문제 {bad}개 — 인터넷 연결을 확인하고 다시 실행")
    return min(1, bad)


if __name__ == "__main__":
    sys.exit(main())
