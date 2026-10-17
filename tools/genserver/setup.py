"""IRIS 생성 서버 설치(윈도우·리눅스·맥 공통):  python setup.py
하는 일: 가상환경(venv) 만들기 → GPU용 PyTorch(cu126, 실패하면 cu118) → 나머지 패키지 → 환경 검사(check.py).
처음 한 번만 실행. 다시 실행하면 빠진 것만 채운다. 10~15분.
(배치 파일은 한글 인코딩 문제로 PC마다 깨질 수 있어 파이썬으로 둔다)
"""
from __future__ import annotations

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WIN = os.name == "nt"
VENV = os.path.join(HERE, "venv")
VPY = os.path.join(VENV, "Scripts" if WIN else "bin", "python.exe" if WIN else "python")


def run(args: list[str], **kw) -> int:
    print("  $", " ".join(args), flush=True)
    return subprocess.call(args, cwd=HERE, **kw)


def fail(msg: str) -> None:
    print(f"\n[문제] {msg}")
    sys.exit(1)


def main() -> None:
    v = sys.version_info
    print(f"파이썬 {v.major}.{v.minor}.{v.micro}")
    if v < (3, 10) or v >= (3, 13):
        fail("파이썬 3.10~3.12 가 필요합니다. python.org 에서 3.11 설치 때 'Add to PATH' 를 체크하세요.")

    if not os.path.exists(VPY):
        print("\n[1/4] 가상환경 만드는 중...")
        if run([sys.executable, "-m", "venv", VENV]) != 0:
            fail("가상환경 생성 실패")
    else:
        print("\n[1/4] 가상환경 있음")

    run([VPY, "-m", "pip", "install", "--upgrade", "pip", "-q"])

    print("\n[2/4] GPU용 PyTorch 설치 중 (약 2.5GB, 몇 분 걸림)...")
    if run([VPY, "-m", "pip", "install", "torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cu126"]) != 0:
        print("cu126 설치 실패 - 드라이버가 오래된 경우용 cu118 로 다시 시도합니다...")
        if run([VPY, "-m", "pip", "install", "torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cu118"]) != 0:
            fail("PyTorch 설치 실패. 인터넷 연결을 확인하고 다시 실행하세요.")

    print("\n[3/4] 나머지 패키지 설치 중...")
    if run([VPY, "-m", "pip", "install", "-r", "requirements.txt"]) != 0:
        fail("패키지 설치 실패. 다시 실행하면 이어서 설치합니다.")

    print("\n[4/4] 환경 검사...")
    run([VPY, "check.py"])
    print("\n위에 '모든 준비됨' 이 보이면 다음 명령으로 서버를 켭니다:")
    print("  run.bat      (또는  venv\\Scripts\\python server.py --preload)" if WIN else "  ./run.sh")


if __name__ == "__main__":
    main()
