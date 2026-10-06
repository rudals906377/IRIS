"""AnyDoor 서버 실행(포트 8766):  python run.py   (먼저 python setup.py 로 설치. 생성 서버 8765 와 함께 켜 둔다)"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "AnyDoor")
WIN = os.name == "nt"
VPY = os.path.join(REPO, "venv_anydoor", "Scripts" if WIN else "bin", "python.exe" if WIN else "python")
if not os.path.exists(VPY):
    print("[문제] 아직 설치 전입니다. 먼저  python setup.py")
    sys.exit(1)
sys.exit(subprocess.call([VPY, "anydoor_server.py", "--port", "8766", *sys.argv[1:]], cwd=REPO))
