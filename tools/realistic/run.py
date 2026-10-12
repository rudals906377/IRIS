"""사진처럼 그리기 서버 실행(포트 8767):  python run.py [--no-makeup] [--no-hair]
(먼저 python setup.py 로 설치. 생성 서버 8765 와 함께 켜 둔다)"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WIN = os.name == "nt"
VPY = os.path.join(HERE, "venv_realistic", "Scripts" if WIN else "bin", "python.exe" if WIN else "python")
if not os.path.exists(VPY):
    print("[문제] 아직 설치 전입니다. 먼저  python setup.py")
    sys.exit(1)
sys.exit(subprocess.call([VPY, os.path.join(HERE, "realistic_server.py"), "--port", "8767", *sys.argv[1:]], cwd=HERE))
