"""사진처럼 그리기 서버 실행(포트 8767) + 헤어 2(Stable-Hair, 포트 8768):  python run.py [--no-makeup] [--no-hair] [--no-hair2]
(먼저 python setup.py 로 설치. 생성 서버 8765 와 함께 켜 둔다)
헤어 2 는 Stable-Makeup 과 다른 diffusers 를 저장소 안에 두고 써서 따로 띄운다. 이 창 하나에 두 서버의 글이 함께 나온다."""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WIN = os.name == "nt"
VPY = os.path.join(HERE, "venv_realistic", "Scripts" if WIN else "bin", "python.exe" if WIN else "python")
if not os.path.exists(VPY):
    print("[문제] 아직 설치 전입니다. 먼저  python setup.py")
    sys.exit(1)
ENV = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
args = [a for a in sys.argv[1:] if a != "--no-hair2"]
hair2 = None
if "--no-hair2" not in sys.argv and os.path.exists(os.path.join(HERE, "Stable-Hair", "models", "stage2", "pytorch_model.bin")):
    hair2 = subprocess.Popen([VPY, os.path.join(HERE, "stablehair_server.py"), "--port", "8768"], cwd=HERE, env=ENV)
only_hair2 = "--no-hair" in args and "--no-makeup" in args
try:
    code = hair2.wait() if (only_hair2 and hair2) else subprocess.call([VPY, os.path.join(HERE, "realistic_server.py"), "--port", "8767", *args], cwd=HERE, env=ENV)
finally:
    if hair2 and hair2.poll() is None:
        hair2.terminate()
sys.exit(code)
