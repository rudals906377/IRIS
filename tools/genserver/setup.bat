@echo off
rem IRIS 생성 서버 한 번에 설치(윈도우, 엔비디아 GPU). 처음 한 번만 실행.
rem 하는 일: 가상환경 만들기 → GPU용 PyTorch 설치 → 나머지 패키지 설치 → 환경 검사(check.py)
cd /d %~dp0
where python >nul 2>nul || (echo [문제] python 을 찾지 못했습니다. python.org 에서 3.11 설치 때 "Add to PATH" 를 체크하세요. & pause & exit /b 1)
if not exist venv (
  echo 가상환경 만드는 중...
  python -m venv venv || (echo [문제] 가상환경 생성 실패 & pause & exit /b 1)
)
call venv\Scripts\activate.bat
python -m pip install --upgrade pip
echo GPU용 PyTorch 설치 중(약 2.5GB, 몇 분 걸림)...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
if errorlevel 1 (
  echo cu126 설치 실패 - 드라이버가 오래된 경우 cu118 로 다시 시도합니다...
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
)
echo 나머지 패키지 설치 중...
pip install -r requirements.txt || (echo [문제] 패키지 설치 실패 & pause & exit /b 1)
echo.
python check.py
echo.
echo 위에 "모두 준비됨" 이 보이면 run.bat 를 실행하세요.
pause
