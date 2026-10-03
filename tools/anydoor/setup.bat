@echo off
rem AnyDoor(물체 합성) 설치 - 윈도우, 엔비디아 GPU. 처음 한 번만. 이 창은 명령 프롬프트에서  setup.bat  로 실행.
rem 하는 일: 공식 저장소 받기 -> 전용 가상환경(venv_anydoor) -> PyTorch 2.0.1(cu118) + 패키지 -> 가중치 2개(약 9.4GB) -> 설정 파일 경로 -> 서버 파일 복사
cd /d %~dp0
where python >nul 2>nul || (echo [문제] python 을 찾지 못했습니다 & exit /b 1)
where git >nul 2>nul || (echo [문제] git 이 없습니다. https://git-scm.com/download/win 에서 설치 후 다시 & exit /b 1)

if not exist AnyDoor (
  echo 저장소 받는 중...
  git clone --depth 1 https://github.com/ali-vilab/AnyDoor.git || (echo [문제] git clone 실패 & exit /b 1)
)
cd AnyDoor
if not exist venv_anydoor (
  echo 가상환경 만드는 중...
  python -m venv venv_anydoor || (echo [문제] 가상환경 생성 실패 & exit /b 1)
)
call venv_anydoor\Scripts\activate.bat
python -m pip install --upgrade pip
echo PyTorch 2.0.1 (CUDA 11.8) 설치 중... (AnyDoor 코드가 이 버전에 맞춰져 있음)
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118 || (echo [문제] torch 설치 실패 - 파이썬 3.10/3.11 인지 확인 & exit /b 1)
pip install --no-deps xformers==0.0.22
echo 나머지 패키지 설치 중...
pip install -r ..\requirements-anydoor.txt || (echo [문제] 패키지 설치 실패 & exit /b 1)

if not exist path mkdir path
if not exist path\anydoor-pruned.ckpt (
  echo AnyDoor 가중치(축약본 4.9GB) 내려받는 중...
  curl -L -o path\anydoor-pruned.ckpt "https://huggingface.co/bdsqlsz/AnyDoor-Pruned/resolve/main/epoch%%3D1-step%%3D8687-pruned.ckpt" || (echo [문제] 가중치 내려받기 실패 & exit /b 1)
)
if not exist path\dinov2_vitg14_pretrain.pth (
  echo DINOv2 가중치(4.5GB) 내려받는 중...
  curl -L -o path\dinov2_vitg14_pretrain.pth "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_pretrain.pth" || (echo [문제] DINOv2 내려받기 실패 & exit /b 1)
)
echo 설정 파일 경로 쓰는 중...
python ..\setup_paths.py || exit /b 1
copy /y ..\anydoor_server.py anydoor_server.py >nul
echo.
echo 설치 끝. 실행:  run.bat  (AnyDoor 서버, 포트 8766)
