@echo off
rem AnyDoor 서버 실행(포트 8766). 먼저 setup.bat 로 설치. 생성 서버(8765)와 함께 켜 둔다.
cd /d %~dp0\AnyDoor
call venv_anydoor\Scripts\activate.bat
python anydoor_server.py --port 8766 %*
