@echo off
rem IRIS 생성 서버 실행(윈도우). 먼저 README.md의 설치를 마칠 것.
cd /d %~dp0
if exist venv\Scripts\activate.bat call venv\Scripts\activate.bat
python server.py --preload --port 8765
pause
