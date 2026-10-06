@echo off
cd /d %~dp0
if exist venv\Scripts\python.exe (venv\Scripts\python.exe server.py --preload --port 8765) else (python server.py --preload --port 8765)
pause
