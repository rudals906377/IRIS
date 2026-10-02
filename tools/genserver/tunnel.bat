@echo off
rem 학교 PC의 생성 서버를 다른 컴퓨터(맥북 등)에서도 쓰게 인터넷 주소(https)로 열어 준다(Cloudflare 임시 터널, 무료, 계정 불필요).
rem 먼저 run.bat 로 서버를 켜 둔 상태에서 이 파일을 실행. 화면에 나오는 https://....trycloudflare.com 주소를
rem 웹 앱 설정(톱니바퀴) -> 생성 서버 주소에 넣으면 된다. 이 창을 닫으면 주소가 사라진다.
cd /d %~dp0
if not exist cloudflared.exe (
  echo cloudflared.exe 내려받는 중(약 60MB)...
  curl -L -o cloudflared.exe https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe
  if errorlevel 1 (echo [문제] 내려받기 실패. 브라우저에서 위 주소로 직접 받아 이 폴더에 cloudflared.exe 로 두세요. & pause & exit /b 1)
)
cloudflared.exe tunnel --url http://127.0.0.1:8765
pause
