"""학교 PC의 생성 서버를 다른 컴퓨터(맥북 등)에서도 쓰게 인터넷 주소(https)로 열어 준다:  python tunnel.py
Cloudflare 임시 터널(무료, 계정 불필요). 먼저 서버(run.bat)를 켜 둔 상태에서 실행.
화면에 나오는 https://....trycloudflare.com 주소를 웹 앱 ✨ 생성 패널의 서버 주소에 넣으면 된다.
주소는 마우스로 드래그해 복사(Ctrl+C 를 누르면 터널이 꺼진다). 이 창을 닫으면 주소가 사라진다.
"""
from __future__ import annotations

import os
import subprocess
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
WIN = os.name == "nt"
EXE = os.path.join(HERE, "cloudflared.exe" if WIN else "cloudflared")
URL = "https://github.com/cloudflare/cloudflared/releases/latest/download/" + ("cloudflared-windows-amd64.exe" if WIN else "cloudflared-linux-amd64")
PORT = sys.argv[1] if len(sys.argv) > 1 else "8765"


def main() -> None:
    if not os.path.exists(EXE):
        print("cloudflared 내려받는 중(약 60MB)...")
        try:
            urllib.request.urlretrieve(URL, EXE)
        except Exception as e:  # noqa: BLE001
            print(f"[문제] 내려받기 실패: {e}\n브라우저에서 {URL} 를 직접 받아 이 폴더에 {os.path.basename(EXE)} 이름으로 두세요.")
            sys.exit(1)
        if not WIN:
            os.chmod(EXE, 0o755)
    print("터널 여는 중... 아래에 https://....trycloudflare.com 주소가 나오면 드래그로 복사하세요. (Ctrl+C 는 터널을 끕니다)\n")
    try:
        sys.exit(subprocess.call([EXE, "tunnel", "--url", f"http://127.0.0.1:{PORT}"]))
    except KeyboardInterrupt:
        print("\n터널을 껐습니다. 다시 열려면 python tunnel.py")


if __name__ == "__main__":
    main()
