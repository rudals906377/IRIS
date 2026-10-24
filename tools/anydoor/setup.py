"""AnyDoor(물체 합성) 설치 - 윈도우·리눅스, 엔비디아 GPU. 처음 한 번만:  python setup.py
하는 일: 공식 저장소 받기(git 없으면 ZIP) → 전용 가상환경(venv_anydoor) → PyTorch 2.0.1(cu118) + 패키지
        → 가중치 2개(약 9.4GB, 끊기면 이어받기) → 설정 파일 경로 → 서버 파일 복사.
다시 실행하면 빠진 것만 채운다. 20~40분(인터넷 속도에 따라).
"""
from __future__ import annotations

import io
import os
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
WIN = os.name == "nt"
REPO = os.path.join(HERE, "AnyDoor")
VENV = os.path.join(REPO, "venv_anydoor")
VPY = os.path.join(VENV, "Scripts" if WIN else "bin", "python.exe" if WIN else "python")
WEIGHTS = [
    ("path/anydoor-pruned.ckpt", "https://huggingface.co/bdsqlsz/AnyDoor-Pruned/resolve/main/epoch%3D1-step%3D8687-pruned.ckpt", "AnyDoor 가중치(축약본 4.9GB)"),
    ("path/dinov2_vitg14_pretrain.pth", "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_pretrain.pth", "DINOv2 가중치(4.5GB)"),
]


def run(args: list[str], cwd: str = REPO) -> int:
    print("  $", " ".join(args), flush=True)
    return subprocess.call(args, cwd=cwd)


def fail(msg: str) -> None:
    print(f"\n[문제] {msg}")
    sys.exit(1)


def fetch_repo() -> None:
    if os.path.isdir(os.path.join(REPO, "configs")):
        print("[1/6] AnyDoor 저장소 있음")
        return
    print("[1/6] AnyDoor 저장소 받는 중...")
    if shutil.which("git"):
        if run(["git", "clone", "--depth", "1", "https://github.com/ali-vilab/AnyDoor.git", REPO], cwd=HERE) == 0:
            return
        print("git clone 실패 - ZIP 으로 받습니다")
    url = "https://github.com/ali-vilab/AnyDoor/archive/refs/heads/main.zip"
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            data = r.read()
    except Exception as e:  # noqa: BLE001
        fail(f"저장소 ZIP 내려받기 실패: {e}")
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        top = z.namelist()[0].split("/")[0]
        z.extractall(HERE)
    os.rename(os.path.join(HERE, top), REPO)


def download(rel: str, url: str, label: str) -> None:
    dst = os.path.join(REPO, rel)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    part = dst + ".part"
    if os.path.exists(dst):
        print(f"  {label}: 있음")
        return
    have = os.path.getsize(part) if os.path.exists(part) else 0
    req = urllib.request.Request(url, headers={"User-Agent": "iris-setup"})
    if have:
        req.add_header("Range", f"bytes={have}-")
        print(f"  {label}: {have / 2**30:.2f}GB 부터 이어받기")
    else:
        print(f"  {label}: 내려받는 중")
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            if have and r.status != 206:
                have = 0
            total = have + int(r.headers.get("Content-Length") or 0)
            mode = "ab" if have else "wb"
            done = have
            t0 = time.time()
            last = -1
            with open(part, mode) as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
                    done += len(chunk)
                    pct = int(done * 100 / total) if total else 0
                    if pct != last and pct % 2 == 0:
                        last = pct
                        el = max(1e-6, time.time() - t0)
                        print(f"\r    {pct:3d}%  {done / 2**30:.2f}GB  {(done - have) / 2**20 / el:.1f}MB/s   ", end="", flush=True)
            print()
    except Exception as e:  # noqa: BLE001
        fail(f"{label} 내려받기 실패: {e}\n다시 실행하면 받던 데서 이어받습니다.")
    os.replace(part, dst)


def main() -> None:
    v = sys.version_info
    print(f"파이썬 {v.major}.{v.minor}.{v.micro}")
    if v < (3, 10) or v >= (3, 12):
        fail("AnyDoor 는 파이썬 3.10 또는 3.11 이 필요합니다(PyTorch 2.0.1 이 3.12 를 지원하지 않음).")
    fetch_repo()

    if not os.path.exists(VPY):
        print("\n[2/6] 가상환경 만드는 중...")
        if run([sys.executable, "-m", "venv", VENV]) != 0:
            fail("가상환경 생성 실패")
    else:
        print("\n[2/6] 가상환경 있음")
    run([VPY, "-m", "pip", "install", "--upgrade", "pip", "-q"])

    print("\n[3/6] PyTorch 2.0.1 (CUDA 11.8) 설치 중... (AnyDoor 코드가 이 버전에 맞춰져 있음, 약 2.5GB)")
    if run([VPY, "-m", "pip", "install", "torch==2.0.1+cu118", "torchvision==0.15.2+cu118", "--extra-index-url", "https://download.pytorch.org/whl/cu118"]) != 0:
        fail("torch 설치 실패 - 인터넷 연결과 파이썬 3.10/3.11 인지 확인")
    run([VPY, "-m", "pip", "install", "--no-deps", "xformers==0.0.22"])

    print("\n[4/6] 나머지 패키지 설치 중...")
    if run([VPY, "-m", "pip", "install", "-r", os.path.join(HERE, "requirements-anydoor.txt")]) != 0:
        fail("패키지 설치 실패. 다시 실행하면 이어서 설치합니다.")

    print("\n[5/6] 가중치 내려받기(약 9.4GB)...")
    for rel, url, label in WEIGHTS:
        download(rel, url, label)

    print("\n[6/6] 설정 파일 경로 쓰기 + 서버 파일 복사...")
    if run([VPY, os.path.join(HERE, "setup_paths.py")]) != 0:
        fail("설정 파일 쓰기 실패")
    shutil.copy(os.path.join(HERE, "anydoor_server.py"), os.path.join(REPO, "anydoor_server.py"))
    print("\n설치 끝. 실행:  run.bat   (또는  python run.py) → '준비 완료' 가 뜨면 AnyDoor 서버(포트 8766)")


if __name__ == "__main__":
    main()
