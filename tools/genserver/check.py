"""생성 서버를 돌리기 전에 환경을 확인한다:  python check.py
파이썬 버전, PyTorch·CUDA(GPU), 필요한 패키지, 모델 캐시를 차례로 검사하고 한국어로 결과를 알려준다.
"""
from __future__ import annotations

import importlib
import os
import sys


def ok(msg: str) -> None:
    print(f"  [OK] {msg}")


def bad(msg: str, fix: str = "") -> None:
    print(f"  [문제] {msg}")
    if fix:
        print(f"         → {fix}")


def main() -> int:
    problems = 0
    print("1) 파이썬")
    v = sys.version_info
    if (3, 10) <= (v.major, v.minor) <= (3, 12):
        ok(f"Python {v.major}.{v.minor}.{v.micro}")
    else:
        bad(f"Python {v.major}.{v.minor} — 3.10~3.12 권장(mediapipe·torch 휠이 없을 수 있음)", "python.org에서 3.11 설치 후 venv 다시 만들기")
        problems += 1

    print("2) PyTorch · GPU")
    try:
        import torch

        ok(f"torch {torch.__version__}")
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_memory / 2**30
            ok(f"CUDA 사용 가능: {name}, 메모리 {mem:.1f}GB")
            if mem < 6.5:
                bad("GPU 메모리 6GB 이하", "set IRIS_GEN_MAX_SIDE=512 (리눅스: export) 로 생성 크기를 줄이기")
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            ok("애플 MPS 사용 가능(GPU PC보다 느림)")
        else:
            bad("GPU(CUDA)를 못 찾음 — CPU로 돌면 한 장에 몇 분", "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126  (드라이버가 오래됐으면 cu118)")
            problems += 1
    except Exception as e:  # noqa: BLE001
        bad(f"torch 없음: {e}", "README의 2단계(pip install torch ...)")
        problems += 1

    print("3) 패키지")
    for mod, pipname in [("diffusers", "diffusers"), ("transformers", "transformers"), ("accelerate", "accelerate"), ("fastapi", "fastapi"), ("uvicorn", "uvicorn[standard]"), ("PIL", "pillow"), ("cv2", "opencv-python-headless"), ("mediapipe", "mediapipe")]:
        try:
            m = importlib.import_module(mod)
            ok(f"{mod} {getattr(m, '__version__', '')}")
        except Exception as e:  # noqa: BLE001
            bad(f"{mod} 없음 ({e})", f"pip install {pipname}  (또는 pip install -r requirements.txt)")
            problems += 1

    print("4) 모델 캐시")
    hf = os.environ.get("HF_HOME") or os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
    hub = os.path.join(hf, "hub")
    want = [
        ("models--diffusers--stable-diffusion-xl-1.0-inpainting-0.1", "SDXL 인페인팅(GPU 10GB 이상이면 기본, 약 7GB)"),
        ("models--madebyollin--sdxl-vae-fp16-fix", "SDXL용 VAE(약 0.3GB)"),
        ("models--stable-diffusion-v1-5--stable-diffusion-inpainting", "SD1.5 인페인팅(GPU 10GB 미만이면 기본, 약 2GB)"),
        ("models--h94--IP-Adapter", "IP-Adapter(참고 사진 따라 그리기, 약 2~4GB)"),
    ]
    for w, desc in want:
        if os.path.isdir(os.path.join(hub, w)):
            ok(f"내려받음: {desc}")
        else:
            print(f"  [안내] 아직 없음: {desc} — 첫 실행 때 자동으로 내려받음")
    cache = os.environ.get("IRIS_GEN_CACHE", os.path.join(os.path.expanduser("~"), ".cache", "iris-genserver"))
    print(f"  MediaPipe 모델 폴더: {cache}")

    print()
    if problems == 0:
        print("모두 준비됨. 다음:  python server.py --preload   → 브라우저에서 http://127.0.0.1:8765/health")
    else:
        print(f"문제 {problems}개. 위의 → 안내대로 고친 뒤 다시 python check.py")
    return problems


if __name__ == "__main__":
    sys.exit(min(1, main()))
