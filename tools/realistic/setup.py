"""사진처럼 그리기(헤어 HairFastGAN + 헤어 2 Stable-Hair + 메이크업 Stable-Makeup) 설치 - 윈도우·리눅스, 엔비디아 GPU. 처음 한 번만:  python setup.py
하는 일: 공식 저장소 받기(git 없으면 ZIP) → 전용 가상환경(venv_realistic) → PyTorch 2.0.1(cu118) + 패키지
        → 가중치(헤어 약 6.4GB: Hugging Face / 헤어 2 약 6.1GB: Google Drive / 메이크업 약 4.3GB: Google Drive + SD1.5 4GB + CLIP 1.6GB)
        → CUDA 컴파일 없이 도는 패치 복사 → 서버 파일 복사.
다시 실행하면 빠진 것만 채운다(이미 설치했으면 새로 추가된 헤어 2 만 받는다). 30~60분(인터넷 속도에 따라). 디스크 약 26GB.
옵션:  --no-makeup  (메이크업 빼기)   --no-hair  (HairFastGAN 빼기)   --no-hair2  (Stable-Hair 빼기)
"""
from __future__ import annotations

import io
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
WIN = os.name == "nt"
VENV = os.path.join(HERE, "venv_realistic")
VPY = os.path.join(VENV, "Scripts" if WIN else "bin", "python.exe" if WIN else "python")
HAIR = os.path.join(HERE, "HairFastGAN")
MAKEUP = os.path.join(HERE, "Stable-Makeup")
HAIR2 = os.path.join(HERE, "Stable-Hair")

HAIR_FILES = [
    "pretrained_models/StyleGAN/ffhq.pt",
    "pretrained_models/StyleGAN/ffhq_PCA.npz",
    "pretrained_models/encoder4editing/e4e_ffhq_encode.pt",
    "pretrained_models/ShapeAdaptor/mask_generator.pth",
    "pretrained_models/ShapeAdaptor/shape_predictor_68_face_landmarks.dat",
    "pretrained_models/ArcFace/backbone_ir50.pth",
    "pretrained_models/PostProcess/latent_avg.pt",
    "pretrained_models/PostProcess/pp_model.pth",
    "pretrained_models/FeatureStyleEncoder/143_enc.pth",
    "pretrained_models/FeatureStyleEncoder/79999_iter.pth",
    "pretrained_models/FeatureStyleEncoder/backbone.pth",
    "pretrained_models/FeatureStyleEncoder/psp_ffhq_encode.pt",
    "pretrained_models/BiSeNet/seg.pth",
    "pretrained_models/BiSeNet/face_parsing_79999_iter.pth",
    "pretrained_models/Blending/checkpoint.pth",
    "pretrained_models/Rotate/rotate_best.pth",
    "pretrained_models/STAR/WFLW_STARLoss_NME_4_02_FR_2_32_AUC_0_605.pkl",
    "pretrained_models/sean_checkpoints/CelebA-HQ_pretrained/latest_net_G.pth",
]
# Stable-Makeup 가중치(공식 README 의 Google Drive 폴더)
MAKEUP_DRIVE = [
    ("1hXwUNLFTUuT_AseMEiLOTkwj0JVoP9PT", "models/stablemakeup/pytorch_model.bin"),
    ("1erdooW-JT9YP6jNNc8ZR4-Kc5KOBumss", "models/stablemakeup/pytorch_model_1.bin"),
    ("1vY1akep_tWxh36iv6IZYxu-HX0yTcbpZ", "models/stablemakeup/pytorch_model_2.bin"),
]
# Stable-Hair 가중치(공식 README 의 Google Drive 폴더): 1단계 머리 지우기, 2단계 참고 머리 인코더·어댑터·ControlNet
HAIR2_DRIVE = [
    ("1oYNoKPEN0mZpRhZ7s3_xSDlaO209vFn4", "models/stage1/pytorch_model.bin"),
    ("1voeq4Uw9RBv-c4vcfc8GR3MI8Qs0XNNa", "models/stage2/pytorch_model.bin"),
    ("15wytQlOe94dF1KZJUwqf0RGKaH9WQZe-", "models/stage2/pytorch_model_1.bin"),
    ("1Uv3l3JCo-aOJbEWR1shINwl2sY5eGcIL", "models/stage2/pytorch_model_2.bin"),
]


def run(args: list[str], cwd: str = HERE) -> int:
    print("  $", " ".join(args), flush=True)
    # 한글 윈도우(cp949)에서 일부 패키지(FaceLib)의 설치 파일이 UTF-8 README 를 읽다 깨진다 → 파이썬을 UTF-8 모드로
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    return subprocess.call(args, cwd=cwd, env=env)


def fail(msg: str) -> None:
    print(f"\n[문제] {msg}")
    sys.exit(1)


def fetch_repo(dst: str, owner_repo: str, marker: str, label: str) -> None:
    if os.path.exists(os.path.join(dst, marker)):
        print(f"  {label} 저장소 있음")
        return
    print(f"  {label} 저장소 받는 중...")
    if shutil.which("git"):
        if run(["git", "clone", "--depth", "1", f"https://github.com/{owner_repo}.git", dst], cwd=HERE) == 0:
            return
        print("  git clone 실패 - ZIP 으로 받습니다")
    url = f"https://github.com/{owner_repo}/archive/refs/heads/main.zip"
    try:
        with urllib.request.urlopen(url, timeout=120) as r:
            data = r.read()
    except Exception as e:  # noqa: BLE001
        fail(f"{label} ZIP 내려받기 실패: {e}")
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        top = z.namelist()[0].split("/")[0]
        z.extractall(HERE)
    os.rename(os.path.join(HERE, top), dst)


def main() -> None:
    v = sys.version_info
    print(f"파이썬 {v.major}.{v.minor}.{v.micro}")
    if v < (3, 10) or v >= (3, 12):
        fail("파이썬 3.10 또는 3.11 이 필요합니다(PyTorch 2.0.1).")
    want_hair = "--no-hair" not in sys.argv
    want_makeup = "--no-makeup" not in sys.argv
    want_hair2 = "--no-hair2" not in sys.argv

    print("\n[1/6] 저장소")
    if want_hair:
        fetch_repo(HAIR, "AIRI-Institute/HairFastGAN", "hair_swap.py", "HairFastGAN")
    if want_makeup:
        fetch_repo(MAKEUP, "Xiaojiu-z/Stable-Makeup", "pipeline_sd15.py", "Stable-Makeup")
    if want_hair2:
        fetch_repo(HAIR2, "Xiaojiu-z/Stable-Hair", "infer_full.py", "Stable-Hair")

    if not os.path.exists(VPY):
        print("\n[2/6] 가상환경 만드는 중...")
        if run([sys.executable, "-m", "venv", VENV]) != 0:
            fail("가상환경 생성 실패")
    else:
        print("\n[2/6] 가상환경 있음")
    run([VPY, "-m", "pip", "install", "--upgrade", "pip", "-q"])

    print("\n[3/6] PyTorch 2.0.1 (CUDA 11.8) 설치 중... (약 2.5GB)")
    if run([VPY, "-m", "pip", "install", "torch==2.0.1+cu118", "torchvision==0.15.2+cu118", "--extra-index-url", "https://download.pytorch.org/whl/cu118"]) != 0:
        fail("torch 설치 실패 - 인터넷 연결과 파이썬 3.10/3.11 인지 확인")
    print("\n[4/6] 나머지 패키지 설치 중...")
    if run([VPY, "-m", "pip", "install", "-r", os.path.join(HERE, "requirements-realistic.txt")]) != 0:
        fail("패키지 설치 실패. 다시 실행하면 이어서 설치합니다.")

    print("\n[5/6] 가중치 내려받기(끊기면 다시 실행: 받은 것은 건너뜀)")
    dl = os.path.join(HERE, "download_weights.py")
    args = [VPY, dl]
    if not want_hair:
        args.append("--no-hair")
    if not want_makeup:
        args.append("--no-makeup")
    if not want_hair2:
        args.append("--no-hair2")
    if run(args) != 0:
        fail("가중치 내려받기 실패. 다시 실행하면 이어받습니다.")

    print("\n[6/6] 패치·서버 파일 복사")
    if want_hair:
        for name in ("fused_act.py", "upfirdn2d.py"):
            shutil.copy(os.path.join(HERE, "patches", name), os.path.join(HAIR, "models", "stylegan2", "op", name))
        print("  HairFastGAN: StyleGAN CUDA 확장 → 순수 PyTorch 판으로 바꿈(컴파일 불필요)")
    print("\n설치 끝. 실행:  run.bat   (또는  python run.py) → '준비 완료' 가 뜨면 사진처럼 그리기 서버(포트 8767, 헤어 2 는 8768)")


if __name__ == "__main__":
    main()
