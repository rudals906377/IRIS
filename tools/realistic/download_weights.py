"""가중치 내려받기(setup.py 가 가상환경 안에서 부른다). 받은 파일은 건너뛰고, 끊긴 것은 이어받는다.
  헤어(HairFastGAN): Hugging Face AIRI-Institute/HairFastGAN 의 필요한 파일만(약 6.4GB)
  메이크업(Stable-Makeup): 공식 Google Drive 폴더의 가중치 3개(약 4.3GB) + SPIGA 얼굴 점 + SD1.5 기본 모델(4GB) + CLIP ViT-L 이미지 인코더(1.6GB)
Hugging Face 파일은 전용 도구 대신 직접 주소(resolve/main)로 받는다: 윈도우에서 도구가 파일 잠금·바로가기 문제로 멈추는 일이 있었다.
"""
from __future__ import annotations

import os
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
HAIR = os.path.join(HERE, "HairFastGAN")
MAKEUP = os.path.join(HERE, "Stable-Makeup")
sys.path.insert(0, HERE)
from setup import HAIR_FILES, MAKEUP_DRIVE  # noqa: E402

HF = "https://huggingface.co/{repo}/resolve/main/{path}"
CLIP_FILES = ["config.json", "model.safetensors", "preprocessor_config.json"]
SD15_FILES = [
    "model_index.json",
    "unet/config.json",
    "unet/diffusion_pytorch_model.safetensors",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
    "text_encoder/config.json",
    "text_encoder/model.safetensors",
    "tokenizer/merges.txt",
    "tokenizer/special_tokens_map.json",
    "tokenizer/tokenizer_config.json",
    "tokenizer/vocab.json",
    "scheduler/scheduler_config.json",
    "feature_extractor/preprocessor_config.json",
]


def fetch(url: str, dst: str, label: str) -> None:
    """이어받기 되는 내려받기(Range). 끝나면 .part 를 제 이름으로."""
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst):
        return
    part = dst + ".part"
    have = os.path.getsize(part) if os.path.exists(part) else 0
    req = urllib.request.Request(url, headers={"User-Agent": "iris-setup"})
    if have:
        req.add_header("Range", f"bytes={have}-")
    print(f"   {label}" + (f" ({have / 2**20:.0f}MB 부터 이어받기)" if have else ""), flush=True)
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                if have and r.status != 206:
                    have = 0
                total = have + int(r.headers.get("Content-Length") or 0)
                done = have
                t0 = time.time()
                last = -1
                with open(part, "ab" if have else "wb") as f:
                    while True:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
                        done += len(chunk)
                        pct = int(done * 100 / total) if total else 0
                        if pct != last and pct % 5 == 0:
                            last = pct
                            el = max(1e-6, time.time() - t0)
                            print(f"\r     {pct:3d}%  {done / 2**20:.0f}MB  {(done - have) / 2**20 / el:.1f}MB/s   ", end="", flush=True)
                print()
            os.replace(part, dst)
            return
        except Exception as e:  # noqa: BLE001
            print(f"\n   [잠깐] {label}: {e} — 다시 시도 {attempt + 1}/3")
            have = os.path.getsize(part) if os.path.exists(part) else 0
            req = urllib.request.Request(url, headers={"User-Agent": "iris-setup", **({"Range": f"bytes={have}-"} if have else {})})
            time.sleep(3)
    raise SystemExit(f"[문제] {label} 내려받기 실패. 다시 실행하면 받던 데서 이어받습니다.")


def main() -> int:
    if "--no-hair" not in sys.argv:
        print("[헤어] HairFastGAN 가중치")
        for rel in HAIR_FILES:
            fetch(HF.format(repo="AIRI-Institute/HairFastGAN", path=rel), os.path.join(HAIR, rel), rel)
        print("  완료")
    if "--no-makeup" not in sys.argv:
        import gdown

        print("[메이크업] Stable-Makeup 가중치(Google Drive)")
        os.makedirs(os.path.join(MAKEUP, "models", "stablemakeup"), exist_ok=True)
        for fid, rel in MAKEUP_DRIVE:
            dst = os.path.join(MAKEUP, rel)
            if os.path.exists(dst) and os.path.getsize(dst) > 1_000_000_000:
                continue
            print("  ", rel, flush=True)
            out = gdown.download(id=fid, output=dst, quiet=False, resume=True)
            if not out:
                print("  [문제] Google Drive 에서 받지 못했습니다(하루 한도일 수 있음). 잠시 뒤 다시 실행하세요")
                return 1
        # SPIGA(얼굴 점) 가중치: 패키지가 스스로 받는 주소는 Google Drive 안내 페이지를 받아 깨진다 → 직접 받아 패키지 폴더에 둔다
        import pkg_resources

        spiga_dir = pkg_resources.resource_filename("spiga", "models/weights")
        os.makedirs(spiga_dir, exist_ok=True)
        spiga_w = os.path.join(spiga_dir, "spiga_300wpublic.pt")
        if not (os.path.exists(spiga_w) and os.path.getsize(spiga_w) > 10_000_000):
            print("   SPIGA 얼굴 점 가중치", flush=True)
            if not gdown.download(id="1YrbScfMzrAAWMJQYgxdLZ9l57nmTdpQC", output=spiga_w, quiet=False, resume=True):
                print("  [문제] SPIGA 가중치를 받지 못했습니다. 잠시 뒤 다시 실행하세요")
                return 1
        print("[메이크업] CLIP 이미지 인코더(1.6GB)")
        for rel in CLIP_FILES:
            fetch(HF.format(repo="openai/clip-vit-large-patch14", path=rel), os.path.join(MAKEUP, "models", "image_encoder_l", rel), rel)
        print("[메이크업] SD1.5 기본 모델(4GB)")
        for rel in SD15_FILES:
            fetch(HF.format(repo="stable-diffusion-v1-5/stable-diffusion-v1-5", path=rel), os.path.join(MAKEUP, "models", "sd15", rel), rel)
        print("  완료")
    return 0


if __name__ == "__main__":
    sys.exit(main())
