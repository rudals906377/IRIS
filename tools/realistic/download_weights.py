"""가중치 내려받기(setup.py 가 가상환경 안에서 부른다). 받은 파일은 건너뛰고, 끊긴 것은 이어받는다.
  헤어(HairFastGAN): Hugging Face AIRI-Institute/HairFastGAN 의 필요한 파일만(약 6.4GB)
  메이크업(Stable-Makeup): 공식 Google Drive 폴더의 가중치 3개(약 4.3GB) + SD1.5 기본 모델(4GB) + CLIP ViT-L 이미지 인코더(1.6GB)
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
HAIR = os.path.join(HERE, "HairFastGAN")
MAKEUP = os.path.join(HERE, "Stable-Makeup")
sys.path.insert(0, HERE)
from setup import HAIR_FILES, MAKEUP_DRIVE  # noqa: E402


def main() -> int:
    from huggingface_hub import hf_hub_download, snapshot_download

    if "--no-hair" not in sys.argv:
        print("[헤어] HairFastGAN 가중치")
        for rel in HAIR_FILES:
            if os.path.exists(os.path.join(HAIR, rel)):
                continue
            print("  ", rel, flush=True)
            hf_hub_download("AIRI-Institute/HairFastGAN", rel, local_dir=HAIR)
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
        print("[메이크업] SD1.5 기본 모델 + CLIP 이미지 인코더(Hugging Face)")
        snapshot_download("openai/clip-vit-large-patch14", local_dir=os.path.join(MAKEUP, "models", "image_encoder_l"), allow_patterns=["config.json", "model.safetensors", "preprocessor_config.json"])
        snapshot_download(
            "stable-diffusion-v1-5/stable-diffusion-v1-5",
            local_dir=os.path.join(MAKEUP, "models", "sd15"),
            allow_patterns=["model_index.json", "unet/config.json", "unet/diffusion_pytorch_model.safetensors", "vae/config.json", "vae/diffusion_pytorch_model.safetensors", "text_encoder/config.json", "text_encoder/model.safetensors", "tokenizer/*", "scheduler/*", "feature_extractor/*"],
        )
        print("  완료")
    return 0


if __name__ == "__main__":
    sys.exit(main())
