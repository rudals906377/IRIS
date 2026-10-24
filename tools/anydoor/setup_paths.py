"""AnyDoor 설정 파일에 가중치 경로를 써 넣는다(AnyDoor 저장소 루트에서 실행).
- configs/inference.yaml: pretrained_model → path/anydoor-pruned.ckpt
- configs/anydoor.yaml:  DINOv2 weight → path/dinov2_vitg14_pretrain.pth
"""
import os
import re

root = os.getcwd()
inf = os.path.join(root, "configs", "inference.yaml")
cfg = os.path.join(root, "configs", "anydoor.yaml")
if not os.path.exists(inf) or not os.path.exists(cfg):
    raise SystemExit("AnyDoor 저장소 루트(configs 폴더가 있는 곳)에서 실행하세요")
ckpt = "path/anydoor-pruned.ckpt"
dino = "path/dinov2_vitg14_pretrain.pth"
s = open(inf, encoding="utf-8").read()
s = re.sub(r"pretrained_model:.*", f"pretrained_model: {ckpt}", s)
open(inf, "w", encoding="utf-8").write(s)
s = open(cfg, encoding="utf-8").read()
s = re.sub(r"weight:\s*path/dinov2_vitg14_pretrain\.pth", f"weight: {dino}", s)
open(cfg, "w", encoding="utf-8").write(s)
missing = [p for p in (ckpt, dino) if not os.path.exists(os.path.join(root, p))]
print("설정 완료:", ckpt, dino)
if missing:
    print("[주의] 아직 없는 파일:", missing)
