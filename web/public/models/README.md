# 자체 학습 모델 자리

- `hair-matte-256.onnx` / `hair-matte-384.onnx`: tools/train/hair 로 학습한 머리카락 매팅 모델. 주소 뒤에 `?hairmodel=models/hair-matte-256.onnx` 를 붙이면 쓴다.
- `nail-seg.onnx`: tools/train/nail 로 학습한 손톱 분할 모델(연결 예정).
- `face-parse-512.onnx`: 사진 모드 정밀 얼굴 분할(머리카락·얼굴·목·안경 확률). BiSeNet 얼굴 파싱(zllrunning/face-parsing.PyTorch, MIT) 공개 가중치를 `tools/train/face_parse_export.py` 로 변환(26MB).
- 사진 모드 네일의 손톱 오려 내기는 SlimSAM(Xenova/slimsam-77-uniform, Apache-2.0, 약 14MB)을 Transformers.js 가 Hugging Face 에서 받아 쓴다(이 폴더에 두지 않음).
