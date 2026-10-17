# 머리카락 전용 매팅 모델 만들기 (자체 모델 1번)

실시간 화면에서 머리카락 올 하나하나가 살아나게 하려면, 구글 범용 분할(256픽셀) 대신
**머리카락만 또렷하게 떼어내는 작은 모델**이 필요하다. 이 폴더가 그 모델을 만든다.

## 어떻게 만드나

1. **데이터**: 공개 얼굴 데이터 CelebAMask-HQ(28,500장, 1024픽셀)에 사람이 그린 머리카락 라벨이 있다. 다만 테두리가 거칠다.
2. **선생님 모델로 정밀화**(`prepare.py`): 거친 라벨의 테두리 띠를 큰 모델 두 개(ViTMatte, BiRefNet)에 맡겨 올 단위 알파(0~1)로 바꾼다.
   내 웹캠 사진도 같은 방법으로 자동 라벨을 만들어 섞는다(우리 환경에 맞추기).
3. **학생 모델 학습**(`train.py`): 약 1M 매개변수의 가벼운 U-Net이 384×384 사진에서 알파를 바로 내도록 학습한다.
   브라우저(onnxruntime-web, WebGPU)에서 몇 ms에 돈다.
4. **웹 앱에 넣기**: 머리 주변 잘라낸 그림을 이 모델에 넣어 머리카락 확률(R)로 쓴다(`tracker.ts`의 머리 분할 자리).

## 실행 (GPU PC)

```bat
cd IRIS\tools\train\hair
..\..\genserver\venv\Scripts\activate
pip install -r requirements.txt

rem 1) 데이터: 먼저 1조각(약 4,700장)으로. 선생님 모델 두 개가 처음 한 번 내려받아진다(약 1.5GB)
python prepare.py --out data --shards 1
rem    잘 되면 전부:  python prepare.py --out data --shards 6 --test   (3~4시간)
rem    내 웹캠 사진:  python prepare.py --out data --webcam C:\경로\내사진폴더

rem 2) 학습
python train.py --data data --out out --epochs 20
```

- `prepare.py`는 중간에 끊겨도 다시 실행하면 이어서 한다(이미 만든 파일은 건너뜀).
- `train.py`는 매 회 `검증 평균오차(MAD)`를 찍는다. 3% 아래면 쓸 만하고 2% 아래면 좋다. `out/samples/`에 사진·정답·예측 그림이 쌓인다.
- 결과 `out/hair-matte-384.onnx`(약 4MB)와 `hair-matte-256.onnx`를 보내 주면 웹 앱에 연결한다.

## 웹캠 사진 모으기 (선택이지만 효과 큼)

공개 데이터는 정면 스튜디오 사진이 많아서, 우리 웹캠 환경(책상 조명, 노트북 카메라 화질)과 다르다.
본인·친구의 웹캠 캡처 100~300장을 `--webcam`으로 넣으면 실제 화면에서 훨씬 안정적이다.
머리 모양·조명·배경을 다양하게, 얼굴이 화면 폭의 1/3 이상이면 된다. 라벨은 자동이라 손이 안 간다.

## 시험용(작게)

```
python prepare.py --out data_small --shards 0 --test --limit 100 --no-person
python train.py --data data_small --out out_small --epochs 3
```
