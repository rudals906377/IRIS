# AnyDoor 연동 (과제 핵심: 참고 사진의 물체를 내 사진에 합성)

[AnyDoor](https://github.com/ali-vilab/AnyDoor)(ali-vilab, MIT)는 "참고 사진 속 물체 + 그 마스크"를 "대상 사진의 지정한 자리"에 자연스럽게 합성하는 모델이다.
IRIS 에서는 ✨ 생성 모드의 엔진으로 쓴다: 참고 사진의 헤어/손톱/타투를 웹캠 사진의 머리/손톱/팔 자리에 합성.

```
웹 앱(✨ 생성, 엔진: AnyDoor) → IRIS 생성 서버(8765): 영역 마스크·참고 물체 마스크 만들기, 손톱은 하나씩
                              → AnyDoor 서버(8766, 이 폴더): 합성 → 결과 → 색 고정·되붙이기 → 웹 앱
```

## 설치 (윈도우, 엔비디아 GPU, 처음 한 번)

1. 명령 프롬프트에서(한 줄씩):
   ```
   cd 저장소폴더\tools\anydoor
   python setup.py
   ```
   공식 저장소를 받고(git 이 없으면 ZIP 으로), 전용 가상환경에 PyTorch 2.0.1(CUDA 11.8, AnyDoor 코드가 맞춰진 버전)과 패키지를 설치하고, 가중치 두 개(AnyDoor 축약본 4.9GB, DINOv2 4.5GB)를 내려받는다. 20~40분.
3. 실행: `run.bat`(= `python run.py`) → `준비 완료` 가 뜨면 http://127.0.0.1:8766/health 에서 `"ok": true`.
4. IRIS 생성 서버(`tools\genserver`)도 평소처럼 켠다. 두 창 모두 켜 둔 채 웹 앱 ✨ 생성에서 **엔진: AnyDoor** 를 고른다.

## 메모리

AnyDoor 는 Stable Diffusion 2.1 + ControlNet + DINOv2(ViT-G) 조합이라 통째로 올리면 10GB 를 넘는다.
기본은 **저메모리 모드**(확산 단계마다 필요한 부분만 GPU에 올림)로, RTX 3080 10GB 를 겨냥했다. 한 장에 15~30초.
16GB 이상 카드는 `run.bat --no-save-memory` 로 더 빠르게.

## 요청 형식 (AnyDoor 서버)

`POST /compose` `{ ref_image, ref_mask, tar_image, tar_mask (dataURL), guidance 5.0, steps 30, seed }` → `{ image, elapsed_ms }`
마스크는 흰색 = 물체/자리. 대상 자리의 가로세로 비율이 참고 물체와 다르면 찌그러지므로, 생성 서버가 비율을 맞춘 상자를 만든다.

## 알려진 한계 (논문·저장소 기준)

- 2023년 모델이라 손 같은 세밀한 구조·글자는 약하다. 타투(피부 위 평면 도안)가 가장 잘 맞고, 헤어는 모양이 크게 바뀌는 경우 어색할 수 있다.
- 대상 마스크가 너무 거칠면 품질이 떨어진다(저장소 안내). 생성 서버는 MediaPipe 분할·손 점으로 마스크를 만든다.
- 과제 보고서용 비교: 같은 참고 사진으로 기본 엔진(SDXL + IP-Adapter)과 AnyDoor 결과를 나란히 두고 색 오차·정체성·사용자 평가를 표로.
