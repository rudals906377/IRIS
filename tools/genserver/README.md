# IRIS 생성 서버 (사진 한 장 모드)

실시간으로는 안 되는 것 — 헤어 **모양·길이** 바꾸기, 세밀한 **네일아트**, 사진 그대로의 **타투** — 을 생성형 AI로 그린다.
웹 앱에서 웹캠 한 장을 찍어 이 서버로 보내면, 참고 사진의 스타일을 입혀 몇 초 뒤 돌려준다.
**사진은 이 컴퓨터 밖으로 나가지 않는다**(모델 파일만 처음 한 번 내려받는다).

## 무엇을 쓰나

| 역할 | 모델 | 크기 |
|---|---|---|
| 다시 그리기 | Stable Diffusion 1.5 인페인팅 (`stable-diffusion-v1-5/stable-diffusion-inpainting`) | 약 2GB |
| 참고 사진 따라 그리기 | IP-Adapter (`h94/IP-Adapter`, SD1.5용) | 약 1.7GB |
| 어디를 그릴지(마스크) | MediaPipe 분할·손 점·자세 점 (웹 앱과 같은 모델) | 약 20MB |

마스크 안만 새로 그리고 나머지(얼굴·배경)는 원본 픽셀을 그대로 둔다.
작은 부위(손톱·팔)는 마스크 주변만 잘라 크게 그린 뒤 되붙여 세밀하게 나온다.

분야별 기본 세기(`strength`): 헤어 0.95(모양을 새로 그림), 네일 0.65, 타투 0.45.
웹 앱이 보내는 그림에는 실시간 합성(네일 색·타투 도안)이 이미 올라가 있어서, 네일·타투는 그것을 살리며 피부에 녹이는 정도로만 다시 그린다.

## 설치 (윈도우, 엔비디아 그래픽카드)

1. Python 3.10~3.12 설치 (python.org, "Add to PATH" 체크)
2. 명령 프롬프트에서:
   ```bat
   cd IRIS\tools\genserver
   python -m venv venv
   venv\Scripts\activate
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   pip install -r requirements.txt
   ```
3. 실행: `run.bat` (또는 `python server.py --preload`)
4. 브라우저에서 http://127.0.0.1:8765/health 가 `"ok": true` 면 준비 끝.
5. IRIS 웹 앱 → 설정 → **생성 서버 주소**가 `http://127.0.0.1:8765` 인지 확인 → 아래 **✨ 생성** 버튼.

처음 실행 때 모델을 내려받아 5~10분 걸린다. 그 뒤로는 한 장에 GPU 8GB 기준 5~15초.
GPU 메모리 6GB 이하면 `set IRIS_GEN_MAX_SIDE=512` 로 줄인다. GPU가 없으면 CPU로도 돌지만 한 장에 몇 분이 걸린다.

맥(애플 실리콘)에서는 `pip install torch torchvision`(일반)로 설치하면 MPS로 돈다. 속도는 GPU PC보다 느리다.

## 연결만 시험하기

모델 없이 `python server.py --dry-run` 으로 띄우면, 다시 그릴 영역을 보라색으로 칠한 그림을 돌려준다.
웹 앱과의 연결, 마스크 위치(머리·손톱·팔)를 확인하는 용도다.

## 요청 형식

`POST /generate` (JSON)

| 항목 | 뜻 |
|---|---|
| `category` | `hair` · `nail` · `tattoo` |
| `image` | 웹캠 한 장 (dataURL) |
| `reference` | 참고 사진 (dataURL, 선택) — 있으면 IP-Adapter로 스타일을 따른다 |
| `desc` | 영어 설명(선택). 웹 앱은 스타일 AI가 읽은 속성을 영어로 넣어 준다 |
| `place` | 타투 부위: forearmL/R, upperArmL/R, neckL/R, chest |
| `grow` | 헤어: 머리 주변을 얼마나 넓게 다시 그릴지(0.25 기본, 길게 기르려면 0.5) |
| `extend` | 네일: 손가락 끝 너머로 늘릴 길이(0~1) |
| `steps`, `strength`, `guidance`, `ip_scale`, `seed` | 생성 세부 조절(비우면 분야별 기본값) |

응답: `image`(결과 dataURL), `mask`, `elapsed_ms`, `prompt`.

## 한계와 다음 단계

- 얼굴은 마스크 밖이라 그대로지만, 머리카락이 얼굴 가장자리와 만나는 곳은 가끔 어색하다. `grow`를 줄이면 덜하다.
- 네일은 손톱 영역 추정이 실시간 모드와 같은 방식이라, 손이 작게 찍히면 어긋난다. 손을 카메라에 가깝게.
- 더 좋은 전용 모델(헤어: Stable-Hair, HairFastGAN / 네일·타투: SDXL 인페인팅 + ControlNet)로 바꾸려면 `pipelines.py`의 `Generator`만 교체하면 된다.
