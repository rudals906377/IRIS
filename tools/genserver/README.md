# IRIS 생성 서버 (사진 한 장 모드)

실시간으로는 안 되는 것 — 헤어 **모양·길이** 바꾸기, 세밀한 **네일아트**, 사진 그대로의 **타투** — 을 생성형 AI로 그린다.
웹 앱에서 웹캠 한 장을 찍어 이 서버로 보내면, 참고 사진의 스타일을 입혀 몇 초 뒤 돌려준다.
**사진은 이 컴퓨터 밖으로 나가지 않는다**(모델 파일만 처음 한 번 내려받는다).

## 무엇을 쓰나

| 역할 | 모델 | 크기 |
|---|---|---|
| 다시 그리기(기본, GPU 10GB 이상) | SDXL 인페인팅 (`diffusers/stable-diffusion-xl-1.0-inpainting-0.1`) + fp16 VAE | 약 7GB |
| 다시 그리기(GPU 10GB 미만·CPU) | Stable Diffusion 1.5 인페인팅 (`stable-diffusion-v1-5/stable-diffusion-inpainting`) | 약 2GB |
| 참고 사진 따라 그리기 | IP-Adapter (`h94/IP-Adapter`; SDXL은 Plus ViT-H, SD1.5는 기본) | 약 2~4GB |
| 어디를 그릴지(마스크) | MediaPipe 분할·손 점·자세 점 (웹 앱과 같은 모델) | 약 20MB |

모델은 GPU 메모리를 보고 자동으로 고른다(`IRIS_GEN_MODEL=auto`). 바꾸려면 `python server.py --preload --model sd15` 또는 `--model sdxl`.
SDXL은 12GB 미만 GPU에서 일부를 CPU에 두고 번갈아 올리므로(자동) 한 장에 10~20초 걸린다.

마스크 안만 새로 그리고 나머지(얼굴·배경)는 원본 픽셀을 그대로 둔다.
작은 부위(손톱·팔)는 마스크 주변만 잘라 크게 그린 뒤 되붙여 세밀하게 나온다.

**헤어 색 고정**: 생성 모델은 색을 자주 틀린다(노랑 → 연두, 반쪽만 검정 등). 그래서 참고 사진이 있으면 생성 뒤
머리 영역의 색 분포를 참고 사진 머리색 분포에 맞춘다(Lab 분위수 대응). 끄려면 요청에 `color_lock: false`.

분야별 기본 세기(`strength`): 헤어 0.95(모양을 새로 그림), 네일 0.65, 타투 0.45.
웹 앱이 보내는 그림에는 실시간 합성(네일 색·타투 도안)이 이미 올라가 있어서, 네일·타투는 그것을 살리며 피부에 녹이는 정도로만 다시 그린다.

## 파일 갱신

서버 코드가 바뀌면(이 저장소에 새 커밋) 서버를 끄고 `python update.py` → 다시 `python server.py --preload`.
venv·모델·cloudflared.exe 는 그대로 둔다. 처음 한 번은 `curl -L -o update.py "https://raw.githubusercontent.com/rudals906377/iris/claude/virtual-try-on-realtime-7uwjlm/tools/genserver/update.py"` 로 받는다.

## 설치 (윈도우, 엔비디아 그래픽카드) — 한 번만

1. Python 3.10~3.12 설치 (python.org, **"Add to PATH" 체크**). `python --version` 으로 확인.
2. 이 저장소를 내려받아(초록 Code 버튼 → Download ZIP, 또는 `git clone`) `IRIS\tools\genserver` 폴더를 연다.
3. **`setup.bat` 더블클릭** — 가상환경 만들기 → GPU용 PyTorch(cu126, 실패하면 cu118) → 나머지 패키지 → `check.py` 환경 검사까지 한 번에 한다. 10~15분.
4. 마지막에 `모두 준비됨` 이 보이면 끝. `[문제]` 가 보이면 그 줄의 → 안내대로 고치고 `python check.py` 로 다시 확인.

리눅스는 `bash setup.sh`, 실행은 `bash run.sh`.

손으로 할 때:
```bat
cd IRIS\tools\genserver
python -m venv venv
venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
python check.py
```

## 실행과 첫 생성 (내일 할 순서)

1. `run.bat` 더블클릭 (= `python server.py --preload`). 처음엔 모델 약 4GB를 내려받아 5~10분, 그 뒤 `준비 완료`.
2. 브라우저에서 http://127.0.0.1:8765/health → `"ok": true, "device": "cuda"` 인지 확인. `"device": "cpu"` 면 GPU용 torch가 안 깔린 것(`python check.py`).
3. https://rudals906377.github.io/IRIS/ 열기 → 카메라 허용 → **✨ 생성** 버튼 → 상태가 `연결됨 · GPU` 인지 확인.
   - 크롬이 "이 사이트가 로컬 네트워크 기기에 접근하려고 합니다" 같은 허용 창을 띄우면 **허용**.
   - `생성 서버가 꺼져 있어요` 가 계속 나오면: 설정(⚙) → 생성 서버 주소가 `http://127.0.0.1:8765` 인지, 서버 창에 오류가 없는지 확인.
4. 📷 사진 따라하기로 참고 사진을 먼저 올리면 생성 설명(영어)이 자동으로 채워진다. 그다음 ✨ 생성에서 대상(헤어·네일·타투) 고르고 **생성**.
   - 헤어: 5~15초(GPU 8GB 기준). 결과가 어색하면 `grow` 를 줄이거나(0.15) 설명을 고쳐 다시.
   - 네일·타투: 손·팔이 화면에 크게 보여야 한다.
5. 결과 화면(전·후 슬라이더)을 스크린샷으로 남겨 두면 다음 날 품질 개선에 바로 쓸 수 있다.

GPU 메모리 6GB 이하면 `set IRIS_GEN_MAX_SIDE=512` 뒤 `run.bat`. GPU가 없으면 CPU로도 돌지만 한 장에 몇 분.
맥(애플 실리콘)은 `pip install torch torchvision`(일반)로 설치하면 MPS로 돈다.

### 다른 컴퓨터(맥북)에서 이 서버 쓰기

서버는 한 대(GPU PC)에만 켜 두고, 다른 컴퓨터의 웹 앱이 그 서버를 부르게 할 수 있다.

- **가장 쉬운 방법: 임시 인터넷 주소(터널)**. GPU PC에서 `run.bat` 로 서버를 켠 뒤 **`tunnel.bat`** 실행 → 화면에 `https://xxxx.trycloudflare.com` 주소가 나온다.
  맥북의 웹 앱(https://rudals906377.github.io/IRIS/) → 설정(톱니바퀴) → **생성 서버 주소**에 그 주소를 넣는다. 같은 와이파이가 아니어도(집에서도) 된다.
  주의: 그 주소를 아는 사람은 누구나 서버를 쓸 수 있고, 사진이 Cloudflare를 거쳐 전달된다(암호화됨). 창을 닫으면 주소가 사라지고 다음에 켜면 새 주소가 나온다.
- 같은 와이파이 안에서만: `python server.py --preload --host 0.0.0.0` 로 켜고 맥북에서 `http://<GPU PC의 IP>:8765` 를 쓴다.
  단, 배포된 https 사이트에서 http 주소를 부르는 건 크롬이 막으므로, 이 방법은 맥북에서 웹 앱을 직접 띄울 때(`web` 폴더에서 `npm run dev`)만 된다. 윈도우 방화벽 허용 창도 떠야 한다.

### 자주 막히는 곳

| 증상 | 원인·해결 |
|---|---|
| `python` 을 찾을 수 없음 | 설치 때 Add to PATH 미체크 → 재설치, 또는 `py -3.11` 로 실행 |
| torch 설치 중 "No matching distribution" | 파이썬 3.13 이상 → 3.11 설치 후 venv 다시 |
| `device: cpu` | GPU용 torch 미설치 → `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126` 다시 |
| `CUDA out of memory` | `set IRIS_GEN_MAX_SIDE=512` (또는 384) 뒤 다시 실행 |
| 모델 내려받기 실패 | 학교 망이 huggingface.co 를 막음 → 휴대폰 핫스팟으로 한 번 받아 두면 캐시에 남는다 |
| 웹 앱이 서버를 못 봄 | 크롬의 로컬 네트워크 접근 허용 창 → 허용. 그래도 안 되면 `chrome://flags/#block-insecure-private-network-requests` 를 Disabled |
| libEGL 오류(리눅스) | `sudo apt-get install libegl1 libgles2 libgl1 libglib2.0-0` |

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
