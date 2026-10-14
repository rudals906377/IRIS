# 사진처럼 그리기 서버 (헤어 모양·색 + 메이크업을 고른 사진 그대로)

실시간 합성은 색·질감을 덧입히는 방식이라 "사진처럼" 보이는 데 한계가 있다. 이 폴더는 학교 GPU 서버에 **전용 생성 모델** 두 개를 붙여,
✨ AI 생성에서 고른 사진의 머리 모양·색과 화장을 내 사진에 그대로 옮긴다.

| 분야 | 모델 | 하는 일 | 시간(GPU) |
|---|---|---|---|
| 헤어 | [HairFastGAN](https://github.com/AIRI-Institute/HairFastGAN) (MIT) | 고른 사진의 **머리 모양과 색**을 내 얼굴에 옮김(StyleGAN 기반, 정렬된 1024 얼굴) | 1~3초 |
| 헤어 2 | [Stable-Hair](https://github.com/Xiaojiu-z/Stable-Hair) (Apache-2.0) | 내 머리를 먼저 지우고(Bald Converter) 고른 사진의 머리를 **새로 그림**(SD1.5 + 참고 머리 인코더 + ControlNet, 512). 고른 사진에서 얼굴을 못 찾아도 사진 가운데를 참고해 그린다 | 10~20초 |
| 메이크업 | [Stable-Makeup](https://github.com/Xiaojiu-z/Stable-Makeup) (Apache-2.0) | 고른 사진의 **화장**을 내 얼굴에 옮김(SD1.5 + 디테일 인코더 + 구조 ControlNet, 512) | 10~30초 |

```
웹 앱(✨ AI 생성, 엔진: 사진처럼) → IRIS 생성 서버(8765): 섞을 자리(머리카락·얼굴 피부) 마스크, 색감 맞추기
                                  → 이 서버(8767, HairFastGAN·Stable-Makeup): 얼굴 찾기·정렬 → 생성 → 원래 자리로 되돌림
                                  → 헤어 2 서버(8768, Stable-Hair): 같은 일을 디퓨전으로(run.py 가 함께 켬)
```

## 설치 (윈도우, 엔비디아 GPU, 처음 한 번, 30~60분, 디스크 약 20GB)

```
cd 저장소폴더\tools\realistic
python setup.py            (헤어만: python setup.py --no-makeup / 메이크업만: --no-hair)
```
두 공식 저장소를 받고(git 이 없으면 ZIP), 전용 가상환경(`venv_realistic`)에 PyTorch 2.0.1(CUDA 11.8)과 패키지를 설치한 뒤 가중치를 내려받는다:
헤어 6.4GB(Hugging Face), 헤어 2 6.1GB(공식 Google Drive), 메이크업 4.3GB(공식 Google Drive) + SD1.5 4GB(메이크업·헤어 2 공용) + CLIP 1.6GB(Hugging Face). 끊기면 다시 실행하면 이어받는다.
**이미 설치한 PC** 는 `python setup.py` 를 한 번 더 실행하면 새로 추가된 헤어 2 만 받는다(빼려면 `--no-hair2`).

설치가 윈도우에서도 되게 한 것:
- StyleGAN 의 CUDA 확장(컴파일에 Visual Studio C++ 와 CUDA 툴킷 필요)을 **순수 PyTorch 판**(`patches/`)으로 바꿔 복사한다. 조금 느리지만 컴파일이 없다.
- dlib 는 소스 빌드 대신 미리 빌드된 `dlib-bin` 을 쓴다.
- SPIGA 얼굴 점 가중치는 패키지가 Google Drive 안내 페이지를 받아 깨지므로 직접 내려받아 둔다.
- Stable-Makeup 의 `spiga_draw.py` 는 학습용 패키지까지 불러와 추론만 해도 실패하므로 필요한 부분만 `makeup_draw.py` 로 옮겼다.
- Stable-Hair 와 Stable-Makeup 은 각자 고친 diffusers 를 저장소 안에 두고 써서 한 프로세스에 함께 올릴 수 없다 → 헤어 2 는 `stablehair_server.py`(8768)로 따로 띄운다.
  10GB 카드에서 다른 모델과 함께 돌도록 단계마다 필요한 모델만 GPU 에 올렸다 내린다(24GB 이상이면 `--keep-on-gpu`).

## 실행

`run.bat`(= `python run.py`) → `준비 완료` 가 뜨면 http://127.0.0.1:8767/health 에서 `"hair": true, "makeup": true`.
헤어 2 를 설치했으면 같은 창에서 `헤어 2 준비 완료` 도 뜬다(http://127.0.0.1:8768/health). 웹 앱의 헤어는 기본(자동)이 HairFastGAN 이고,
고른 사진에서 얼굴을 못 찾으면 헤어 2 로 다시 그린다. ✨ 패널 고급 → 헤어 모델에서 **다시 그리기**(헤어 2)를 직접 고를 수도 있다.

두 모델 비교(CPU, 같은 얼굴·같은 참고 사진 3쌍): 결과 품질은 비슷했다. HairFastGAN 은 더 매끈하고 분홍 단발의 풍성함을 더 잘 살렸고,
Stable-Hair 는 앞머리 결이 사진에 조금 더 가깝지만 배경이 얼룩지는 경우가 있었다(CPU 검증은 bf16·20단계라 GPU fp16·30단계보다 거칠 수 있음).
그래서 기본은 빠른 HairFastGAN 으로 두었다. 생성 서버는 머리카락 자리만 섞으므로 디퓨전이 얼굴을 조금 바꾸는 것은 결과에 남지 않는다.
IRIS 생성 서버(`tools\genserver`, 먼저 `python update.py` 로 갱신)와 터널도 평소처럼 켠다. 웹 앱 ✨ AI 생성에서 **무엇을: 헤어 또는 메이크업**, 엔진은 서버가 켜져 있으면 자동으로 **사진처럼**.

메모리: 두 모델을 함께 올리면 약 9GB(헤어 fp32 약 3GB + 메이크업 fp16 + CPU 오프로드). 헤어 2 는 쉬는 동안 GPU 를 거의 안 쓰고, 생성 중에만 단계별로 약 3~5GB 를 쓴다.
10GB 카드가 빠듯하면 `run.bat --no-makeup`, `--no-hair`, `--no-hair2` 로 줄인다.

## 요청 형식

- `POST /hair` `{ face, shape, color? }` → `{ image(원래 크기), valid(덮인 곳), aligned(1024), elapsed_ms }` — color 를 비우면 shape 사진의 모양과 색을 함께
- `POST /makeup` `{ face, makeup, steps 30, guidance 1.6, seed? }` → `{ image(원래 크기), valid(얼굴 상자), elapsed_ms }`
- 헤어 2: `POST :8768/hair` `{ face, shape, steps 30, seed? }` → `{ image, valid, aligned(1024), bald(512, 머리를 지운 내 얼굴), elapsed_ms }`

생성 서버(`realistic_client.py`)는 헤어 = (원래 머리카락 ∪ 결과 머리카락) 자리, 메이크업 = 얼굴 피부(구멍 메움) 자리만 부드럽게 섞는다.

## 알려진 한계

- 둘 다 **정면에 가깝고 얼굴이 큰 사진**에서 잘 된다. 옆얼굴·작은 얼굴·가려진 얼굴은 실패하거나 어색하다(422 로 알려 준다).
- HairFastGAN 은 FFHQ(정면 인물 사진)로 학습되어 웹캠의 넓은 화각·어두운 조명에서는 머리 가장자리가 뭉개질 수 있다. 결과 머리카락 자리만 섞고 색감을 맞춰 완화했다.
- Stable-Makeup 은 얼굴 512 해상도로 다시 그리므로 아주 큰 사진에서는 얼굴 결이 약간 부드러워진다.
- Stable-Hair 도 FFHQ(정렬된 정면 얼굴)로 학습되었고, 1단계(머리 지우기)가 잘 안 되면 결과도 어색하다(공식 설명). 512 로 그려 1024 로 키우므로 머리 결이 조금 부드럽다.
- 이 클라우드 환경에는 GPU 가 없어 **설치 스크립트·서버 코드·정렬/되붙이기 계산은 CPU 로 검증**했고, 실제 품질은 학교 GPU 에서 확인해야 한다.
