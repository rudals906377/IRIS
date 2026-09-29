# IRIS — 실시간 웹캠 가상 착용

쇼핑몰 상품 이미지를 웹캠 속 **지금 움직이는 내 몸**에 거울처럼 바로 입혀 보는 웹 서비스입니다.
미래내일 일경험 「마스크 기반 영상합성 기술 구현」 프로젝트.

- 사이트: https://rudals906377.github.io/IRIS/ (GitHub Pages 활성화 후 열림)
- **크롬 확장 프로그램**: 쇼핑몰 상세페이지에서 바로 입어 보기 → [설치·사용 방법](docs/extension-guide.md) · [설치 파일](release/iris-extension.zip)
- 조사·설계 문서: [docs/realtime-virtual-try-on-plan.md](docs/realtime-virtual-try-on-plan.md)
- 작업 인수인계: [docs/handoff.md](docs/handoff.md)

## 특징

- **모든 처리가 브라우저 안에서**: 카메라 영상은 기기 밖으로 전송하거나 저장하지 않습니다. 서버 비용도 없습니다.
- **마스크 기반 레이어 합성**: 신체 추적(MediaPipe) → 교체·보존·가림 마스크 → 상품 부위별 변형 → 앞뒤 순서 합성.
  - 머리카락·얼굴·손·팔이 옷 앞에 오면 옷을 가립니다(가이디드 필터로 머리카락 올 단위 경계).
  - 원래 입은 옷의 주름·그림자를 새 옷에 입혀 입체감을 냅니다.
  - 상품 픽셀을 그대로 쓰므로 색·무늬·로고가 바뀌지 않습니다.
- **아무 쇼핑몰 사진이나**: 상품 사진을 올리거나(끌어놓기) 확장 프로그램으로 상세페이지 사진을 고르면, 배경 제거·목/어깨/소매/밑단 위치 찾기·부위 구분을 브라우저 안에서 자동으로 해서 바로 입힙니다.
- **실시간성 측정 내장**: 단계별 처리 시간, 촬영→그리기 지연, 건너뛴 프레임, 1분 단위 추세, 거울 루프백 지연 측정, JSON 내보내기.

## 사용법

1. 사이트를 열고 아래 목록에서 상품을 고릅니다.
2. **카메라로 시작**을 누르고 카메라 권한을 허용합니다. 카메라가 없으면 **예시 영상으로 체험**.
3. 상반신(양 어깨)이 화면에 들어오게 서면 옷이 입혀집니다.
4. 잘 맞지 않으면 **설정 → 옷 맞춤**에서 어깨 폭·높이를 조정하세요.

잘 나오는 조건: 밝은 곳, 정면, 몸에 맞는 밝은 무지 반팔. 뒤돌기(60° 이상 회전)는 상품 사진에 뒷면 정보가 없어 옷이 서서히 사라집니다.

## 지연 측정 방법

| 방법 | 방법 요약 | 정확도 |
|---|---|---|
| 앱 안의 측정 표시 | 상단 **측정** 버튼 → 처리 시간·촬영→그리기 지연(브라우저가 지원할 때) | 소프트웨어 구간만 |
| 거울 루프백 | **설정 → 거울 루프백 측정**. 노트북 화면 앞에 거울을 들면 화면 깜빡임을 카메라가 보고 왕복 지연을 30회 잽니다 | 카메라+화면 구간(자동) |
| 슬로모션 촬영 | 휴대폰 240fps 슬로모션으로 손(또는 플래시)과 노트북 화면을 한 화면에 찍고, 실제 움직임과 화면 반영 사이 프레임 수 × 4.2ms | 전체 구간(가장 신뢰) |

측정 후 **측정 기록 내려받기(JSON)** 로 원자료를 저장할 수 있습니다.

## 개발

```bash
cd web
npm install
npm run dev        # http://localhost:5173 (카메라는 localhost 또는 HTTPS에서만 동작)
npm test           # 측정·필터 로직 단위 테스트
npm run build      # dist/ 생성
```

상품 이미지 다시 만들기(파이썬: numpy, pillow, opencv-python-headless):

```bash
python3 tools/garments/generate.py --out web/public/products
```

크롬 확장 프로그램 빌드(`web/dist-ext/`, `release/iris-extension.zip`):

```bash
cd web && npm run build:ext
```

헤드리스 브라우저 시험 도구는 [tools/harness](tools/harness/README.md), 상품 사진 분석 결과표는 [tools/analyze](tools/analyze/README.md).

주소 인자(개발·시험용): `?src=sample`(예시 영상으로 바로 시작), `?product=<상품 id>`, `?hud=1`, `?debug=lm,seg,occ`, `?pose=lite|full|heavy`, `?delegate=CPU|GPU`, `?img=<상품 사진 주소>`

### 구조

```
web/src/engine/
  source.ts     카메라·동영상 입력, 프레임 콜백
  tracker.ts    MediaPipe 자세·분할·손 추적
  body.ts       관절점 → 안정된 신체 좌표계(One Euro 필터, 추적 손실 처리)
  garment.ts    상품 이미지·부위 라벨 → 부위별 격자 메쉬
  rig-top.ts    상의 변형(몸판: 신체 좌표계, 소매: 팔 방향)
  occluders.ts  팔·손 가림 도형
  renderer.ts   WebGL2 합성(가림 버퍼, 가이디드 필터, 셰이딩 전이)
  metrics.ts    지연·처리 시간 측정
  loopback.ts   거울 루프백 지연 측정
web/src/app/    화면 구성
tools/garments/ 저작권 문제 없는 상품 이미지 생성기
```

## 배포

`main` 또는 개발 브랜치에 `web/`이 바뀌면 GitHub Actions가 빌드해 GitHub Pages에 올립니다.
처음 한 번은 저장소 **Settings → Pages → Build and deployment → Source**를 **GitHub Actions**로 바꿔야 합니다.

## 출처

- 예시 영상: [Intel IoT DevKit sample-videos](https://github.com/intel-iot-devkit/sample-videos) (CC BY 4.0), 앞 26초를 잘라 다시 인코딩
- 상품 이미지: IRIS 자체 제작(생성기 코드 포함)
- 추적·분할 모델: [MediaPipe](https://github.com/google-ai-edge/mediapipe) (Apache 2.0)
