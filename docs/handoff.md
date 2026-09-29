# 작업 인수인계 (다음 세션용)

작성: 2026-09-29. 이 문서는 새 세션이 이 프로젝트를 이어받아 **나이키·아디다스 공식 홈페이지 기준으로 완성**하기 위한 것이다.

## 1. 사용자와 약속된 방식

- **모든 대화·진행 안내·문서·커밋 메시지는 한국어**로 쓴다. 영어는 기술 용어에만 쓴다.
- 사용자는 이 분야를 잘 모르며 "알아서 해 달라"고 했다. 결정은 스스로 내리고, 사용자 손이 꼭 필요한 일(맥북에서 직접 써 보기, 설정 한 번 바꾸기)만 짧고 쉽게 요청한다.
- 목표는 "최고의 결과물". 교수님 평가는 경험 위주라 기술 선택은 자유.
- 조건: MacBook Air M5 16GB, 윈도우 NVIDIA GPU PC 있음(필요 시), 추가 예산 없음(유료 서버·클라우드 불가), 1인 개발, 결과물은 **공개 웹 + 크롬 확장 프로그램**.
- 공개 저장소이므로 **나이키·아디다스 상품 사진 파일은 절대 커밋하지 않는다**(저작권). 시험용 사진은 `web/public/testdata/`(gitignore)에만 둔다.

## 2. 지금까지 된 것

| 영역 | 상태 |
|---|---|
| 실시간 착용 엔진(상의) | 완료. 브라우저 안에서 MediaPipe 자세·분할 → 신체 좌표계 → 몸판·소매 변형 → WebGL 합성. 영상은 기기 밖으로 안 나감 |
| 마스크 합성 | 교체(옷), 보존(얼굴·머리카락·배경), 가림(손·팔), 목 안쪽(뒷깃) 부위, 가이디드 필터로 머리카락 올 단위 경계 |
| 착용 품질 | 가로·세로 배율 분리, 소매 4꼭짓점 곡면 모델(진동 둘레에 붙고 팔 바깥·안쪽 선을 따라감, 긴팔은 팔꿈치에서 굽음), 원래 옷 주름 음영 전이(인쇄 무늬는 걸러냄) |
| 측정 | 단계별 처리 시간, 촬영→그리기 지연, 건너뛴 프레임, 1분 추세, 거울 루프백 지연 측정, JSON 내보내기 |
| 상품 사진 자동 분석 | 배경 제거(경계 기반 flood fill) + 윤곽 분석으로 기준점·부위 라벨 + 비율 검증. 신뢰도 낮으면 MediaPipe 대화형 분할로 재시도, 그래도 낮으면 안내 문구 |
| 크롬 확장 프로그램 | 동작 확인(헤드리스 크롬에 설치해 가짜 쇼핑몰 → 후보 수집 → 착용 창 → 자동 분석 → 가짜 웹캠 착용). `release/iris-extension.zip`, 설치 안내 `docs/extension-guide.md` |
| 배포 | GitHub Actions → Pages. **사용자가 Settings → Pages → Source를 "GitHub Actions"로 바꿔야 첫 배포 성공**(아직 미확인) |

## 3. 구조

```
web/src/engine/
  engine.ts      전체 흐름(프레임마다 추적→변형→합성→측정), setProduct/setAsset
  tracker.ts     MediaPipe Pose + 다중 클래스 분할(확률 마스크) + 손(선택)
  body.ts        관절점 → 신체 좌표계(One Euro 필터, 추적 손실 시 서서히 사라짐)
  rig-top.ts     상의 변형(몸판 신체 좌표계, 소매 역쌍선형 곡면) — 핵심 로직
  occluders.ts   팔·손 캡슐(R 팔 중심, G 팔 주변 피부 판정, B 손)
  renderer.ts / shaders.ts   WebGL2 합성, 가이디드 필터, 디버그 표시
  garment.ts     상품 자산(이미지·라벨·기준점) → 부위별 격자 메쉬
  analyze/       상품 사진 자동 분석(background.ts, top.ts, index.ts)
web/src/app/main.ts   화면·설정·확장 프로그램 후보 목록·사진 업로드
web/extension/        manifest.json, background.ts(서비스 워커), icons
tools/garments/       자체 제작 상품 이미지 생성기(파이썬)
tools/harness/        헤드리스 시험 도구(README 참고)
tools/analyze/        상품 사진 분석 결과표 도구(README 참고)
```

부위 라벨 번호: 1 몸판, 2 착용자 왼쪽 소매(사진 오른쪽), 3 오른쪽 소매, 4 목 안쪽. 기준점 이름의 L/R은 **착용자 기준**.

## 4. 개발·시험 명령

```bash
cd web && npm ci && npm test && npm run build          # 단위 테스트, 사이트 빌드(dist/)
npm run build:ext                                       # 확장 프로그램(dist-ext/) + release/iris-extension.zip
npx vite preview --port 4173 &                          # 시험 서버
bash tools/harness/fetch-testdata.sh                    # 시험 자료
cd tools/harness && npm install                         # playwright
```

`tools/harness/README.md`, `tools/analyze/README.md`에 사용법이 있다. 헤드리스 크롬은 GPU가 없어 느리므로 **속도 판단은 하지 말고 모양만** 본다.

## 5. 다음 세션이 할 일 (우선순위 순)

1. **나이키·아디다스 접속 확인**: `curl -I https://www.nike.com/kr/`, `https://www.adidas.co.kr/`, `static.nike.com`, `assets.adidas.com`. 막히면 사용자에게 네트워크 허용(세션 제목 표시줄의 클라우드 환경 메뉴 → Edit → Network access)을 한국어로 짧게 요청.
2. **상세페이지 구조 조사**(상의 10개씩: 티셔츠·맨투맨·후드·긴팔·재킷):
   - 상품 사진 목록과 순서, 배경색, 옷만 찍힌 사진(아디다스 `_laydown`, 나이키 상품 컷)과 모델 착용 사진 구분법
   - `og:image`가 무엇을 가리키는지, 갤러리 DOM 구조, 지연 로딩(srcset·data-src) 여부
   - 이미지 CDN 주소 규칙과 고해상도 변환(`web/extension/background.ts`의 `upgradeImageUrl`을 실제에 맞게 수정)
   - 확장 프로그램 후보 수집(`collectProductImages`)이 옷만 찍힌 사진을 앞순위로 올리도록 개선(파일명·순서·배경 균일도 활용)
   - 조사 결과는 텍스트로 `docs/research/nike-adidas.md`에 기록(이미지 파일은 커밋 금지)
3. **자동 분석 정확도**: 두 사이트 상의 사진 20장 이상을 `web/public/testdata/`에 받아 `tools/analyze`로 결과표를 만들고, 실패 유형별로 `analyze/background.ts`·`top.ts` 개선. 목표: 옷만 찍힌 사진의 90% 이상이 신뢰도 0.6 이상 + 기준점이 눈으로 봐도 맞음.
   - 후드(모자 부분 때문에 목둘레 검출이 흔들림), 지퍼 재킷(앞이 열림), 오버핏(어깨가 떨어짐) 처리
   - 흰 옷·연회색 배경: 규칙 방식 실패 시 대화형 분할 재시도 경로를 브라우저에서 확인
4. **확장 프로그램 실전 시험**: `tools/harness/ext.mjs`로 실제 나이키·아디다스 상세페이지 URL에서 전체 흐름 확인(가짜 웹캠 인물). 결과 캡처를 보고 착용 품질 개선.
5. **모델 착용 사진 지원(선택)**: 사진 속 모델에 MediaPipe 자세 추정 → 분할의 '옷' 영역 → 모델 관절점을 기준점으로 삼아 사용자 몸으로 옮기기.
6. 완료 후: `npm test && npm run build && npm run build:ext` → `release/iris-extension.zip` 갱신 → 커밋·푸시(한국어 메시지) → 사용자에게 한국어로 결과와 "맥북 크롬에서 확장 프로그램 다시 불러오기" 방법 안내.

## 6. 알려진 문제·아이디어

- 겨드랑이 근처에 원래 옷이 가늘게 보이는 틈이 가끔 남음(팔 캡슐 반경·소매 안쪽 폭 조정 여지).
- 민소매·보디수트·끈 달린 옷은 자동 분석이 거절됨(의도). 민소매 겨드랑이 검출은 개선 여지.
- 하의·모자·장갑은 아직 없음(`docs/realtime-virtual-try-on-plan.md`의 설계 참고, 순서: 모자 → 하의 → 장갑).
- 실제 기기 지연 측정 결과가 아직 없음. 사용자가 맥북에서 **측정 → 설정 → 거울 루프백 측정**을 해 주면 판단 가능.
- 확장 프로그램 `background.ts`의 `irisTest`는 자동 시험용 훅(사용자 기능 영향 없음).
