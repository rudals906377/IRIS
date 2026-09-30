# 헤드리스 브라우저 시험 도구

실제 웹캠 없이, 인물 영상·사진을 입력으로 넣어 메이크업 결과를 캡처하고 내부 수치를 확인한다.

```bash
bash tools/harness/fetch-testdata.sh            # 시험 자료(인물 영상·얼굴 확대 영상) → web/public/testdata (커밋 안 됨)
cd tools/harness && npm install && cd -
cd web && npm run build && npx vite preview --port 4173 &   # 시험 대상 서버

# 영상·룩 조합을 차례로 캡처(crop: true면 얼굴만 잘라 저장)
node tools/harness/batch.mjs /tmp/b '[{"src":"testdata/face/00055_00.webm","look":"rose"},{"src":"testdata/female.webm","look":"red","crop":true}]'
# 같은 프레임에서 설정만 바꿔 비교
node tools/harness/freeze.mjs 'src=testdata/female.webm&look=daily&delegate=CPU' /tmp/fz 12 '[{"refine":false},{"refine":true}]'
```

**커밋 전 스모크 시험**: 모든 효과를 켜고 셰이더 컴파일 오류·예외가 없는지 확인한다(셰이더 오류는 빌드·단위 테스트로 안 잡힌다).

```bash
node tools/harness/smoke.mjs     # 실패하면 종료 코드 1
```

시험 얼굴 영상(`testdata/face/`)에는 밝은·중간·어두운 피부 톤이 모두 있다(00034·00055·01992·sam1·00035·00121). 색 관련 변경은 여러 톤에서 함께 확인한다.

주의: 미리보기 서버는 `web/dist`를 보여 주므로 `testdata`에 파일을 새로 넣었으면 다시 빌드하거나 `web/dist/testdata`에 복사한다.

디버그 설정(`settings`): `debugLandmarks` 얼굴 점 / `debugSeg` 분할(빨강 머리카락, 초록 몸 피부, 파랑 얼굴 피부).

클라우드 환경의 헤드리스 Chromium은 GPU가 없어 CPU·소프트웨어 렌더링으로 느리다(초당 1프레임 안팎). 속도는 반드시 실제 기기(맥북 브라우저)에서 앱의 **측정** 표시로 확인한다.

## 얼굴 점 떨림·지연 측정(jitter/)

정답을 아는 합성 영상(얼굴 사진을 알려진 이동·회전·크기로 움직임)으로 떨림 필터를 비교한다. 원본 점에 웹캠과 비슷한 잡음(얼굴 전체 + 점별)을 섞어 같은 조건에서 잰다.

```bash
python3 tools/harness/jitter/make_motion.py web/public/testdata/face/01992_00.png /tmp/jit && cp /tmp/jit/motion.webm web/dist/testdata/
node tools/harness/jitter/record.mjs /tmp/jit                                   # 프레임별 원본 얼굴 점
cd web && node --experimental-strip-types ../tools/harness/jitter/bench.ts /tmp/jit 0.5 0.4   # 머리 움직임
node --experimental-strip-types ../tools/harness/jitter/talk.ts /tmp/jit 12                    # 말하기(입 벌림)
```

2026-09-30 결과(잡음 0.5/0.4px): 이전 점별 One Euro → 앱의 2단 필터 — 정지 떨림 0.18 → 0.06px/프레임, 빠른 흔들기 오차 16.7 → 4.2px, 급정지 후 2.6 → 1.6px, 말할 때 입술 오차 3.0 → 1.2px.

손(네일)도 같은 방식: MediaPipe 예시 사진(`storage.googleapis.com/mediapipe-tasks/hand_landmarker/woman_hands.jpg`, 시험 전용)으로 합성 영상을 만들고 손 점을 기록해(`?nail=0`으로 손 추적을 켠 뒤 `r.hands.landmarks`) `jitter/hands.ts`로 비교한다. 결과(잡음 0.6/0.6px): 이전 점별 One Euro → 앱 묶음 필터 — 빠른 흔들기 오차 11.4 → 2.5px, 정지 떨림 0.30 → 0.13px, 급정지 후 1.8 → 1.2px.
