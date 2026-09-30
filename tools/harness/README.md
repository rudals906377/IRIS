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

주의: 미리보기 서버는 `web/dist`를 보여 주므로 `testdata`에 파일을 새로 넣었으면 다시 빌드하거나 `web/dist/testdata`에 복사한다.

디버그 설정(`settings`): `debugLandmarks` 얼굴 점 / `debugSeg` 분할(빨강 머리카락, 초록 몸 피부, 파랑 얼굴 피부).

클라우드 환경의 헤드리스 Chromium은 GPU가 없어 CPU·소프트웨어 렌더링으로 느리다(초당 1프레임 안팎). 속도는 반드시 실제 기기(맥북 브라우저)에서 앱의 **측정** 표시로 확인한다.
