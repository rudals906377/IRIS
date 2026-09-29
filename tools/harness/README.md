# 헤드리스 브라우저 시험 도구

실제 웹캠 없이, 인물 영상·사진을 입력으로 넣어 착용 결과를 캡처하고 내부 수치를 확인한다.

```bash
bash tools/harness/fetch-testdata.sh            # 시험 자료(인물 영상·상품 사진) → web/public/testdata (커밋 안 됨)
cd tools/harness && npm install && cd -
cd web && npm run build && npx vite preview --port 4173 &   # 시험 대상 서버

node tools/harness/batch.mjs /tmp/b '[{"src":"testdata/still_00121_00.webm","product":"tee-stripe"},{"src":"testdata/still_00055_00.webm","img":"testdata/cloth/09163_00.jpg"}]'
node tools/harness/freeze.mjs 'src=testdata/female.webm&product=tee-navy-logo&delegate=CPU' /tmp/fz 12 '[{"refine":false},{"refine":true}]'
node tools/harness/rigprobe.mjs 'src=testdata/still_00121_00.webm&product=tee-stripe'
node tools/harness/ext.mjs https://www.nike.com/kr/t/<상품> /tmp/person.y4m /tmp/ext.png   # 확장 프로그램 전체 흐름
```

디버그 설정(`settings`): `debugGarment` 1 부위 라벨 색, 2 통과 픽셀, 3 가림 버퍼, 4 가림 원인, 5 소매만 / `debugLandmarks` 관절점 / `debugSeg` 분할 / `debugOcc` 가림 영역.

클라우드 환경의 헤드리스 Chromium은 GPU가 없어 CPU·소프트웨어 렌더링으로 느리다(초당 1프레임 안팎). 속도는 반드시 실제 기기(맥북 브라우저)에서 앱의 **측정** 표시로 확인한다.
