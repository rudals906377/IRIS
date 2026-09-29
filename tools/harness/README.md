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
node tools/harness/fake-shop.mjs    # 나이키·아디다스 구조를 흉내 낸 시험 쇼핑몰(testdata/shop/{nike,adidas,model}.html)
node tools/harness/ext.mjs http://localhost:4173/testdata/shop/nike.html /tmp/person.y4m /tmp/ext.png
node tools/harness/asset.mjs /tmp/a.png testdata/human/00055_00.jpg testdata/cloth/09163_00.jpg   # 분석 결과(잘라 낸 옷·부위·기준점) 그림
```

가짜 웹캠(y4m) 만들기: `ffmpeg -loop 1 -t 2 -i 인물.jpg -vf "scale=576:768,fps=15" -pix_fmt yuv420p /tmp/person.y4m`
모델 착용 사진 시험 자료: IDM-VTON 예시 인물 사진을 `web/public/testdata/human/`에 둔다(`fetch-testdata.sh`).
`ext.mjs` 출력의 `selected`는 착용 창이 고른 사진, `fit`은 사진별 평가(good = 옷만 찍힌 사진)다.

디버그 설정(`settings`): `debugGarment` 1 부위 라벨 색, 2 통과 픽셀, 3 가림 버퍼, 4 가림 원인, 5 소매만 / `debugLandmarks` 관절점 / `debugSeg` 분할 / `debugOcc` 가림 영역.

클라우드 환경의 헤드리스 Chromium은 GPU가 없어 CPU·소프트웨어 렌더링으로 느리다(초당 1프레임 안팎). 속도는 반드시 실제 기기(맥북 브라우저)에서 앱의 **측정** 표시로 확인한다.
