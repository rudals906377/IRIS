# 상품 사진 자동 분석 시험

브라우저와 같은 분석 코드(`web/src/engine/analyze`)를 노드에서 돌려, 배경 제거·기준점·부위 라벨 결과를 한 장의 결과표로 확인한다.

```bash
python3 tools/analyze/prep.py /tmp/an web/public/testdata/cloth/*.jpg   # 사진 준비(512px)
node tools/analyze/run.ts /tmp/an                                        # 분석(노드 22 이상)
python3 tools/analyze/render.py /tmp/an                                  # /tmp/an/sheet.jpg
```

결과표 색: 파랑 몸판, 빨강 착용자 왼쪽 소매(사진 오른쪽), 초록 오른쪽 소매, 노랑 목 안쪽. 분홍 점이 기준점.
