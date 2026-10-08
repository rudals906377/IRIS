# 평가 도구 (정확성·자연스러움을 숫자로)

고칠 때마다 같은 자료로 점수를 재서, 좋아졌는지 나빠졌는지 확인한다. 결과는 `tools/eval/results/*.json` 에 남는다.

## 준비 (처음 한 번)

```bash
bash tools/harness/fetch-testdata.sh          # 기본 시험 자료
# 웹캠 같은 영상(무료 Mixkit, 커밋 안 함) → web/public/testdata/webcam/  (받는 방법은 2026-10-08 개발일지)
python tools/eval/make_hair_gt.py --split eval --out web/public/testdata/eval/hair   # 머리카락 평가 정답 26장
python tools/eval/hair_refs.py                # 실제 머리카락 사진 32장의 색·명암 분포
cd web && npm run build && npx vite preview --port 4173 &
```

정답은 앱과 무관한 강한 오프라인 모델로 만든다(얼굴 파싱 SegFormer → ViTMatte 경계 정밀화 → BiRefNet 배경 제거).
평가에 쓴 사람은 학습 자료에서 뺀다(`make_hair_gt.py` 의 EVAL·EXCLUDE).

**주의**: 평가가 도는 동안 `npm run build` 하지 말 것(dist/testdata 가 잠깐 지워져 평가가 멈춘다).

## 평가 항목

| 도구 | 재는 것 | 점수(낮을수록 좋음, IoU 만 높을수록) |
|---|---|---|
| `node tools/eval/hair.mjs <이름> '<주소 매개변수>'` | 머리카락 영역 | IoU, MAE, 경계MAE, 번짐(이마·배경에 칠해짐), 빠짐 |
| `node tools/eval/hair.mjs ... --flicker` | 프레임 간 떨림 | 연속 프레임 확률 변화 |
| `node tools/eval/haircolor.mjs <이름> '<매개변수>'` | 염색 색·결 | 색오차(ΔE), 결오차(밝기 변동계수 로그 비), 폭오차(그늘~윤기 폭) |
| `node tools/eval/makeup.mjs <이름> <얼굴 영상> 21` | 사진 따라하기 색 | 입술, 볼앞·볼중간, 섀도바깥·섀도안(피부 대비 색조 거리) |

매개변수 예: `'{"hairmodel":"models/hair-matte-256.onnx"}'`(자체 머리카락 모델), `'{"covgain":0.5}'`(염색 명암 계수 시험).
`--save 폴더` 를 붙이면 장면별 그림(원본 | 정답 | 예측, 또는 염색 결과)을 남긴다.

## 기록

| 날짜 | 항목 | 이전 → 이후 |
|---|---|---|
| 2026-10-08 | 염색 결오차·폭오차(명암 분포 맞추기, 계수 0.6) | 0.51·0.51 → 0.31·0.31 (색오차 5.1 → 5.7) |
| 2026-10-08 | 사진 따라하기(채널별 되먹임 4회) | 볼앞 0.033→0.028, 볼중간 0.044→0.041, 섀도바깥 0.055→0.047, 섀도안 0.064→0.056 |
| 2026-10-08 | 머리카락 경계(가이디드 필터 반경 6→4) | IoU 0.881→0.884, 경계 0.149→0.143, 떨림 0.023→0.025 |
| 2026-10-08 | 자체 머리카락 모델 v2 → v3(웹캠형 라벨 섞어 재학습) | IoU 0.534→0.789, 경계 0.294→0.179 |
