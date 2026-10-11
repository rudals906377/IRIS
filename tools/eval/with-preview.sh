#!/bin/bash
# 미리보기 서버를 같은 프로세스 묶음 안에서 띄우고 명령을 실행한 뒤 끈다
cd "$(dirname "$0")/../../web"
if ! curl -s -o /dev/null http://localhost:4173/; then
  npx vite preview --port 4173 --strictPort > /dev/null 2>&1 &
  PV=$!
  for i in $(seq 1 30); do curl -s -o /dev/null http://localhost:4173/ && break; sleep 1; done
fi
cd "$(dirname "$0")/../.."
"$@"
RC=$?
[ -n "$PV" ] && kill $PV 2>/dev/null
exit $RC
