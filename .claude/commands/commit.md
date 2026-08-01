---
description: 스테이징된 변경을 팀 커밋 규칙에 맞게 커밋한다. push는 하지 않는다.
---

현재 브랜치:
!`git branch --show-current`

스테이징된 변경:
!`git diff --staged --stat`

스테이징 안 된 변경:
!`git status --short`

위 변경을 분석해 커밋 메시지를 작성하고 `git commit` 하라.

## 형식 (`GROUND_RULES.md` §4)
```
<타입>: <간단한 설명> (#이슈번호)
```
- 타입: `feat` / `fix` / `docs` / `chore`
- 이슈번호는 브랜치명(`feat/#N`)에서 가져온다. 브랜치에서 못 찾으면 **사람에게 묻는다**(임의 생략 금지).

## 규칙
- **`Co-Authored-By` 트레일러 절대 금지.** 기본으로 붙이지 말 것 — 팀 규칙 위반이다.
- 커밋 전 검증(`make test` / `make typecheck` / `make lint`, RAG 변경이면 `make eval-gate`)이 안 됐으면 먼저 알리고 멈춘다.
- `--no-verify` 금지. 이미 push된 커밋에 `--amend` 금지.
- `.env` 등 시크릿이 스테이징돼 있으면 커밋하지 말고 경고한다.
- **push는 하지 않는다.** push/PR은 사람 승인 후.
