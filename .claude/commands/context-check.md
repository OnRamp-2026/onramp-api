---
description: git 상태·최근 커밋·검증/인수인계 리마인드를 보여준다.
---

!`git status --short`
!`git log --oneline -5`

현재 작업 상태를 한 줄로 요약하고, 다음을 리마인드하라:
- 검증(테스트/타입체크/lint) 했는가 → 안 했으면 `/verify`
- 세션이 길면 인수인계 → `/handoff`
- 커밋할 게 있으면 → `/commit` (push는 사람 승인 후)
- 컨텍스트가 무거우면 정리(→ `docs/dev-workflow/context-and-cost.md`)
