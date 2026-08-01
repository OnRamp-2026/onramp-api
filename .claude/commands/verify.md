---
description: 테스트·타입체크·lint(+RAG 지표)를 실행하고 결과를 요약한다.
---

다음을 순서대로 실행하고 결과를 요약하라:

1. `make test` — pytest
2. `make typecheck` — mypy `app/`
3. `make lint` — ruff check + format --check
4. **리트리버·청킹·리랭커·프롬프트를 건드렸으면** `make eval-gate` — 골든셋 지표 게이트 (실 Qdrant + OpenAI 임베딩 필요. 환경이 없으면 실행 불가라고 명시하고 넘어간다)

## 규칙
- 실패가 있으면 **원인과 수정안을 제시**만 한다. 자동 수정은 사람 승인 후(`make format`은 diff를 바꾸므로 승인 필요).
- **실패를 통과라고 말하지 않는다.** 실행 못 한 항목은 "미실행"이라고 그대로 적는다.
- 전부 통과하면 커밋 가능 상태임을 알린다(→ `/commit`).
- 검증 상세: `docs/dev-workflow/verification-loop.md`.
