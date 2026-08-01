---
description: 현재 브랜치의 변경으로 PR 초안을 만든다(gh). 생성 전 사람 확인 필수.
---

현재 브랜치:
!`git branch --show-current`

main과의 차이:
!`git diff main...HEAD --stat 2>/dev/null || echo "(main 브랜치 확인 필요)"`

위 변경으로 PR을 만들어라. 형식은 `GROUND_RULES.md` §5를 따른다.

## 제목
```
<타입>: <설명> (#N)
```

## 본문
```markdown
## 변경 사항
- (무엇을 바꿨는지)

## 작업 이유
- (왜 필요한지 — 이슈 컨텍스트로 충분하면 생략 가능)

## 확인 방법
- (리뷰어가 어떻게 테스트해볼 수 있는지 — 실행할 명령 포함)

Close #N
```

## 규칙
- **먼저 사람에게 제목·본문을 보여주고 확인받은 뒤에만** `gh pr create` 실행.
- `Close #N` 누락 금지. 이슈번호는 브랜치명에서, 없으면 사람에게 묻는다.
- 검증(`make test`/`make typecheck`/`make lint`) 결과를 "확인 방법"에 사실대로 적는다. 안 돌렸으면 안 돌렸다고 쓴다.
- 머지 전략은 **Squash and Merge**. 머지 후 `feat/#N` 브랜치는 로컬·원격 모두 삭제.
- `gh` 미설치/미인증이면 명령만 안내하고 멈춘다.
