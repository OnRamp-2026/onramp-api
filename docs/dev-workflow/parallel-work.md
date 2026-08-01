# 병렬 작업 (Git Worktree)

> 독립적인 작업을 동시에 진행할 때. 각 작업이 별도 디렉터리(checkout)를 가져 파일 충돌이 없다.

## Worktree로 병렬
```bash
# 기능별 독립 워크트리 생성
git worktree add ../feat-auth feature/auth
git worktree add ../feat-ui   feature/ui

# 각 워크트리에서 별도 세션 실행 (동시)
cd ../feat-auth && claude
cd ../feat-ui   && claude
```
- 각 워크트리는 독립 checkout → 동시 작업해도 충돌 없음.
- 정리: `git worktree remove ../feat-auth` (변경 없으면).

## 서브에이전트 병렬
- 대규모 분석/수정은 기능별 서브에이전트(`.claude/agents/`)에 나눠 병렬 실행.
- 파일을 병렬로 고쳐야 하면 서브에이전트에 **worktree 격리**를 줘 충돌 방지.

## 주의 (통제 우선)
- 병렬은 **독립적인 작업**에만. 서로 얽힌 변경은 순차로.
- 각 세션도 계획 승인·검증 루프를 똑같이 지킨다(병렬이라고 검증 건너뛰지 않기).
- push는 각 브랜치별로 사람이 승인.
