# OnRamp — 개발 Ground Rule

> **팀**: Cortex (cloud_3팀) · 양정우 PM · 김현문 PL · 민지홍 TL · 배민 · 박지현
> **레포**: `OnRamp-2026/onramp-api` · `confluence-data-crawler` · `infra`
> **버전**: v1.0 (2026-05-28)

---

## 1. 브랜치 전략

### 1-1. 기본 원칙
- **단일 메인 브랜치** = `main`
- `develop` / `release` 브랜치 **없음** 
- 모든 작업은 **이슈 단위 feature 브랜치**에서 진행

### 1-2. 브랜치 명명
```
feat/#<이슈번호>              → 신규 기능
fix/#<이슈번호>               → 버그 수정
chore/#<이슈번호>             → 환경설정·문서·리팩터
```

예시:
- `feat/#1` — Confluence 변경분 수집 스크립트
- `fix/#7` — Markdown 변환 인코딩 오류
- `chore/#3` — Makefile 추가

### 1-3. 보호 규칙 (권장 — Github Settings에서 설정)
- `main` 브랜치 **직접 push 금지** → PR 필수
- PR **최소 1명 리뷰 승인** 후 merge
- merge 후 **feature 브랜치 자동 삭제**

---

## 2. 작업 흐름 (Issue → PR → Merge → Delete)

```
[1] Github Issue 생성 (#N)
       ↓
[2] feat/#N 브랜치 생성 (local + remote push)
       ↓
[3] 개발 + 커밋 (작업 단위로 여러 커밋 OK)
       ↓
[4] 원격 push + PR 작성 (이슈 #N 자동 연결)
       ↓
[5] 리뷰 (1명 이상 승인) (선택)
- 코드 변경량이 많을 경우 승인이 필요하다고 판단할 것.
       ↓
[6] main 머지 (Squash and Merge 필수)
       ↓
[7] feat/#N 브랜치 즉시 삭제 (local + remote)
       ↓
[8] 이슈 PR close
```

### 2-1. 명령어 치트시트
```bash
# 1) issue 양식에 맞춰서 생성
gh issue create --title --body # 위의 양식을 참조하여 구성할 것.

# 2) main 최신화
git checkout main && git pull

# 3) feat 브랜치 생성
git checkout -b feat/#1

# 4) 작업·커밋
git add . && git commit -m "feat: Confluence 변경분 수집 스크립트 추가 (#1)"

# 5) main branch로 rebase
git pull origin main
git rebase main feat/#1

# 6) push + PR
git push -u origin feat/#1 # 기존에 브랜치 push를 해놨으면 --force 옵션으로 force push
gh pr create --title "feat: Confluence 변경분 수집 (#1)" --body "Close #1"

# 7) (머지 후) 로컬 브랜치 삭제
git checkout main && git pull
git branch -d feat/#1
git push origin --delete feat/#1   # 원격도 삭제 (자동 삭제 설정 안 했을 경우)
```

---

## 3. 이슈 작성 규칙

### 3-1. 이슈 제목
```
[<유형>] <간단한 한 줄 요약>
```
예: `[feat] Confluence 변경분 수집 스크립트`

### 3-2. 이슈 본문 템플릿
```markdown
## 배경 / 목적
(왜 필요한지 1~2줄)

## 작업 항목
- [ ]
- [ ]

## 완료 조건
- (어떻게 되면 끝인지)
```

### 3-3. 라벨 (Github Labels)
| 라벨 | 의미 |
|---|---|
| `feat` | 신규 기능 |
| `fix` | 버그 수정 |
| `chore` | 환경설정·문서·리팩터 |
| `track-A` | 인프라 |
| `track-B` | 수집·인덱싱 |
| `track-C` | RAG 코어 |
| `track-D` | Orchestration·자산화 |
| `track-E` | Frontend |
| `priority-P0` / `P1` / `P2` | 우선순위 |
| `blocker` | 다른 작업 차단 중 |

---

## 4. 커밋 메시지 규칙

### 4-1. 형식 (Conventional Commits)
```
<타입>: <간단한 설명> (#이슈번호)

[본문 — 선택]
[푸터 — 선택, 예: Close #N]
```

### 4-2. 타입
- `feat` — 신규 기능
- `fix` — 버그 수정
- `docs` — 문서
- `chore` — 환경·설정

### 4-3. 예시
```
feat: Confluence Webhook 핸들러 추가 (#1)
fix: HTML→MD 변환 시 UTF-8 인코딩 오류 (#7)
chore: Makefile에 lint 타깃 추가 (#3)
```

### 4-4. 금지
- ❌ `Co-Authored-By` 트레일러 **절대 금지** (자동 생성기 끄기) (클로드 한정)
- ❌ `--no-verify` (hook 우회) 금지
- ❌ `--amend` (이미 push된 커밋) 금지
- ❌ `git push --force` (main) 금지

---

## 5. PR (Pull Request) 규칙

### 5-1. PR 제목
```
<타입>: <설명> (#N)
```
예: `feat: Confluence 변경분 수집 스크립트 (#1)`

### 5-2. PR 본문 템플릿
```markdown
## 변경 사항
- (무엇을 바꿨는지)

## 작업 이유
- (왜 이 변경이 필요한지 — 이슈 컨텍스트로 충분하면 생략 가능)

## 확인 방법
- (리뷰어가 어떻게 테스트해볼 수 있는지)

Close #N
```

### 5-3. 머지 전략
- **Squash and Merge** 기본 (커밋 히스토리 깔끔)
- main에는 PR 1개 = 커밋 1개

### 5-4. 리뷰 정책
- **최소 1명 승인 필수**

---

## 6. 레포별 작업 매핑

| 레포                        | 트랙               | 주 담당       | 보조                   |
| ------------------------- | ---------------- | ---------- | -------------------- |
| `infra`                   | 🟣 A 인프라         | 박지현        | 김현문 (PL)             |
| `confluence-data-crawler` | 🔵 B 수집·인덱싱      | 김현문 PL     | 양정우 PM               |
| `onramp-api`              | 🟢 C·D RAG·Agent | 양정우 (C 코어) | 배민·민지홍 (D LangGraph) |

### 6-1. Cross-repo 작업
- 한 PR이 여러 레포에 걸칠 경우 → 각 레포에 별도 PR + 본문에 상호 링크
- 의존성 있을 경우 PR 본문 명시: `Depends on OnRamp-2026/infra#5`

---

## 7. 환경 변수 / 시크릿

- `.env` 파일 **절대 commit 금지** (`.gitignore` 확인 필수)
- 슬랙 채널에 ENV 파일 공유 (김현문 PL 관리)
- API Key (Confluence·OpenAI) — 본인 발급 권장, 공유 키는 보조
- Github Secret = 수동 매뉴얼 관리 (Jenkins/CI용)

---

## 8. Repo 공개 정책

| 시점           | 정책                         |
| ------------ | -------------------------- |
| 개발 중         | **Public** (일부 레포 private) |
| 5/29 중간 발표 후 | Public 유지                  |
| 6/25 최종 발표 후 | **Private 전환**             |

---

## 9. 위반 / 예외 대응

- Ground Rule 위반 발견 시 → PR 코멘트 / 슬랙 가이드
- 룰 변경 필요 시 → 회의 안건으로 올림 → 합의 시 v1.x 갱신

---

## 10. 체크리스트 (작업 시작 전 / PR 작성 전)

### 작업 시작 전
- [ ] main 최신 pull 받았는가
- [ ] 이슈 #N 생성했는가
- [ ] feat/#N 브랜치 생성했는가
- [ ] 트랙 라벨 / 우선순위 라벨 붙였는가

### PR 작성 전
- [ ] 로컬 테스트 통과
- [ ] `.env` 등 시크릿 파일 commit 안 됐는가
- [ ] 커밋 메시지 규칙 준수 (Co-Authored-By 없음(claude 한정))
- [ ] PR 본문에 `Close #N` 포함

### 머지 후
- [ ] feat/#N 브랜치 로컬·원격 삭제
- [ ] 이슈 자동 close 확인
- [ ] 다음 작업 main에서 pull

---