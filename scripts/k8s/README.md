# K8s 적재 런북 (one-off Jobs)

쿠버네티스(dev `onramp` 네임스페이스)에서 **데이터 적재를 처음부터 다시** 수행하는 일회성 Job 모음.
ArgoCD 비관리(차트 밖) — 필요할 때 `kubectl apply`로 직접 실행한다.

## 실행 순서

```bash
# 1) 데이터 초기화 (OpenSearch 청크/문서 인덱스 DELETE, Qdrant 컬렉션 DELETE, Postgres TRUNCATE)
kubectl apply -f scripts/k8s/wipe-stores-job.yaml
kubectl logs -n onramp job/onramp-wipe-stores -f

# 2) 멀티소스 재적재 (Confluence 전체 + GitHub + 문서 투영, 문서 단위 LLM 도메인 분류)
kubectl apply -f scripts/k8s/ingest-job.yaml
kubectl logs -n onramp job/onramp-ingest -f        # ~20분

# 3) 검증 (3중 저장소 카운트 정합 + domain_source/domain 분포)
kubectl apply -f scripts/k8s/verify-job.yaml
kubectl logs -n onramp job/onramp-verify -f
```

스토어를 비운 직후 적재하므로 `--reindex`는 불필요하다(내용 미변경 dedup 스킵 없음).
스키마는 TRUNCATE라 유지되며, 마이그레이션은 ArgoCD sync 훅(`alembic upgrade head`)이 관리한다.

## ⚠️ 적용 전 확인

- **이미지**: 세 Job 모두 `image:`에 특정 digest가 핀돼 있다. **현재 배포 이미지로 갱신**할 것:
  ```bash
  kubectl get deploy onramp-api -n onramp -o jsonpath='{.spec.template.spec.containers[0].image}'
  ```
  (옛 이미지로 적재하면 분류/스키마가 옛 코드 기준이 된다.)
- **시크릿**: `onramp-api-secret`에 `OPENAI_API_KEY`(임베딩+LLM 분류), `CONFLUENCE_*`, `GITHUB_TOKEN`(GitHub 적재 시)이 있어야 한다.
- **`LANGFUSE_ENABLED=false` override**: Job은 langfuse가 불필요한데, configmap의 `LANGFUSE_ENABLED=true`를
  끄지 않으면 config 검증(`_check_langfuse`)이 키 부재로 fail-fast 한다. 그래서 각 Job env에 명시적으로 false를 준다.

## Job별 메모

| Job | 동작 | 비고 |
|---|---|---|
| `wipe-stores-job` | OS/Qdrant DELETE + PG TRUNCATE | 파괴적·되돌릴 수 없음. dev 데이터는 소스에서 재생성 가능 |
| `ingest-job` | Confluence(`--limit 2000`) + GitHub 10 repo + 문서 투영 | `LLM_CLASSIFY_ENABLED=true`, `activeDeadlineSeconds: 3600` |
| `verify-job` | `ingest_status.py` + OpenSearch 집계 | 읽기 전용 |

모든 Job은 `ttlSecondsAfterFinished`로 완료 후 자동 삭제된다.
