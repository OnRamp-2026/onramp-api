# on-demand GPU 리랭커 (VESSL) 운영 스크립트 (#73)

VESSL L40S 리랭커를 **쓸 때만 켜고 끄는** on-demand 운영용. URL이 스핀업마다 바뀌므로
onramp-api는 Redis 키 `reranker:service_url`에서 URL을 런타임 조회한다(없으면 vector 폴백).

## 사용
```bash
./up.sh      # VESSL 생성 → uvicorn 자동기동 → URL 파싱 → /health/ready → Redis SET
# ... 질의/평가 실행 (onramp-api가 ~30s 내 GPU 리랭킹 반영) ...
./down.sh    # Redis DEL(→vector 폴백) → VESSL terminate(과금 중단)
```

`set-url.sh <url>` / `clear.sh` 는 Redis만 만지는 primitive(수동 갱신용). `up`/`down`이 내부에서 호출.

## 전제
- `vesslctl` 로그인: 토큰 만료 시 `up`/`down`이 **자동으로 `vesslctl auth login`**(브라우저 OAuth)을 띄움.
- `kubectl` 컨텍스트가 대상 클러스터(dev, ns `onramp`)를 가리킬 것.
- `jq` 설치.
- onramp-api는 `RERANKER_BACKEND=remote` + `RERANKER_SERVICE_URL`은 빈값(URL은 Redis가 공급).

## 기본값 (환경변수로 override)
| 변수 | 기본값 |
|---|---|
| `VESSL_WS_NAME` | `onramp-reranker` |
| `VESSL_CLUSTER` | `cluster-betelgeuse` |
| `VESSL_SPEC` | `resourcespec-eaac8rxxpz6j` (L40S ×1, $1.80/hr) |
| `VESSL_IMAGE` | `ghcr.io/onramp-2026/onramp-reranker:gpu-v1` |
| `VESSL_SSHKEY` | `sshkey-umdldri4c2ha` |
| `RERANKER_REDIS_NS` / `_WORKLOAD` / `_KEY` | `onramp` / `deploy/redis` / `reranker:service_url` |

## 첫 실행 주의 (URL 파싱)
`workspace show -o json`의 노출 URL 필드는 환경에 따라 다를 수 있다. `up.sh`는 `https://*.vessl.ai`
엔드포인트를 재귀 스캔(8080/reranker 우선)하며, 못 찾으면 **원본 JSON을 출력**한다.
그 경우 JSON에서 8080 https 엔드포인트를 확인해 `RERANKER_URL_OVERRIDE=<url> ./up.sh` 로 재실행하면 된다.

## 안전장치
- GPU OFF/장애여도 onramp-api는 **서킷브레이커 → vector 폴백**으로 죽지 않는다(#123).
- `down.sh`는 **Redis 정리를 먼저** 하고 terminate — 죽은 URL을 더 호출하지 않게.
