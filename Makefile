.PHONY: dev test lint format typecheck migrate clean eval eval-gate setup-reranker-onnx install-onnx build-reranker-onnx bench-reranker-onnx

# ONNX 리랭커 양자화 타깃 아키텍처 (Apple Silicon=arm64 / 운영 x86 파드=avx512_vnni)
ARCH ?= arm64

# ─── 개발 서버 ───
dev:
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# ─── 테스트 ───
test:
	pytest

test-unit:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v

test-cov:
	pytest --cov=app --cov-report=html
	open htmlcov/index.html

# ─── 린트 & 포맷 ───
lint:
	ruff check app/ tests/
	ruff format --check app/ tests/

format:
	ruff check app/ tests/ --fix
	ruff format app/ tests/

typecheck:
	mypy app/

# ─── DB 마이그레이션 ───
migrate:
	alembic upgrade head

migrate-new:
	@read -p "Migration name: " name; \
	alembic revision --autogenerate -m "$$name"

migrate-down:
	alembic downgrade -1

# ─── 검색 평가 (실 Qdrant + OpenAI 임베딩 필요) ───
eval:
	python scripts/eval_retrieval.py --modes dense,rerank

eval-gate:
	python scripts/eval_retrieval.py --gate

# 생성 평가 (RAGAS LLM-judge, 비결정 → 비차단·nightly). 설치: uv pip install -e ".[eval]"
eval-gen:
	python scripts/eval_generation.py

# ─── 유틸 ───
clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage

# ─── Docker (로컬 개발) ───
up:
	docker compose up -d

down:
	docker compose down

logs:
	docker compose logs -f

# ─── 설치 ───
install:
	uv pip install -e ".[dev]"
	pre-commit install

# 리랭커(bge-reranker-v2-m3) 의존성 — CPU torch 휠 고정(CUDA 미설치, 이미지/디스크 경량화).
# Dockerfile과 동일 순서: ① CPU torch 선설치 → ② sentence-transformers(.[rerank]).
install-rerank:
	uv pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.5.1"
	uv pip install -e ".[rerank]"

# ─── ONNX 리랭커 백엔드 (#60, opt-in · 기본 backend는 torch) ───
# int8 양자화로 CPU 추론 경량화. 현 PoC는 .[onnx]로 빌드/실행 모두 가능
# (운영 의존성 분리 = optimum/torch 제거는 #72에서). 활성화: RERANKER_BACKEND=onnx + 산출물 경로.

# 최초 1회 셋업 — 의존성 설치 + 산출물 생성을 한 번에. 이후 활성화는 .env만.
# (아키텍처만 바꿔 재생성할 땐 make build-reranker-onnx ARCH=avx512_vnni 만 다시 실행)
setup-reranker-onnx: install-onnx build-reranker-onnx

# (1회성) ONNX 변환/추론 의존성 설치.
install-onnx:
	uv pip install -e ".[onnx]"

# (1회성/아키텍처별) bge-reranker-v2-m3 → ONNX fp32 → int8 양자화 산출물
# (models/bge-reranker-onnx-int8/model_quantized.onnx). 운영 x86 파드는 ARCH=avx512_vnni 로 재빌드.
build-reranker-onnx:
	python scripts/build_reranker_onnx.py --out models/bge-reranker-onnx-int8 --arch $(ARCH)

# torch vs ONNX(int8) 벤치 — 쿼리당 latency + 골든셋 hit@5/recall@5/mrr@10. (실 Qdrant + OpenAI 임베딩 필요)
bench-reranker-onnx:
	python scripts/bench_reranker_onnx.py --onnx-dir models/bge-reranker-onnx-int8

# ─── 도움말 ───
help:
	@echo ""
	@echo "  make dev              개발 서버 실행 (--reload)"
	@echo "  make test             전체 테스트"
	@echo "  make test-unit        단위 테스트만"
	@echo "  make test-cov         커버리지 리포트 생성"
	@echo "  make eval             검색 평가 점수표 (dense vs rerank)"
	@echo "  make eval-gate        baseline 대비 회귀 게이트"
	@echo "  make lint             린트 검사"
	@echo "  make format           자동 포맷 + 린트 수정"
	@echo "  make typecheck        mypy 타입 체크"
	@echo "  make migrate          DB 마이그레이션 적용"
	@echo "  make migrate-new      새 마이그레이션 생성"
	@echo "  make up               로컬 인프라 실행 (docker compose)"
	@echo "  make down             로컬 인프라 중지"
	@echo "  make install          의존성 + pre-commit 설치 (1회성)"
	@echo "  make install-rerank   리랭커 의존성(CPU torch + sentence-transformers) (1회성)"
	@echo "  make setup-reranker-onnx  ONNX 리랭커 셋업 = install-onnx + build (최초 1회)"
	@echo "  make install-onnx     ↳ ONNX 의존성만 설치 (1회성)"
	@echo "  make build-reranker-onnx  ↳ ONNX int8 산출물 생성 (ARCH=arm64|avx512_vnni)"
	@echo "  make bench-reranker-onnx  torch vs ONNX(int8) 속도·품질 벤치"
	@echo "  make clean            캐시 파일 정리"
	@echo ""