.PHONY: dev test lint format typecheck migrate seed clean eval eval-gate

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

# ─── 유틸 ───
seed:
	python scripts/seed_data.py

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
	@echo "  make install          의존성 + pre-commit 설치"
	@echo "  make install-rerank   리랭커 의존성(CPU torch + sentence-transformers)"
	@echo "  make clean            캐시 파일 정리"
	@echo ""