.PHONY: help install lint test run-pipeline run-api run-dashboard docker-up docker-down clean

help:
	@echo "🛡️ CyberShield-BigData Automation Commands:"
	@echo "  make install        - Install all pinned dependencies"
	@echo "  make lint           - Run Flake8 and formatting checks"
	@echo "  make test           - Run PyTest distributed unit & integration suite"
	@echo "  make run-pipeline   - Run complete End-to-End MLOps pipeline"
	@echo "  make run-api        - Start FastAPI serving microservice"
	@echo "  make run-dashboard  - Start Streamlit SOC dashboard"
	@echo "  make docker-up      - Build and spin up containers via Docker Compose"
	@echo "  make docker-down    - Stop and remove all running containers"
	@echo "  make clean          - Remove temporary cache and pyc files"

install:
	pip install --upgrade pip
	pip install -r requirements.txt

lint:
	flake8 src serving configs --max-line-length=127 --statistics

test:
	pytest tests/ -v --cov=src --cov-report=term-missing

run-pipeline:
	python main.py --mode all

run-api:
	uvicorn serving.api.main:app --host 0.0.0.0 --port 8000 --reload

run-dashboard:
	streamlit run serving/dashboard/app.py --server.port 8501

docker-up:
	docker compose -f deployment/docker-compose.yml up --build -d

docker-down:
	docker compose -f deployment/docker-compose.yml down

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .coverage coverage.xml