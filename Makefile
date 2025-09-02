# Batch Email Generator Docker Commands

.PHONY: help build up down restart logs status clean test backup

# Default target
help:
	@echo "Batch Email Generator - Docker Commands"
	@echo ""
	@echo "Basic Commands:"
	@echo "  make up              - Start all services"
	@echo "  make up-alpine       - Start with Alpine Linux + Frontend (RECOMMENDED)"
	@echo "  make up-ubuntu       - Start with Ubuntu repositories + Frontend"
	@echo "  make up-frontend     - Add frontend to any existing Docker setup"
	@echo "  make up-prebuilt     - Start without Docker build (fallback)"
	@echo "  make up-robust       - Start with network-resilient build"
	@echo "  make up-network-fix  - Start with BuildKit disabled"
	@echo "  make up-simple       - Start with simple single-stage build"
	@echo "  make up-fresh        - Fresh start with cache cleanup"
	@echo "  make down            - Stop all services" 
	@echo "  make restart         - Restart all services"
	@echo "  make logs            - View application logs"
	@echo "  make status          - Check service status"
	@echo ""
	@echo "Development:"
	@echo "  make dev         - Start in development mode (live reload)"
	@echo "  make dev-logs    - View development logs"
	@echo "  make serve-ui    - Start Python frontend server (for local dev without Docker)"
	@echo ""
	@echo "Database:"
	@echo "  make db-logs     - View database logs"
	@echo "  make db-connect  - Connect to database"
	@echo "  make db-reset    - Reset database (DESTRUCTIVE!)"
	@echo ""
	@echo "Testing:"
	@echo "  make test           - Run comprehensive test suite"
	@echo "  make test-quick     - Quick API health tests"
	@echo "  make test-data      - Generate test data"
	@echo "  make test-email     - Test email generation"
	@echo "  make test-db        - Test database connectivity"
	@echo "  make test-performance - Performance test with 50 emails"
	@echo ""
	@echo "Maintenance:"
	@echo "  make build       - Rebuild application"
	@echo "  make clean       - Clean up containers and volumes"
	@echo "  make fix         - Fix Docker build issues (troubleshooter)"
	@echo "  make backup      - Backup database"
	@echo ""
	@echo "Admin Tools:"
	@echo "  make admin       - Start with PgAdmin"
	@echo "  make full        - Start with all optional services"

# Basic operations
up:
	@echo "Starting Email Generator services..."
	docker-compose -f docker/docker-compose.yml up -d
	@echo "[OK] Services started! API: http://localhost:8000"

up-simple:
	@echo "Starting with simple build..."
	docker-compose -f docker/docker-compose.yml -f docker/docker-compose.simple.yml up -d
	@echo "[OK] Simple build started! API: http://localhost:8000"

up-prebuilt:
	@echo "Starting with pre-built approach (no Docker build)..."
	docker-compose -f docker/docker-compose.prebuilt.yml up -d
	@echo "[OK] Pre-built setup started! API: http://localhost:8000"

up-fresh:
	@echo "Fresh start with cache cleanup..."
	docker-compose -f docker/docker-compose.yml down -v
	docker system prune -a -f
	docker-compose -f docker/docker-compose.yml build --no-cache --pull
	docker-compose -f docker/docker-compose.yml up -d
	@echo "[OK] Fresh build completed! API: http://localhost:8000"

up-robust:
	@echo "Starting with robust network-resilient build..."
	docker-compose -f docker/docker-compose.yml -f docker/docker-compose.robust.yml up -d --build
	@echo "[OK] Robust build completed! API: http://localhost:8000"

up-network-fix:
	@echo "Starting with network fixes (disabling BuildKit)..."
	@export DOCKER_BUILDKIT=0 && docker-compose -f docker/docker-compose.yml build --no-cache app
	@docker-compose -f docker/docker-compose.yml up -d
	@echo "[OK] Network-fixed build completed! API: http://localhost:8000"

up-alpine:
	@echo "Starting with Alpine Linux + Frontend (bypasses Debian issues)..."
	docker-compose -f docker/docker-compose.alpine.yml --env-file .env up -d --build
	@echo "[OK] Alpine build completed! API: http://localhost:8000 | Frontend: http://localhost:3000"

up-ubuntu:
	@echo "Starting with Ubuntu repositories + Frontend (stable alternative)..."
	docker-compose -f docker/docker-compose.yml -f docker/docker-compose.ubuntu.yml -f docker/docker-compose.frontend.yml up -d --build
	@echo "[OK] Ubuntu build completed! API: http://localhost:8000 | Frontend: http://localhost:3000"

up-frontend:
	@echo "Adding frontend to existing Docker setup..."
	docker-compose -f docker/docker-compose.yml -f docker/docker-compose.frontend.yml up -d --build frontend
	@echo "[OK] Frontend added! Web UI: http://localhost:3000"

down:
	@echo "Stopping Email Generator services..."
	docker-compose -f docker/docker-compose.yml --env-file .env down

restart: down up
	@echo "Services restarted!"

# Development mode
dev:
	@echo "Starting in development mode..."
	docker-compose -f docker/docker-compose.yml -f docker/docker-compose.dev.yml up -d
	@echo "[OK] Development mode started with live reload!"

dev-logs:
	docker-compose -f docker/docker-compose.yml -f docker/docker-compose.dev.yml logs -f app

# Python frontend server (for local development without Docker)
serve-ui:
	@echo "Starting Python frontend server for local development..."
	@echo "This will serve the UI on port 3001 (Docker frontend uses port 3000)"
	@echo "Make sure your FastAPI server is running on port 8000 first!"
	@echo ""
	python serve_test_page.py

# Logs and status
logs:
	docker-compose -f docker/docker-compose.yml logs -f app

logs-frontend:
	@echo "Showing frontend logs..."
	docker-compose -f docker/docker-compose.yml logs -f frontend

db-logs:
	docker-compose -f docker/docker-compose.yml logs -f postgres

logs-all:
	@echo "Showing all service logs..."
	docker-compose -f docker/docker-compose.yml logs -f

status:
	@echo "Service Status:"
	docker-compose -f docker/docker-compose.yml ps
	@echo ""
	@echo "Health Check:"
	@echo "API: " && curl -s http://localhost:8000/health | grep -o '"status":"[^"]*"' || echo "API not responding"
	@echo "Frontend: " && curl -s http://localhost:3000/frontend-health || echo "Frontend not responding"

# Database operations
db-connect:
	docker-compose -f docker/docker-compose.yml exec postgres psql -U email_user -d email_generator

db-reset:
	@echo "WARNING: This will destroy all data!"
	@read -p "Are you sure? (y/N): " confirm && [ "$$confirm" = "y" ]
	docker-compose -f docker/docker-compose.yml down -v
	docker-compose -f docker/docker-compose.yml up -d
	@echo "Database reset complete!"

# Build and maintenance
build:
	@echo "Building application..."
	docker-compose -f docker/docker-compose.yml build --no-cache app
	docker-compose -f docker/docker-compose.yml up -d app
	@echo "[OK] Build complete!"

clean:
	@echo "Cleaning up Docker resources..."
	docker-compose -f docker/docker-compose.yml down -v
	docker system prune -f
	@echo "[OK] Cleanup complete!"

fix:
	@echo "Running Docker build troubleshooter..."
	docker system prune -a -f
	@echo "Attempting to rebuild with cache cleanup..."
	docker-compose -f docker/docker-compose.yml build --no-cache
	docker-compose -f docker/docker-compose.yml up -d

# Backup
backup:
	@echo "[BACKUP] Creating database backup..."
	@mkdir -p backups
	docker-compose -f docker/docker-compose.yml exec postgres pg_dump -U email_user email_generator > backups/db_backup_$(shell date +%Y%m%d_%H%M%S).sql
	@tar -czf backups/uploads_backup_$(shell date +%Y%m%d_%H%M%S).tar.gz uploads/ || true
	@echo "[OK] Backup created in backups/ directory"

# Testing
test:
	@echo "Running comprehensive test suite..."
	./run_tests.sh

test-quick:
	@echo "Quick API tests..."
	@echo "Health check:"
	@curl -s http://localhost:8000/health | python3 -m json.tool || echo "[FAILED] Health check failed"
	@echo ""
	@echo "Templates:"
	@curl -s http://localhost:8000/templates | python3 -c "import sys, json; data=json.load(sys.stdin); print(f'[OK] {len(data)} templates available')" || echo "[FAILED] Templates endpoint failed"

test-data:
	@echo "Generating test data..."
	@docker-compose -f docker/docker-compose.yml exec app python scripts/generate_test_data.py --count 10
	@docker-compose -f docker/docker-compose.yml exec app ls -la uploads/ | grep test_data

test-email:
	@echo "Testing email generation..."
	@echo "name,company,linkedin_url,intelligence,template_type" > test.csv
	@echo "John Smith,TechCorp Inc,https://linkedin.com/in/johnsmith,false,sales_outreach" >> test.csv
	@curl -X POST "http://localhost:8000/generate-emails" \
		-H "accept: application/json" \
		-H "Content-Type: multipart/form-data" \
		-F "file=@test.csv;type=text/csv" \
		-F "template_type=sales_outreach" \
		-o test_result.csv
	@echo "[OK] Email generation completed - check test_result.csv"
	@rm -f test.csv

test-db:
	@echo "Testing database..."
	@docker-compose -f docker/docker-compose.yml exec app python init_database.py --test
	@docker-compose -f docker/docker-compose.yml exec postgres psql -U email_user -d email_generator -c "SELECT COUNT(*) as total_tables FROM information_schema.tables WHERE table_schema='public';"

test-performance:
	@echo "Performance test..."
	@docker-compose -f docker/docker-compose.yml exec app python scripts/generate_test_data.py --count 50
	@time curl -X POST "http://localhost:8000/generate-emails" \
		-H "accept: application/json" \
		-H "Content-Type: multipart/form-data" \
		-F "file=@uploads/test_data_50.csv;type=text/csv" \
		-F "template_type=sales_outreach" \
		-o performance_result.csv
	@echo "[OK] Performance test completed"

# Admin tools
admin:
	@echo "Starting with admin tools..."
	docker-compose -f docker/docker-compose.yml --profile admin up -d
	@echo "[OK] PgAdmin available at: http://localhost:5050"
	@echo "   Email: admin@email-generator.local"
	@echo "   Password: admin123"

full:
	@echo "Starting with all services..."
	docker-compose -f docker/docker-compose.yml --profile admin --profile cache up -d
	@echo "[OK] All services started:"
	@echo "   API: http://localhost:8000"
	@echo "   PgAdmin: http://localhost:5050"
	@echo "   Redis: localhost:6379"

# Setup helper
setup:
	@echo "First-time setup..."
	@if [ ! -f docker/.env ]; then \
		echo "Creating docker/.env file from template..."; \
		cp env.example docker/.env; \
		echo "Please edit docker/.env and add your OPENAI_API_KEY"; \
		echo "   Then run: make up-alpine"; \
	else \
		echo "[OK] docker/.env file already exists"; \
		make up-alpine; \
	fi
