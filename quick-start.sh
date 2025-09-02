#!/bin/bash

# Quick Start Script for Batch Email Generator
set -e

echo "Batch Email Generator - Quick Start"
echo "====================================="
echo

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Docker is not installed or not in PATH"
    echo "   Please install Docker: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if Docker Compose is available
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo "Docker Compose is not installed"
    echo "   Please install Docker Compose: https://docs.docker.com/compose/install/"
    exit 1
fi

echo "Docker and Docker Compose are installed"

# Check if docker/.env file exists
if [ ! -f "docker/.env" ]; then
    echo "Creating docker/.env file from template..."
    cp env.example docker/.env
    echo
    echo "  IMPORTANT: Please edit docker/.env file and set your OPENAI_API_KEY"
    echo "   You can get one at: https://platform.openai.com/"
    echo
    read -p "Press Enter to continue after setting your API key..."
fi

# Check if OPENAI_API_KEY is set
if ! grep -q "OPENAI_API_KEY=sk-" docker/.env; then
    echo "  Warning: OPENAI_API_KEY not set in docker/.env file"
    echo "   AI features will be disabled"
    echo
    read -p "Continue anyway? (y/N): " continue_without_key
    if [[ ! $continue_without_key =~ ^[Yy]$ ]]; then
        echo "Please set your OPENAI_API_KEY in docker/.env and run this script again"
        exit 1
    fi
fi

echo "🏗️  Starting services..."
echo

# Start services with frontend (using Alpine setup)
echo "Starting complete stack (database + API + frontend)..."
if command -v docker-compose &> /dev/null; then
    docker-compose -f docker/docker-compose.alpine.yml up -d --build
else
    docker compose -f docker/docker-compose.alpine.yml up -d --build
fi

echo
echo "⏳ Waiting for services to be ready..."

# Wait for health check
attempts=0
max_attempts=30
while [ $attempts -lt $max_attempts ]; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "Services are ready!"
        break
    fi
    echo "   Still starting... ($((attempts + 1))/$max_attempts)"
    sleep 2
    attempts=$((attempts + 1))
done

if [ $attempts -eq $max_attempts ]; then
    echo "Services failed to start within expected time"
    echo "   Check logs with: docker-compose -f docker/docker-compose.yml logs -f app"
    exit 1
fi

echo
echo "Email Generator is running!"
echo
echo "Quick Links:"
echo "   Web Interface:     http://localhost:3000  (Main UI - START HERE)"
echo "   API Documentation: http://localhost:8000/docs"
echo "   Health Check:      http://localhost:8000/health"
echo "   Available Templates: http://localhost:8000/templates"
echo
echo "Useful Commands:"
echo "   View logs:        docker-compose -f docker/docker-compose.alpine.yml logs -f app"
echo "   Stop services:    docker-compose -f docker/docker-compose.alpine.yml down"
echo "   Restart:          docker-compose -f docker/docker-compose.alpine.yml restart"
echo "   Generate test data: docker-compose -f docker/docker-compose.alpine.yml exec app python scripts/generate_test_data.py"
echo
echo "   Or use the Makefile (recommended):"
echo "   make status       # Check all services status"
echo "   make logs         # View application logs"
echo "   make test         # Test all endpoints"
echo "   make down         # Stop everything"
echo

# Test basic functionality
echo "Testing basic functionality..."
if curl -s http://localhost:8000/health | grep -q "status"; then
    echo "Health check passed"
else
    echo "Health check failed"
fi

if curl -s http://localhost:8000/templates | grep -q "sales_outreach"; then
    echo "Templates endpoint working"
else
    echo "Templates endpoint failed"
fi

echo
echo "Setup complete! Your email generator is ready to use."
echo "   Open http://localhost:8000/docs in your browser to get started."
