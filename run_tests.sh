#!/bin/bash

# Comprehensive Test Script for Batch Email Generator Docker Setup
set -e

echo "Batch Email Generator - Comprehensive Test Suite"
echo "=================================================="
echo

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test counters
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

# Helper functions
log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
    ((PASSED_TESTS++))
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
    ((FAILED_TESTS++))
}

run_test() {
    local test_name="$1"
    local test_command="$2"
    
    echo
    log_info "Testing: $test_name"
    ((TOTAL_TESTS++))
    
    if eval "$test_command" > /dev/null 2>&1; then
        log_success "$test_name"
        return 0
    else
        log_error "$test_name"
        return 1
    fi
}

run_test_with_output() {
    local test_name="$1"
    local test_command="$2"
    local expected_pattern="$3"
    
    echo
    log_info "Testing: $test_name"
    ((TOTAL_TESTS++))
    
    local output=$(eval "$test_command" 2>&1)
    if [[ $output == *"$expected_pattern"* ]]; then
        log_success "$test_name"
        return 0
    else
        log_error "$test_name - Expected: $expected_pattern"
        echo "Got: $output"
        return 1
    fi
}

# Start testing
echo "Starting comprehensive test suite..."
echo

# Test 1: Check if Docker is available
log_info "Phase 1: Prerequisites"
if ! command -v docker &> /dev/null; then
    log_error "Docker is not installed"
    exit 1
fi

if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    log_error "Docker Compose is not installed"
    exit 1
fi

log_success "Docker and Docker Compose are available"

# Test 2: Start services if not running
log_info "Phase 2: Service Startup"
if ! docker-compose -f docker/docker-compose.yml ps | grep -q "Up"; then
    log_info "Starting services..."
    docker-compose -f docker/docker-compose.yml up -d
    sleep 10
fi

# Test 3: Check container status
run_test "Containers are running" "docker-compose -f docker/docker-compose.yml ps | grep -q 'Up'"

# Test 4: Wait for services to be ready
log_info "Phase 3: Service Health Checks"
log_info "Waiting for services to be ready (up to 60 seconds)..."

attempts=0
max_attempts=30
while [ $attempts -lt $max_attempts ]; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        log_success "API is responding"
        break
    fi
    echo -n "."
    sleep 2
    attempts=$((attempts + 1))
done

if [ $attempts -eq $max_attempts ]; then
    log_error "API failed to respond within 60 seconds"
    echo "Checking logs:"
    docker-compose -f docker/docker-compose.yml logs --tail=20 app
    exit 1
fi

# Test 5: API Health Check
run_test_with_output "API health check" "curl -s http://localhost:8000/health" "healthy"

# Test 6: Database connectivity
run_test_with_output "Database connection" "docker-compose -f docker/docker-compose.yml exec -T app python init_database.py --test" "successful"

# Test 7: Templates endpoint
run_test_with_output "Templates endpoint" "curl -s http://localhost:8000/templates" "sales_outreach"

# Test 8: Generate test data
log_info "Phase 4: Data Generation Tests"
run_test "Test data generation" "docker-compose -f docker/docker-compose.yml exec -T app python scripts/generate_test_data.py --count 5"

# Test 9: Check test files exist
run_test "Test files created" "docker-compose -f docker/docker-compose.yml exec -T app ls uploads/test_data_5.csv"

# Test 10: Template email generation
log_info "Phase 5: Email Generation Tests"
echo
log_info "Testing template-based email generation..."

# Create a small test file
echo "name,company,linkedin_url,intelligence,template_type
John Smith,TechCorp Inc,https://linkedin.com/in/johnsmith,false,sales_outreach
Jane Doe,DataFlow Solutions,https://linkedin.com/in/janedoe,false,recruitment" > test_template.csv

# Test email generation via API
if curl -X POST "http://localhost:8000/generate-emails" \
    -H "accept: application/json" \
    -H "Content-Type: multipart/form-data" \
    -F "file=@test_template.csv;type=text/csv" \
    -F "template_type=sales_outreach" \
    -o test_result.csv > /dev/null 2>&1; then
    
    if [ -s test_result.csv ]; then
        log_success "Template email generation"
        ((PASSED_TESTS++))
    else
        log_error "Template email generation - Empty result file"
        ((FAILED_TESTS++))
    fi
    ((TOTAL_TESTS++))
else
    log_error "Template email generation - API call failed"
    ((FAILED_TESTS++))
    ((TOTAL_TESTS++))
fi

# Test 11: Check OpenAI configuration (if available)
log_info "Phase 6: AI Feature Tests"
openai_status=$(docker-compose -f docker/docker-compose.yml exec -T app python -c "
import os
key = os.getenv('OPENAI_API_KEY', '')
if key.startswith('sk-'):
    print('CONFIGURED')
else:
    print('NOT_CONFIGURED')
" 2>/dev/null)

if [[ "$openai_status" == "CONFIGURED" ]]; then
    log_success "OpenAI API key is configured"
    
    # Test AI email generation
    echo "name,company,linkedin_url,intelligence,template_type
AI Test,TechCorp,https://linkedin.com/in/aitest,true,sales_outreach" > test_ai.csv
    
    if curl -X POST "http://localhost:8000/generate-emails" \
        -H "accept: application/json" \
        -H "Content-Type: multipart/form-data" \
        -F "file=@test_ai.csv;type=text/csv" \
        -F "template_type=sales_outreach" \
        -o test_ai_result.csv > /dev/null 2>&1; then
        
        log_success "AI email generation API call"
        ((PASSED_TESTS++))
    else
        log_error "AI email generation API call failed"
        ((FAILED_TESTS++))
    fi
    ((TOTAL_TESTS++))
    
else
    log_warning "OpenAI API key not configured - AI features disabled"
fi

# Test 12: Database persistence
log_info "Phase 7: Persistence Tests"
run_test "Database tables exist" "docker-compose -f docker/docker-compose.yml exec -T postgres psql -U email_user -d email_generator -c '\dt' | grep -q 'email_requests'"

# Test 13: Performance test (small scale)
log_info "Phase 8: Performance Tests"
if docker-compose -f docker/docker-compose.yml exec -T app python scripts/generate_test_data.py --count 25 > /dev/null 2>&1; then
    start_time=$(date +%s)
    if curl -X POST "http://localhost:8000/generate-emails" \
        -H "accept: application/json" \
        -H "Content-Type: multipart/form-data" \
        -F "file=@uploads/test_data_25.csv;type=text/csv" \
        -F "template_type=sales_outreach" \
        -o test_performance.csv > /dev/null 2>&1; then
        
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        
        if [ $duration -lt 30 ]; then
            log_success "Performance test (25 emails in ${duration}s)"
            ((PASSED_TESTS++))
        else
            log_warning "Performance test took ${duration}s (expected < 30s)"
            ((FAILED_TESTS++))
        fi
    else
        log_error "Performance test failed"
        ((FAILED_TESTS++))
    fi
    ((TOTAL_TESTS++))
fi

# Clean up test files
rm -f test_template.csv test_ai.csv test_result.csv test_ai_result.csv test_performance.csv

# Test 14: Container resource usage
log_info "Phase 9: Resource Usage"
memory_usage=$(docker stats --no-stream --format "table {{.MemUsage}}" email-generator-app | tail -n +2 | awk '{print $1}' | sed 's/MiB//')

if [ -n "$memory_usage" ] && [ "$memory_usage" -lt 500 ]; then
    log_success "Memory usage within limits (${memory_usage}MiB < 500MiB)"
    ((PASSED_TESTS++))
elif [ -n "$memory_usage" ]; then
    log_warning "Memory usage high: ${memory_usage}MiB"
    ((FAILED_TESTS++))
else
    log_warning "Could not measure memory usage"
fi
((TOTAL_TESTS++))

# Final summary
echo
echo "🏁 Test Summary"
echo "==============="
echo -e "Total Tests: ${BLUE}$TOTAL_TESTS${NC}"
echo -e "Passed: ${GREEN}$PASSED_TESTS${NC}"
echo -e "Failed: ${RED}$FAILED_TESTS${NC}"
echo

if [ $FAILED_TESTS -eq 0 ]; then
    echo -e "${GREEN}All tests passed! Your Docker setup is working perfectly.${NC}"
    echo
    echo "Your email generator is ready for:"
    echo "   • Production deployment"
    echo "   • User distribution"
    echo "   • API integrations"
    echo "   • Scale-up operations"
    echo
    echo "📋 Quick access:"
    echo "   • API Docs: http://localhost:8000/docs"
    echo "   • Health Check: http://localhost:8000/health"
    echo "   • View logs: docker-compose -f docker/docker-compose.yml logs -f app"
    exit 0
else
    echo -e "${RED}❌ Some tests failed. Please check the logs above.${NC}"
    echo
    echo "🔧 Common fixes:"
    echo "   • Restart services: docker-compose -f docker/docker-compose.yml restart"
    echo "   • Reset database: docker-compose -f docker/docker-compose.yml down -v && docker-compose -f docker/docker-compose.yml up -d"
    echo "   • Check logs: docker-compose -f docker/docker-compose.yml logs app"
    echo "   • Free up space: docker system prune"
    exit 1
fi
