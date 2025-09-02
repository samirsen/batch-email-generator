# Batch Email Generator

A production-ready FastAPI application that generates personalized outreach emails from CSV data using **unified LLM processing** with PostgreSQL persistence and a modern web interface.

## What This Actually Is

**Enterprise-grade email generation platform** that processes CSV uploads through OpenAI LLM with full database tracking, real-time progress monitoring, and a professional web interface.

**Perfect for**: Sales teams, recruiters, marketing professionals who need personalized emails at scale with full audit trails and analytics.

## Current Architecture (What's Actually Built)

### Backend Features
- **Unified LLM Processing**: All emails generated through OpenAI GPT-4 (no more simple templates)
- **Two Processing Modes**: 
  - Template-guided LLM (`intelligence=false`) - ~3-5s per email
  - AI Research LLM (`intelligence=true`) - ~5-8s per email with LinkedIn data
- **Background Processing**: All generation happens asynchronously with UUID placeholders
- **Full Database Persistence**: PostgreSQL with comprehensive tracking
- **Real-time Streaming**: Server-Sent Events for live progress updates
- **Request Management**: Complete request lifecycle with status tracking

### Frontend Features
- **Modern Web Interface**: Two-page application with drag-and-drop uploads
- **Real-time Progress**: Live progress bars and email completion notifications
- **Request History**: Full audit trail with pagination and detailed views
- **Download Management**: Direct CSV downloads of completed results
- **Professional UI**: Clean, responsive design with proper error handling

### Database Features
- **Email Requests**: Track every CSV upload with metadata and status
- **Generated Emails**: Store all emails with processing details and costs
- **Processing Batches**: Monitor background processing execution
- **Error Tracking**: Comprehensive error logging with context
- **Analytics Views**: Built-in performance and cost analytics

## Quick Start for Your Team

### 1. Zero-Configuration Setup (30 seconds)

   ```bash
# Clone and start

   git clone https://github.com/vivekverma1993/batch-email-generator.git
   cd batch-email-generator

# Create environment file and set your API key
cp env.example docker/.env
# Edit docker/.env file and set OPENAI_API_KEY=sk-your-api-key-here

# Start everything (database + API + frontend)
make up-alpine
```

**Access Points:**
- **Web Interface**: http://localhost:3000 (Primary interface)
- **API Documentation**: http://localhost:8000/docs
- **API Health Check**: http://localhost:8000/health

### 2. Alternative: Automated Script

```bash
# Guided setup with validation (handles environment setup)
./quick-start.sh

# Comprehensive testing
./run_tests.sh
```

## How It Actually Works

### 1. Upload & Processing
1. **Upload CSV** via web interface (name, company, linkedin_url columns required)
2. **Immediate Response** with request_id and UUID placeholders
3. **Background Processing** generates emails using OpenAI
4. **Real-time Updates** via SSE show completion progress
5. **Download Results** when processing completes

### 2. Processing Architecture
- **Template Mode**: LLM uses template as base prompt (faster, cheaper)
- **AI Mode**: LLM includes fake LinkedIn research (slower, more personalized)
- **Mixed Mode**: Single CSV can contain both types
- **All Background**: No blocking - immediate response with placeholders
- **Full Persistence**: Everything stored in PostgreSQL for analytics

### 3. Web Interface
- **Upload Page** (`/`): Drag-and-drop CSV upload with live progress
- **History Page** (`/history`): View all requests with detailed analytics
- **Real-time Updates**: Progress bars, completion notifications
- **Download Management**: Direct CSV downloads when ready

## API Endpoints (What Your Team Can Use)

### Core Processing
- `POST /upload-and-process` - Upload CSV, get request_id (Frontend-optimized)
- `POST /generate-emails` - Upload CSV, get immediate CSV with placeholders  
- `GET /stream/requests/{id}` - SSE stream for real-time progress
- `GET /requests/{id}/download` - Download completed CSV results

### Request Management
- `GET /requests` - List all requests with pagination
- `GET /requests/{id}` - Get detailed request status
- `GET /requests/{id}/details` - Get comprehensive request analytics
- `GET /requests/{id}/emails` - Get individual email results

### System Info
- `GET /health` - System health and configuration
- `GET /templates` - Available template types
- `GET /` - API overview and capabilities

## CSV Format

**Required columns:**
- `name` - Contact name
- `company` - Company name  
- `linkedin_url` - LinkedIn profile URL

**Optional columns:**
- `intelligence` - `true`/`false` (AI research vs template-guided)
- `template_type` - Override template per row

**Example:**
```csv
name,company,linkedin_url,intelligence,template_type
John Smith,TechCorp,https://linkedin.com/in/johnsmith,false,sales_outreach
Sarah Johnson,DataCorp,https://linkedin.com/in/sarahjohnson,true,recruitment
```

## Database Schema (Already Implemented)

### Tables
- **email_requests** - Request tracking with status and metadata
- **generated_emails** - Individual email records with full context
- **processing_batches** - Background processing execution tracking
- **processing_errors** - Comprehensive error logging
- **system_metrics** - Performance and analytics data

### Analytics Capabilities
```sql
-- Cost analysis
SELECT DATE(created_at), SUM(estimated_cost_usd) 
FROM email_requests GROUP BY DATE(created_at);

-- Performance metrics  
SELECT processing_type, AVG(processing_time_seconds)
FROM generated_emails GROUP BY processing_type;

-- Success rates
SELECT template_type, COUNT(*), 
       AVG(CASE WHEN status='completed' THEN 1.0 ELSE 0.0 END) as success_rate
FROM generated_emails GROUP BY template_type;
```

## Configuration

### Environment Variables

**Simple Setup:**
1. Copy `env.example` to `docker/.env`: `cp env.example docker/.env`
2. Set your OpenAI API key in `docker/.env`
3. All other settings use Docker defaults

**Key Variables:**
```bash
# Required
OPENAI_API_KEY=sk-your-api-key-here

# Optional Processing Settings
BATCH_SIZE=100                    # Template processing batch size
INTELLIGENCE_BATCH_SIZE=5         # AI processing batch size  
MAX_CSV_ROWS=50000               # Max rows per upload

# Database Settings (Docker defaults work out-of-box)
DB_HOST=postgres
DB_NAME=email_generator  
DB_USER=email_user
DB_PASSWORD=secure_email_password_123
```

### Available Templates
- `sales_outreach` - B2B lead generation (default)
- `recruitment` - Talent acquisition
- `networking` - Professional connections
- `partnership` - Business partnerships
- `follow_up` - Re-engagement
- `introduction` - Warm introductions
- `cold_email` - Initial outreach

## Docker Management

### Basic Commands
```bash
make up-alpine      # Start complete stack (recommended)
make status         # Check all services
make logs           # View application logs
make logs-frontend  # View web interface logs
make test           # Run health checks
make down           # Stop everything
```

### Direct Docker Commands
```bash
# Start services
docker-compose -f docker/docker-compose.alpine.yml up -d

# View logs
docker-compose -f docker/docker-compose.yml logs -f app

# Clean restart
docker-compose -f docker/docker-compose.yml down -v
docker-compose -f docker/docker-compose.yml up -d
```

## Project Structure

```
├── src/                         # FastAPI backend application
│   ├── database/               # PostgreSQL models & services
│   ├── main.py                 # API routes & SSE endpoints
│   ├── ai_generator.py         # OpenAI integration
│   └── templates.py            # Email template management
├── ui/                         # Web frontend (2 pages)
│   ├── index.html              # Upload & processing interface
│   └── history.html            # Request history & analytics
├── docker/                     # All Docker configuration
│   ├── Dockerfile.alpine       # Backend container
│   ├── Dockerfile.frontend     # Nginx frontend container
│   ├── docker-compose.alpine.yml # Complete stack setup
│   └── docker.env              # Environment template
├── database_schema.sql         # PostgreSQL schema
├── Makefile                   # Docker management commands
├── quick-start.sh             # Automated setup
└── run_tests.sh               # Comprehensive testing
```

## Performance & Scale

### Capacity
- **Max CSV Size**: 50,000 rows per upload
- **Processing Speed**: 3-8 seconds per email (LLM dependent)
- **Concurrent Uploads**: Multiple requests supported
- **Database**: Full PostgreSQL with analytics views

### Cost Estimation
- **Template Mode**: ~$0.001-0.003 per email
- **AI Mode**: ~$0.005-0.015 per email  
- **Mixed Mode**: Cost varies by intelligence distribution
- **Full Cost Tracking**: Database tracks all OpenAI usage

## Production Deployment

### Docker (Recommended)
```bash
# Production stack
make up-alpine

# Monitor
make status
make logs-all

# Scale (modify docker-compose.alpine.yml)
docker-compose scale app=3
```

### System Requirements
- **Minimum**: 2GB RAM, 1 CPU core, 10GB storage
- **Recommended**: 4GB RAM, 2 CPU cores, 50GB storage
- **Database**: PostgreSQL 15+ (included in Docker)
- **External**: OpenAI API access required

## Support & Maintenance

### Health Monitoring
```bash
# Quick health check
curl http://localhost:8000/health

# Frontend health
curl http://localhost:3000/frontend-health

# Database connection
make test-db
```

### Log Access
```bash
make logs           # Application logs
make logs-frontend  # Web interface logs  
make logs-all       # All services
```

### Troubleshooting
1. **Services not starting**: `make down && make up-alpine`
2. **Database connection issues**: Check if docker/.env has correct DB settings (should match env.example defaults)
3. **OpenAI errors**: Verify `OPENAI_API_KEY` is set in docker/.env file
4. **Port conflicts**: Check ports 3000, 8000, 5432 availability
5. **Environment issues**: `cp env.example docker/.env` to reset to defaults

## What Your Team Gets

**Immediate Value:**
- Production-ready email generation platform
- Modern web interface with real-time updates
- Full database persistence and analytics
- Comprehensive API with SSE streaming
- Professional Docker deployment

**Business Benefits:**
- Scale personalized outreach efficiently  
- Track all processing with audit trails
- Reduce manual email writing time
- Professional presentation to prospects
- Cost-effective LLM utilization

This is a complete, production-ready system - not a prototype or MVP. Your team can deploy it immediately and start generating personalized emails at scale.
