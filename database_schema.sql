-- Email Generator Database Schema

-- Extension for UUID generation
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- 1. EMAIL GENERATION REQUESTS
CREATE TABLE email_requests (
    id SERIAL PRIMARY KEY,
    request_id VARCHAR(50) UNIQUE NOT NULL,
    
    -- File information
    original_filename VARCHAR(255) NOT NULL,
    file_size_bytes INTEGER,
    file_size_mb DECIMAL(10,2),
    
    -- Processing metadata
    total_rows INTEGER NOT NULL,
    template_llm_rows INTEGER DEFAULT 0,
    ai_research_rows INTEGER DEFAULT 0,
    
    -- Template settings
    fallback_template_type VARCHAR(50),
    processing_method VARCHAR(50) DEFAULT 'unified_llm',
    
    -- Status tracking
    status VARCHAR(20) DEFAULT 'processing', -- processing, completed, failed, partial
    
    -- Timing
    created_at TIMESTAMP DEFAULT NOW(),
    processing_started_at TIMESTAMP,
    processing_completed_at TIMESTAMP,
    total_processing_time_seconds DECIMAL(10,2),
    
    -- Results summary
    successful_emails INTEGER DEFAULT 0,
    failed_emails INTEGER DEFAULT 0,
    
    -- Cost tracking
    total_llm_tokens_used INTEGER DEFAULT 0,
    estimated_cost_usd DECIMAL(10,6) DEFAULT 0.00
);

-- Indexes for email_requests table
CREATE INDEX IF NOT EXISTS idx_request_id ON email_requests (request_id);
CREATE INDEX IF NOT EXISTS idx_created_at ON email_requests (created_at);
CREATE INDEX IF NOT EXISTS idx_status ON email_requests (status);

-- 2. GENERATED EMAILS
CREATE TABLE generated_emails (
    id SERIAL PRIMARY KEY,
    request_id VARCHAR(50) NOT NULL REFERENCES email_requests(request_id) ON DELETE CASCADE,
    
    -- Original CSV row data
    csv_row_index INTEGER NOT NULL,
    name VARCHAR(255) NOT NULL,
    company VARCHAR(255) NOT NULL,
    linkedin_url TEXT NOT NULL,
    
    -- Email generation settings
    intelligence_used BOOLEAN DEFAULT FALSE,
    template_type VARCHAR(50),
    
    -- Processing details
    processing_type VARCHAR(50), -- 'template_llm' or 'ai_with_research'
    placeholder_uuid UUID DEFAULT uuid_generate_v4(),
    
    -- Generated content
    generated_email TEXT,
    
    -- Processing metadata
    status VARCHAR(20) DEFAULT 'processing', -- processing, completed, failed
    error_message TEXT,
    
    -- AI/LLM details
    llm_model_used VARCHAR(50),
    prompt_tokens INTEGER DEFAULT 0,
    completion_tokens INTEGER DEFAULT 0,
    total_tokens INTEGER DEFAULT 0,
    processing_time_seconds DECIMAL(10,2),
    cost_usd DECIMAL(10,6) DEFAULT 0.00,
    
    -- LinkedIn research metadata (for AI emails)
    linkedin_research_quality DECIMAL(3,2), -- 0.00-1.00 score
    linkedin_research_time_seconds DECIMAL(10,2),
    
    -- Timing
    created_at TIMESTAMP DEFAULT NOW(),
    processing_started_at TIMESTAMP,
    processing_completed_at TIMESTAMP,
    
    -- Full-text search on email content
    generated_emails_search_vector tsvector GENERATED ALWAYS AS (
        to_tsvector('english', COALESCE(generated_email, ''))
    ) STORED
);

-- Indexes for generated_emails table
CREATE INDEX IF NOT EXISTS idx_generated_emails_request_id ON generated_emails (request_id);
CREATE INDEX IF NOT EXISTS idx_generated_emails_processing_type ON generated_emails (processing_type);
CREATE INDEX IF NOT EXISTS idx_generated_emails_status ON generated_emails (status);
CREATE INDEX IF NOT EXISTS idx_generated_emails_created_at ON generated_emails (created_at);
CREATE INDEX IF NOT EXISTS idx_generated_emails_company ON generated_emails (company);
CREATE INDEX IF NOT EXISTS idx_generated_emails_placeholder_uuid ON generated_emails (placeholder_uuid);

-- Full-text search index
CREATE INDEX idx_email_content_search ON generated_emails USING gin(generated_emails_search_vector);

-- 3. PROCESSING BATCHES
CREATE TABLE processing_batches (
    id SERIAL PRIMARY KEY,
    request_id VARCHAR(50) NOT NULL REFERENCES email_requests(request_id) ON DELETE CASCADE,
    
    -- Batch information
    batch_type VARCHAR(20) NOT NULL, -- 'template_llm' or 'ai_research'
    batch_number INTEGER NOT NULL,
    total_batches INTEGER NOT NULL,
    
    -- Batch processing
    emails_in_batch INTEGER NOT NULL,
    batch_size_limit INTEGER DEFAULT 5,
    
    -- Status and timing
    status VARCHAR(20) DEFAULT 'pending', -- pending, processing, completed, failed
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    processing_time_seconds DECIMAL(10,2),
    
    -- Results
    successful_emails INTEGER DEFAULT 0,
    failed_emails INTEGER DEFAULT 0,
    
    -- Batch-level costs
    batch_tokens_used INTEGER DEFAULT 0,
    batch_cost_usd DECIMAL(10,6) DEFAULT 0.00,
    
    -- Error tracking
    error_message TEXT,
    retry_count INTEGER DEFAULT 0
);

-- Indexes for processing_batches table
CREATE INDEX IF NOT EXISTS idx_processing_batches_request_batch ON processing_batches (request_id, batch_type);
CREATE INDEX IF NOT EXISTS idx_processing_batches_status ON processing_batches (status);

-- 4. ERROR LOGS
CREATE TABLE processing_errors (
    id SERIAL PRIMARY KEY,
    request_id VARCHAR(50), -- May be NULL for system errors
    email_id INTEGER REFERENCES generated_emails(id) ON DELETE CASCADE,
    
    -- Error details
    error_type VARCHAR(50) NOT NULL, -- 'llm_api', 'template_generation', 'linkedin_research', 'system'
    error_code VARCHAR(50),
    error_message TEXT NOT NULL,
    error_details JSONB, -- Full error context/stack trace
    
    -- Context
    processing_type VARCHAR(50),
    batch_id INTEGER REFERENCES processing_batches(id) ON DELETE CASCADE,
    
    -- Timing
    occurred_at TIMESTAMP DEFAULT NOW(),
    
    -- Resolution
    resolved BOOLEAN DEFAULT FALSE,
    resolved_at TIMESTAMP,
    resolution_notes TEXT
);

-- Indexes for processing_errors table
CREATE INDEX IF NOT EXISTS idx_processing_errors_request_id ON processing_errors (request_id);
CREATE INDEX IF NOT EXISTS idx_processing_errors_error_type ON processing_errors (error_type);
CREATE INDEX IF NOT EXISTS idx_processing_errors_occurred_at ON processing_errors (occurred_at);
CREATE INDEX IF NOT EXISTS idx_processing_errors_resolved ON processing_errors (resolved);

-- 5. SYSTEM METRICS (Optional)
CREATE TABLE system_metrics (
    id SERIAL PRIMARY KEY,
    metric_name VARCHAR(100) NOT NULL,
    metric_value DECIMAL(15,6) NOT NULL,
    metric_unit VARCHAR(20), -- 'count', 'seconds', 'usd', 'tokens'
    
    -- Time series data
    recorded_at TIMESTAMP DEFAULT NOW(),
    time_period VARCHAR(20), -- 'hourly', 'daily', 'request'
    
    -- Context
    request_id VARCHAR(50)
);

-- Indexes for system_metrics table
CREATE INDEX IF NOT EXISTS idx_system_metrics_name_time ON system_metrics (metric_name, recorded_at);
CREATE INDEX IF NOT EXISTS idx_system_metrics_request_id ON system_metrics (request_id);

-- 6. USEFUL VIEWS

-- Request summary view
CREATE VIEW request_summary AS
SELECT 
    r.request_id,
    r.original_filename,
    r.total_rows,
    r.template_llm_rows,
    r.ai_research_rows,
    r.status,
    r.created_at,
    r.total_processing_time_seconds,
    r.successful_emails,
    r.failed_emails,
    r.total_llm_tokens_used,
    r.estimated_cost_usd,
    COUNT(e.id) as total_generated_emails,
    AVG(e.processing_time_seconds) as avg_email_processing_time,
    SUM(e.total_tokens) as actual_tokens_used,
    SUM(e.cost_usd) as actual_cost_usd
FROM email_requests r
LEFT JOIN generated_emails e ON r.request_id = e.request_id
GROUP BY r.id, r.request_id, r.original_filename, r.total_rows, 
         r.template_llm_rows, r.ai_research_rows, r.status, 
         r.created_at, r.total_processing_time_seconds,
         r.successful_emails, r.failed_emails, 
         r.total_llm_tokens_used, r.estimated_cost_usd;

-- Recent processing performance view
CREATE VIEW recent_performance AS
SELECT 
    DATE(created_at) as date,
    COUNT(*) as requests_processed,
    SUM(total_rows) as total_emails_requested,
    SUM(successful_emails) as total_emails_completed,
    SUM(failed_emails) as total_emails_failed,
    ROUND(AVG(total_processing_time_seconds), 2) as avg_processing_time,
    SUM(total_llm_tokens_used) as total_tokens_consumed,
    SUM(estimated_cost_usd) as total_cost_usd
FROM email_requests 
WHERE created_at >= NOW() - INTERVAL '30 days'
GROUP BY DATE(created_at)
ORDER BY date DESC;
