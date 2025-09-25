"""
Database Models for Email Generator

SQLAlchemy models based on the approved PostgreSQL schema.
"""

from sqlalchemy import (
    Column, Integer, String, Text, Boolean, DECIMAL, TIMESTAMP, 
    ForeignKey, Index, func, text
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID, JSONB
from datetime import datetime
import uuid

Base = declarative_base()


class EmailRequest(Base):
    """
    Tracks each CSV upload request lifecycle
    """
    __tablename__ = "email_requests"
    
    # Primary key
    id = Column(Integer, primary_key=True)
    request_id = Column(String(50), unique=True, nullable=False, index=True)
    
    # File information
    original_filename = Column(String(255), nullable=False)
    file_size_bytes = Column(Integer)
    file_size_mb = Column(DECIMAL(10, 2))
    
    # Processing metadata
    total_rows = Column(Integer, nullable=False)
    template_llm_rows = Column(Integer, default=0)
    ai_research_rows = Column(Integer, default=0)
    
    # Template settings
    fallback_template_type = Column(String(50))
    processing_method = Column(String(50), default='unified_llm')
    
    # Status tracking
    status = Column(String(20), default='processing', index=True)  # processing, completed, failed, partial
    
    # Timing
    created_at = Column(TIMESTAMP, default=datetime.utcnow, index=True)
    processing_started_at = Column(TIMESTAMP)
    processing_completed_at = Column(TIMESTAMP)
    total_processing_time_seconds = Column(DECIMAL(10, 2))
    
    # Results summary
    successful_emails = Column(Integer, default=0)
    failed_emails = Column(Integer, default=0)
    
    # Cost tracking
    total_llm_tokens_used = Column(Integer, default=0)
    estimated_cost_usd = Column(DECIMAL(10, 6), default=0.00)
    
    # Relationships
    generated_emails = relationship("GeneratedEmail", back_populates="request", cascade="all, delete-orphan")
    processing_batches = relationship("ProcessingBatch", back_populates="request", cascade="all, delete-orphan")
    errors = relationship("ProcessingError", back_populates="request", cascade="all, delete-orphan")


class GeneratedEmail(Base):
    """
    Individual email records with full context and metadata
    """
    __tablename__ = "generated_emails"
    
    # Primary key
    id = Column(Integer, primary_key=True)
    request_id = Column(String(50), ForeignKey("email_requests.request_id", ondelete="CASCADE"), nullable=False, index=True)
    
    # Original CSV row data
    csv_row_index = Column(Integer, nullable=False)
    name = Column(String(255), nullable=False)
    company = Column(String(255), nullable=False, index=True)
    linkedin_url = Column(Text, nullable=False)
    
    # Email generation settings
    intelligence_used = Column(Boolean, default=False)
    template_type = Column(String(50))
    
    # Processing details
    processing_type = Column(String(50), index=True)  # 'template_llm' or 'ai_with_research'
    placeholder_uuid = Column(UUID(as_uuid=True), default=uuid.uuid4, index=True)
    
    # Generated content
    generated_email = Column(Text)
    
    # Processing metadata
    status = Column(String(20), default='processing', index=True)  # processing, completed, failed
    error_message = Column(Text)
    
    # AI/LLM details
    llm_model_used = Column(String(50))
    prompt_tokens = Column(Integer, default=0)
    completion_tokens = Column(Integer, default=0)
    total_tokens = Column(Integer, default=0)
    processing_time_seconds = Column(DECIMAL(10, 2))
    cost_usd = Column(DECIMAL(10, 6), default=0.00)
    
    # LinkedIn research metadata (for AI emails)
    linkedin_research_quality = Column(DECIMAL(3, 2))  # 0.00-1.00 score
    linkedin_research_time_seconds = Column(DECIMAL(10, 2))
    
    # Timing
    created_at = Column(TIMESTAMP, default=datetime.utcnow, index=True)
    processing_started_at = Column(TIMESTAMP)
    processing_completed_at = Column(TIMESTAMP)
    
    # Relationships
    request = relationship("EmailRequest", back_populates="generated_emails")
    errors = relationship("ProcessingError", back_populates="email", cascade="all, delete-orphan")
    
    # Full-text search index will be created in migration
    
    # Additional indexes
    __table_args__ = (
        Index('idx_request_processing_type', 'request_id', 'processing_type'),
        Index('idx_status_created', 'status', 'created_at'),
    )


class ProcessingBatch(Base):
    """
    Tracks batch processing execution
    """
    __tablename__ = "processing_batches"
    
    # Primary key
    id = Column(Integer, primary_key=True)
    request_id = Column(String(50), ForeignKey("email_requests.request_id", ondelete="CASCADE"), nullable=False)
    
    # Batch information
    batch_type = Column(String(20), nullable=False)  # 'template_llm' or 'ai_research'
    batch_number = Column(Integer, nullable=False)
    total_batches = Column(Integer, nullable=False)
    
    # Batch processing
    emails_in_batch = Column(Integer, nullable=False)
    batch_size_limit = Column(Integer, default=100)
    
    # Status and timing
    status = Column(String(20), default='pending', index=True)  # pending, processing, completed, failed
    started_at = Column(TIMESTAMP)
    completed_at = Column(TIMESTAMP)
    processing_time_seconds = Column(DECIMAL(10, 2))
    
    # Results
    successful_emails = Column(Integer, default=0)
    failed_emails = Column(Integer, default=0)
    
    # Batch-level costs
    batch_tokens_used = Column(Integer, default=0)
    batch_cost_usd = Column(DECIMAL(10, 6), default=0.00)
    
    # Error tracking
    error_message = Column(Text)
    retry_count = Column(Integer, default=0)
    
    # Relationships
    request = relationship("EmailRequest", back_populates="processing_batches")
    errors = relationship("ProcessingError", back_populates="batch", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index('idx_request_batch_type', 'request_id', 'batch_type'),
    )


class ProcessingError(Base):
    """
    Centralized error logging with context
    """
    __tablename__ = "processing_errors"
    
    # Primary key
    id = Column(Integer, primary_key=True)
    request_id = Column(String(50), ForeignKey("email_requests.request_id", ondelete="CASCADE"))  # May be NULL
    email_id = Column(Integer, ForeignKey("generated_emails.id", ondelete="CASCADE"))
    
    # Error details
    error_type = Column(String(50), nullable=False, index=True)  # 'llm_api', 'template_generation', etc.
    error_code = Column(String(50))
    error_message = Column(Text, nullable=False)
    error_details = Column(JSONB)  # Full error context/stack trace
    
    # Context
    processing_type = Column(String(50))
    batch_id = Column(Integer, ForeignKey("processing_batches.id", ondelete="CASCADE"))
    
    # Timing
    occurred_at = Column(TIMESTAMP, default=datetime.utcnow, index=True)
    
    # Resolution
    resolved = Column(Boolean, default=False, index=True)
    resolved_at = Column(TIMESTAMP)
    resolution_notes = Column(Text)
    
    # Relationships
    request = relationship("EmailRequest", back_populates="errors")
    email = relationship("GeneratedEmail", back_populates="errors")
    batch = relationship("ProcessingBatch", back_populates="errors")


class SystemMetric(Base):
    """
    System performance and analytics metrics
    """
    __tablename__ = "system_metrics"
    
    # Primary key
    id = Column(Integer, primary_key=True)
    metric_name = Column(String(100), nullable=False)
    metric_value = Column(DECIMAL(15, 6), nullable=False)
    metric_unit = Column(String(20))  # 'count', 'seconds', 'usd', 'tokens'
    
    # Time series data
    recorded_at = Column(TIMESTAMP, default=datetime.utcnow)
    time_period = Column(String(20))  # 'hourly', 'daily', 'request'
    
    # Context
    request_id = Column(String(50))
    
    # Indexes
    __table_args__ = (
        Index('idx_metric_name_time', 'metric_name', 'recorded_at'),
        Index('idx_request_metric', 'request_id'),
    )


# Create useful views via raw SQL (will be added in migrations)
create_request_summary_view = text("""
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
         r.total_llm_tokens_used, r.estimated_cost_usd
""")

create_recent_performance_view = text("""
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
ORDER BY date DESC
""")

# Full-text search index creation (for migrations)
create_fulltext_search_index = text("""
ALTER TABLE generated_emails 
ADD COLUMN email_search_vector tsvector 
GENERATED ALWAYS AS (to_tsvector('english', COALESCE(generated_email, ''))) STORED;

CREATE INDEX idx_email_content_search ON generated_emails USING gin(email_search_vector);
""")
