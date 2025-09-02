"""
Database Connection Configuration

Handles PostgreSQL connection setup, session management, and configuration.
"""

import os
from typing import Optional, Generator
from sqlalchemy import create_engine, Engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
from sqlalchemy.exc import SQLAlchemyError
from contextlib import contextmanager
import logging

from .models import Base

# Configure logging
logger = logging.getLogger(__name__)


class DatabaseConfig:
    """Database configuration management"""
    
    def __init__(self):
        # Database connection parameters from environment
        self.host = os.getenv("DB_HOST", "localhost")
        self.port = int(os.getenv("DB_PORT", "5432"))
        self.database = os.getenv("DB_NAME", "email_generator")
        self.username = os.getenv("DB_USER", "email_user")
        self.password = os.getenv("DB_PASSWORD", "secure_email_password_123")
        
        # Connection pool settings
        self.pool_size = int(os.getenv("DB_POOL_SIZE", "10"))
        self.max_overflow = int(os.getenv("DB_MAX_OVERFLOW", "20"))
        self.pool_timeout = int(os.getenv("DB_POOL_TIMEOUT", "30"))
        self.pool_recycle = int(os.getenv("DB_POOL_RECYCLE", "3600"))
        
        # Connection options
        self.echo = os.getenv("DB_ECHO", "false").lower() == "true"
        self.echo_pool = os.getenv("DB_ECHO_POOL", "false").lower() == "true"
        
    @property
    def database_url(self) -> str:
        """Build PostgreSQL connection URL"""
        return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
    
    @property
    def async_database_url(self) -> str:
        """Build async PostgreSQL connection URL"""
        return f"postgresql+asyncpg://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
    
    def validate_config(self) -> tuple[bool, Optional[str]]:
        """Validate database configuration"""
        if not self.host:
            return False, "DB_HOST is required"
        if not self.database:
            return False, "DB_NAME is required"
        if not self.username:
            return False, "DB_USER is required"
        if not self.password:
            return False, "DB_PASSWORD is required"
        
        return True, None


class DatabaseManager:
    """Database connection and session management"""
    
    def __init__(self, config: Optional[DatabaseConfig] = None):
        self.config = config or DatabaseConfig()
        self._engine: Optional[Engine] = None
        self._session_factory: Optional[sessionmaker] = None
        
    @property
    def engine(self) -> Engine:
        """Get or create database engine"""
        if self._engine is None:
            self._engine = create_engine(
                self.config.database_url,
                poolclass=QueuePool,
                pool_size=self.config.pool_size,
                max_overflow=self.config.max_overflow,
                pool_timeout=self.config.pool_timeout,
                pool_recycle=self.config.pool_recycle,
                echo=self.config.echo,
                echo_pool=self.config.echo_pool,
                future=True  # Use SQLAlchemy 2.0 style
            )
        return self._engine
    
    @property
    def session_factory(self) -> sessionmaker:
        """Get or create session factory"""
        if self._session_factory is None:
            self._session_factory = sessionmaker(
                bind=self.engine,
                autocommit=False,
                autoflush=False,
                expire_on_commit=False
            )
        return self._session_factory
    
    def get_session(self) -> Session:
        """Create a new database session"""
        return self.session_factory()
    
    @contextmanager
    def session_scope(self) -> Generator[Session, None, None]:
        """Provide a transactional scope around database operations"""
        session = self.get_session()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    
    def test_connection(self) -> tuple[bool, Optional[str]]:
        """Test database connection"""
        try:
            with self.session_scope() as session:
                session.execute(text("SELECT 1"))
            return True, None
        except SQLAlchemyError as e:
            return False, str(e)
        except Exception as e:
            return False, f"Unexpected error: {str(e)}"
    
    def create_tables(self, drop_first: bool = False) -> bool:
        """Create database tables"""
        try:
            if drop_first:
                Base.metadata.drop_all(bind=self.engine)
                logger.warning("Dropped all existing tables")
            
            Base.metadata.create_all(bind=self.engine)
            logger.info("Successfully created database tables")
            return True
        except SQLAlchemyError as e:
            logger.error(f"Failed to create tables: {e}")
            return False
    
    def close(self):
        """Close database connections"""
        if self._engine:
            self._engine.dispose()
            self._engine = None
        self._session_factory = None
        logger.info("Database connections closed")


# Global database manager instance
_db_manager: Optional[DatabaseManager] = None


def get_database_manager() -> DatabaseManager:
    """Get or create global database manager"""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager


def get_db_session() -> Generator[Session, None, None]:
    """FastAPI dependency for database sessions"""
    db_manager = get_database_manager()
    with db_manager.session_scope() as session:
        yield session


def init_database(drop_first: bool = False) -> tuple[bool, Optional[str]]:
    """Initialize database with tables and views"""
    try:
        # Validate configuration
        config = DatabaseConfig()
        is_valid, error_msg = config.validate_config()
        if not is_valid:
            return False, f"Configuration error: {error_msg}"
        
        # Test connection
        db_manager = get_database_manager()
        connection_ok, connection_error = db_manager.test_connection()
        if not connection_ok:
            return False, f"Connection error: {connection_error}"
        
        # Create tables
        tables_ok = db_manager.create_tables(drop_first=drop_first)
        if not tables_ok:
            return False, "Failed to create database tables"
        
        # Create views and indexes (would normally be in migrations)
        try:
            with db_manager.session_scope() as session:
                # Create full-text search index if it doesn't exist
                session.execute(text("""
                    DO $$
                    BEGIN
                        IF NOT EXISTS (
                            SELECT 1 FROM information_schema.columns 
                            WHERE table_name='generated_emails' 
                            AND column_name='email_search_vector'
                        ) THEN
                            ALTER TABLE generated_emails 
                            ADD COLUMN email_search_vector tsvector 
                            GENERATED ALWAYS AS (to_tsvector('english', COALESCE(generated_email, ''))) STORED;
                            
                            CREATE INDEX idx_email_content_search 
                            ON generated_emails USING gin(email_search_vector);
                        END IF;
                    END
                    $$;
                """))
                
                # Create views if they don't exist
                session.execute(text("""
                    CREATE OR REPLACE VIEW request_summary AS
                    SELECT 
                        r.request_id,
                        r.original_filename,
                        r.total_rows,
                        r.template_llm_rows,
                        r.ai_research_rows,
                        r.status,
                        r.created_at,
                        r.total_processing_time_seconds,
                        COUNT(CASE WHEN e.status = 'completed' THEN 1 END) as successful_emails,
                        COUNT(CASE WHEN e.status = 'failed' THEN 1 END) as failed_emails,
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
                             r.total_llm_tokens_used, r.estimated_cost_usd;
                """))
                
                session.execute(text("""
                    CREATE OR REPLACE VIEW recent_performance AS
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
                """))
                
                logger.info("Successfully created database views and indexes")
        
        except Exception as e:
            logger.warning(f"Failed to create views/indexes (non-fatal): {e}")
        
        return True, "Database initialized successfully"
        
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        return False, str(e)


def close_database():
    """Close database connections"""
    global _db_manager
    if _db_manager:
        _db_manager.close()
        _db_manager = None
