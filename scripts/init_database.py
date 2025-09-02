#!/usr/bin/env python3
"""
Database Initialization Script

Run this script to set up the PostgreSQL database for the Email Generator.
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from sqlalchemy import text
from src.database.connection import init_database, DatabaseConfig
from src.database.services import SystemMetricService


def setup_database(drop_existing: bool = False, test_connection: bool = True) -> bool:
    """
    Set up the database with tables, indexes, and initial data
    
    Args:
        drop_existing: Whether to drop existing tables first
        test_connection: Whether to test connection before setup
        
    Returns:
        bool: Success status
    """
    print("Email Generator Database Setup")
    print("=" * 50)
    
    # Show configuration
    config = DatabaseConfig()
    print(f"Database: {config.database}")
    print(f"Host: {config.host}:{config.port}")
    print(f"User: {config.username}")
    print(f"URL: {config.database_url}")
    print()
    
    # Validate configuration
    print("Validating configuration...")
    is_valid, error_msg = config.validate_config()
    if not is_valid:
        print(f"Configuration Error: {error_msg}")
        print("\nRequired environment variables:")
        print("  - DB_HOST (default: localhost)")
        print("  - DB_PORT (default: 5432)")
        print("  - DB_NAME (default: email_generator)")
        print("  - DB_USER (default: postgres)")
        print("  - DB_PASSWORD (required)")
        return False
    
    print("Configuration valid")
    
    if drop_existing:
        print("\nWARNING: This will DROP all existing tables!")
        confirmation = input("Type 'YES' to confirm: ")
        if confirmation != 'YES':
            print("Aborted by user")
            return False
    
    # Initialize database
    print("\nInitializing database...")
    success, message = init_database(drop_first=drop_existing)
    
    if not success:
        print(f"Database initialization failed: {message}")
        return False
    
    print(f"{message}")
    
    # Record initial system metric
    try:
        SystemMetricService.record_metric(
            metric_name="database_initialized",
            metric_value=1,
            metric_unit="count",
            time_period="event"
        )
        print("Initial metrics recorded")
    except Exception as e:
        print(f"Warning: Could not record initial metrics: {e}")
    
    print("\nDatabase setup completed successfully!")
    print("\nNext steps:")
    print("1. Start your FastAPI server: uvicorn src.main:app --reload")
    print("2. Upload a CSV file to test the system")
    print("3. Check the database for stored results")
    
    return True


def show_connection_info():
    """Show database connection information"""
    print("📋 Database Connection Information")
    print("=" * 40)
    
    config = DatabaseConfig()
    
    print(f"Host: {config.host}")
    print(f"Port: {config.port}")
    print(f"Database: {config.database}")
    print(f"Username: {config.username}")
    print(f"Password: {'*' * len(config.password) if config.password else 'Not set'}")
    print(f"Connection URL: postgresql://{config.username}:***@{config.host}:{config.port}/{config.database}")
    print()
    
    print("Environment Variables:")
    env_vars = [
        ("DB_HOST", config.host),
        ("DB_PORT", config.port),
        ("DB_NAME", config.database),
        ("DB_USER", config.username),
        ("DB_PASSWORD", "***" if config.password else "NOT SET"),
        ("DB_POOL_SIZE", config.pool_size),
        ("DB_MAX_OVERFLOW", config.max_overflow),
    ]
    
    for var, value in env_vars:
        status = "OK" if os.getenv(var) else "MISSING"
        print(f"  {status} {var}: {value}")


def test_connection():
    """Test database connection"""
    print("Testing Database Connection")
    print("=" * 35)
    
    from src.database.connection import get_database_manager
    
    try:
        db_manager = get_database_manager()
        success, error_msg = db_manager.test_connection()
        
        if success:
            print("Connection successful!")
            
            # Try to get some basic info
            with db_manager.session_scope() as session:
                result = session.execute(text("SELECT version()")).fetchone()
                if result:
                    print(f"PostgreSQL Version: {result[0]}")
                
                # Check if tables exist
                table_check = session.execute(text("""
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name IN ('email_requests', 'generated_emails')
                """)).fetchall()
                
                if table_check:
                    print(f"Tables found: {', '.join([t[0] for t in table_check])}")
                else:
                    print("No email generator tables found. Run --setup to create them.")
            
        else:
            print(f"Connection failed: {error_msg}")
            print("\nTroubleshooting:")
            print("1. Make sure PostgreSQL is running")
            print("2. Check your database credentials")
            print("3. Verify the database exists")
            print("4. Check network connectivity to database server")
            return False
            
    except Exception as e:
        print(f"Connection test failed: {e}")
        return False
    
    return True


def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(
        description="Initialize PostgreSQL database for Email Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python init_database.py --setup                 # Set up database
  python init_database.py --setup --drop          # Drop and recreate tables
  python init_database.py --test                  # Test connection
  python init_database.py --info                  # Show connection info
        """
    )
    
    parser.add_argument(
        "--setup", 
        action="store_true", 
        help="Initialize database with tables and views"
    )
    
    parser.add_argument(
        "--drop", 
        action="store_true", 
        help="Drop existing tables before setup (use with --setup)"
    )
    
    parser.add_argument(
        "--test", 
        action="store_true", 
        help="Test database connection"
    )
    
    parser.add_argument(
        "--info", 
        action="store_true", 
        help="Show database connection information"
    )
    
    args = parser.parse_args()
    
    # If no arguments provided, show help
    if not any([args.setup, args.test, args.info]):
        parser.print_help()
        return
    
    success = True
    
    if args.info:
        show_connection_info()
        print()
    
    if args.test:
        success = test_connection() and success
        print()
    
    if args.setup:
        if args.drop and not args.test:
            # Test connection first if dropping tables
            print("Testing connection before dropping tables...")
            if not test_connection():
                print("Aborting setup due to connection failure")
                sys.exit(1)
            print()
        
        success = setup_database(drop_existing=args.drop) and success
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
