import traceback
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from typing import Optional, AsyncGenerator
import pandas as pd
import io
import asyncio
import os
import json
import logging
import uuid
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

# Import template functionality
from .templates import (
    TemplateType, 
    DEFAULT_TEMPLATE_TYPE, 
    get_all_templates,
    get_template_info
)

# Import AI generation
from .ai_generator import AIEmailGenerator, GenerationStatus

# Import modular components
from .csv_processor import (
    validate_and_enhance_csv,
    get_csv_info,
    split_dataframe_by_intelligence,
    validate_csv_size,
    validate_csv_not_empty
)
from .email_generator import (
    generate_single_email,
    process_template_dataframe
)
from .background_processor import (
    process_ai_emails_background,
    create_ai_placeholders,
    create_all_placeholders,
    process_all_emails_background
)
# Database imports
from .database.services import EmailRequestService, GeneratedEmailService, AnalyticsService
from .database.connection import get_database_manager


def create_database_records_and_placeholders(
    request_id: str,
    original_filename: str,
    df,
    template_rows,
    ai_rows,
    file_size_bytes: int,
    template_type: Optional[TemplateType] = None
) -> tuple[dict, dict]:
    """
    Create database records for the request and email placeholders
    
    For large datasets (>1000 rows), uses hybrid approach:
    1. Create request record immediately 
    2. Generate UUID placeholders in memory
    3. Defer bulk email record creation to background
    
    Returns:
        Tuple of (placeholders dict, uuid_mapping dict)
    """
    try:
        # Always create the email request record immediately
        EmailRequestService.create_request(
            request_id=request_id,
            original_filename=original_filename,
            total_rows=len(df),
            template_llm_rows=len(template_rows),
            ai_research_rows=len(ai_rows),
            file_size_bytes=file_size_bytes,
            fallback_template_type=template_type
        )
        
        # For large datasets, use hybrid approach
        dataset_size = len(df)
        if dataset_size > 1000:
            print(f"Large dataset detected ({dataset_size} rows). Using hybrid approach:")
            print("- Request record created immediately")
            print("- Email placeholders generated in memory")
            print("- Database email records will be bulk-created in background")
            
            # Generate UUIDs in memory without database operations
            all_placeholders = {}
            all_uuid_mapping = {}
            
            for _, row in df.iterrows():
                # Determine processing type
                intelligence_used = bool(row.get('intelligence', False))
                processing_prefix = "AI_PROCESSING" if intelligence_used else "TEMPLATE_LLM_PROCESSING"
                
                # Generate UUID in memory
                import uuid
                placeholder_uuid = str(uuid.uuid4())
                
                all_placeholders[row.name] = f"{processing_prefix}:{placeholder_uuid}"
                all_uuid_mapping[row.name] = placeholder_uuid
            
            # Schedule background bulk database creation
            print(f"Scheduling background bulk creation of {dataset_size} email records...")
            asyncio.create_task(
                _bulk_create_email_records_background(request_id, df, all_uuid_mapping)
            )
            
            return all_placeholders, all_uuid_mapping
        
        else:
            # For smaller datasets, use original synchronous approach
            print(f"Small dataset ({dataset_size} rows). Using synchronous database creation.")
            return _create_email_records_synchronous(request_id, df)
        
    except Exception as e:
        print(f"Warning: Failed to create database records: {e}")
        print("Falling back to memory-only placeholders...")
        
        # Fall back to the original placeholder creation if database fails
        return create_all_placeholders(df)


def _create_email_records_synchronous(request_id: str, df) -> tuple[dict, dict]:
    """
    Original synchronous email record creation for small datasets
    """
    all_placeholders = {}
    all_uuid_mapping = {}
    
    for _, row in df.iterrows():
        # Determine processing type
        intelligence_used = bool(row.get('intelligence', False))
        processing_type = "ai_with_research" if intelligence_used else "template_llm"
        
        # Create email placeholder in database
        email_record = GeneratedEmailService.create_email_placeholder(
            request_id=request_id,
            csv_row_index=int(row.name),
            name=str(row['name']),
            company=str(row['company']),
            linkedin_url=str(row['linkedin_url']),
            intelligence_used=intelligence_used,
            template_type=str(row.get('template_type', '')),
            processing_type=processing_type
        )
        
        # Create placeholder strings for CSV response
        placeholder_uuid_str = str(email_record.placeholder_uuid)
        processing_prefix = "AI_PROCESSING" if intelligence_used else "TEMPLATE_LLM_PROCESSING"
        
        all_placeholders[row.name] = f"{processing_prefix}:{placeholder_uuid_str}"
        all_uuid_mapping[row.name] = placeholder_uuid_str
    
    print(f"Database records created: 1 request + {len(df)} email placeholders")
    return all_placeholders, all_uuid_mapping


async def _bulk_create_email_records_background(
    request_id: str, 
    df, 
    uuid_mapping: dict
):
    """
    Background task to bulk-create email records in database
    
    Uses batch operations for efficiency with large datasets
    """
    try:
        print(f"Starting background bulk creation of {len(df)} email records for request {request_id}")
        
        # Import here to avoid circular imports
        from .database.services import GeneratedEmailService
        
        # Prepare all records for bulk insertion
        email_records_data = []
        for _, row in df.iterrows():
            intelligence_used = bool(row.get('intelligence', False))
            processing_type = "ai_with_research" if intelligence_used else "template_llm"
            
            record_data = {
                'request_id': request_id,
                'csv_row_index': int(row.name),
                'name': str(row['name']),
                'company': str(row['company']),
                'linkedin_url': str(row['linkedin_url']),
                'intelligence_used': intelligence_used,
                'template_type': str(row.get('template_type', '')),
                'processing_type': processing_type,
                'placeholder_uuid': uuid_mapping[row.name],
                'status': 'processing'
            }
            email_records_data.append(record_data)
        
        # Use bulk operations to create all records efficiently
        success_count = await GeneratedEmailService.bulk_create_email_placeholders(email_records_data)
        
        if success_count == len(df):
            print(f"Successfully bulk-created {success_count} email records for request {request_id}")
        else:
            print(f"Partial success: {success_count}/{len(df)} email records created for request {request_id}")
            
    except Exception as e:
        print(f"Error in background bulk email record creation for request {request_id}: {e}")
        # Note: We don't raise here since this is a background task
        # The email processing can still continue with the in-memory UUIDs


from .utils import (
    merge_results_in_order,
    generate_request_id,
    calculate_file_size_mb
)

# Configuration settings
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "100"))  # Default: 100, configurable via environment
MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "1000"))  # Maximum allowed batch size
MAX_CSV_ROWS = int(os.getenv("MAX_CSV_ROWS", "50000"))  # Maximum CSV rows to process
INTELLIGENCE_BATCH_SIZE = int(os.getenv("INTELLIGENCE_BATCH_SIZE", "100"))  # Smaller batches for AI processing
AI_FALLBACK_TO_TEMPLATE = os.getenv("AI_FALLBACK_TO_TEMPLATE", "true").lower() == "true"

app = FastAPI(
    title="Batch Email Generator",
    description="Generate personalized outreach emails from CSV data using configurable templates, fake LinkedIn research, and AI-powered generation",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware to handle cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=False,  # Set to False when using "*" for origins
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Mount static files to serve the UI  
app.mount("/ui", StaticFiles(directory="ui", html=True), name="ui")


@app.get("/")
async def root():
    """Root endpoint with basic information"""
    return {
        "message": "Welcome to Batch Email Generator API - NOW WITH UNIFIED LLM PROCESSING",
        "version": "2.1.0",
        "major_update": "All emails now use LLM generation (both template and AI types)",
        "features": [
            "Unified LLM-based email generation",
            "Template-guided LLM generation",
            "AI-powered personalization with LinkedIn research", 
            "Industry-aware data generation",
            "Background processing for all email types",
            "UUID placeholder system",
            "Unified result logging"
        ],
        "processing_types": {
            "template_llm": "LLM generation using template as base prompt (intelligence=false)",
            "ai_research": "LLM generation with fake LinkedIn research (intelligence=true)"
        },
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "templates": "/templates",
            "generate_emails": "/generate-emails",
            "test_metadata": "/process-emails-metadata",
            "task_status": "/task-status/{task_id}"
        },
        "important_note": "NO immediate email results - all processing happens in background with LLM API calls"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint to verify service is running"""
    # Check AI configuration
    try:
        from .ai_generator import validate_ai_configuration
        ai_valid, ai_error = validate_ai_configuration()
        ai_status = "configured" if ai_valid else f"not configured: {ai_error}"
    except Exception as e:
        ai_status = f"error: {str(e)}"
    
    return {
        "status": "healthy", 
        "message": "Batch Email Generator is running",
        "version": "2.0.0",
        "config": {
            "batch_size": BATCH_SIZE,
            "max_batch_size": MAX_BATCH_SIZE,
            "max_csv_rows": MAX_CSV_ROWS,
            "intelligence_batch_size": INTELLIGENCE_BATCH_SIZE,
            "ai_fallback_enabled": AI_FALLBACK_TO_TEMPLATE
        },
        "capabilities": {
            "templates": "available",
            "fake_linkedin_research": "available",
            "ai_generation": ai_status
        }
    }

@app.get("/templates")
async def get_available_templates():
    """Get list of available email templates"""
    return {
        "available_templates": get_all_templates(),
        "default_template": DEFAULT_TEMPLATE_TYPE.value,
        "total_templates": len(get_all_templates())
    }

@app.post("/process-emails-metadata")
async def process_emails_metadata(
    file: UploadFile = File(...),
    template_type: Optional[TemplateType] = Form(None, description="Fallback template type when CSV template_type column is empty. Uses sales_outreach if not provided.")
):
    """
    Process CSV and return metadata only (for testing large files via Swagger UI)
    
    This endpoint processes the CSV file and generates a few sample emails but returns
    only metadata instead of the full processed file. For testing large datasets
    through Swagger UI without download limitations.
    
    - **file**: CSV file with required columns: name, company, linkedin_url
              Optional columns: intelligence (true/false), template_type (template name)
    - **template_type**: Fallback email template type when template_type column is empty. Uses sales_outreach if not provided.
    
    **CSV Column Details:**
    - **name** (required): Contact's name
    - **company** (required): Contact's company name
    - **linkedin_url** (required): Contact's LinkedIn profile URL
    - **intelligence** (optional): true/false - Use AI research and generation (not implemented in Phase 1)
    - **template_type** (optional): Per-row template selection (overrides the fallback template_type parameter)
    
    Available template types: sales_outreach, recruitment, networking, partnership, follow_up, introduction, cold_email
    
    Returns: JSON with processing metadata and sample emails (no file download)
    """
    
    # Validate file type
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV")
    
    # Read file contents
    contents = await file.read()
    
    try:
        # Parse CSV data
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Validate CSV content and structure
        validate_csv_not_empty(df)
        df = validate_and_enhance_csv(df)
        validate_csv_size(df, MAX_CSV_ROWS)
        
        # Calculate processing information for new parallel approach
        ai_rows, template_rows = split_dataframe_by_intelligence(df)
        
        # Generate sample emails from first few rows
        sample_size = min(3, len(df))
        sample_df = df.head(sample_size)
        
        sample_emails = []
        for _, row in sample_df.iterrows():
            email = await generate_single_email(row, template_type)
            sample_emails.append({
                "name": str(row['name']),
                "company": str(row['company']),
                "linkedin_url": str(row['linkedin_url']),
                "intelligence": row.get('intelligence', False),
                "template_type": row.get('template_type', ''),
                "generated_email": email[:200] + "..." if len(email) > 200 else email
            })
        
        return {
            "status": "success",
            "message": "CSV processed successfully - metadata only (NEW: All emails use LLM)",
            "file_info": {
                "filename": file.filename,
                "size_bytes": len(contents),
                "size_mb": round(len(contents) / (1024 * 1024), 2)
            },
            "processing_info": {
                "total_rows": len(df),
                "template_llm_rows": len(template_rows),  # These now use LLM too
                "ai_research_rows": len(ai_rows),
                "processing_method": "unified-llm-background",
                "immediate_emails": 0,  # No immediate processing anymore
                "background_emails": len(df),  # All emails go to background
                "llm_batch_size": INTELLIGENCE_BATCH_SIZE,
                "estimated_output_size_mb": round((len(df) * 1000) / (1024 * 1024), 1),
                "estimated_processing_time_minutes": round((len(df) * 4) / 60, 1)  # ~4 seconds per LLM call
            },
            "template_info": get_template_info(template_type),
            "sample_emails": sample_emails,
            "important_changes": [
                "All emails now use LLM generation (both template and AI types)",
                "No immediate results - all processing happens in background",
                "Response contains UUID placeholders for all emails",
                "Results logged to unified_results_*.json files"
            ],
            "note": "This is metadata only. Use /generate-emails endpoint to get CSV with UUID placeholders. All processing now happens in background with LLM."
        }
        
    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="CSV file is empty or invalid")
    except pd.errors.ParserError:
        raise HTTPException(status_code=400, detail="Invalid CSV format")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File encoding not supported. Please use UTF-8 encoded CSV")
    except Exception as e:
        # Log the full error for debugging while returning safe message to user
        print(f"Unexpected error in process_emails_metadata: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Internal server error occurred while processing your request. Please try again or contact support if the issue persists."
        )










@app.post("/generate-emails")
async def generate_emails(
    file: UploadFile = File(...),
    template_type: Optional[TemplateType] = Form(None, description="Fallback template type when CSV template_type column is empty. Uses sales_outreach if not provided.")
):
    """
    Generate personalized emails from CSV data using unified LLM background processing
    
    **MAJOR CHANGE**: All emails now use LLM generation (both template and AI types)
    
    - **file**: CSV file with required columns: name, company, linkedin_url
              Optional columns: intelligence (true/false), template_type (template name)
    - **template_type**: Fallback email template type when template_type column is empty. Uses sales_outreach if not provided.
    
    **CSV Column Details:**
    - **name** (required): Contact's name
    - **company** (required): Contact's company name  
    - **linkedin_url** (required): Contact's LinkedIn profile URL
    - **intelligence** (optional): true/false - Use AI with LinkedIn research vs LLM with template base
    - **template_type** (optional): Per-row template selection (overrides the fallback template_type parameter)
    
    Available template types: sales_outreach, recruitment, networking, partnership, follow_up, introduction, cold_email
    
    **NEW Processing Architecture:**
    - **Template LLM emails** (intelligence=false): LLM generation using template as base (~3-5s per email)
    - **AI Research emails** (intelligence=true): LLM generation with LinkedIn research (~5-8s per email)
    - **All Background Processing**: NO immediate results - everything processed in background
    - **Immediate Response**: CSV with UUID placeholders for ALL emails
    - **Unified Logging**: All results logged to unified_results_*.json files
    
    **Response Headers:**
    - X-Request-ID: Unique identifier for tracking background processing
    - X-Immediate-Emails: "0" (no immediate emails anymore)
    - X-Background-Template-LLM-Emails: Number of template LLM emails processing
    - X-Background-AI-Research-Emails: Number of AI research emails processing
    - X-LLM-Status: "processing" (all emails use LLM)
    - X-JSON-Logs: Pattern for finding unified result JSON files
    
    Returns: CSV file with UUID placeholders for ALL emails (no immediate results)
    """
    
    # Validate file type
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV")
    
    # Read file contents
    contents = await file.read()
    
    try:
        # Parse CSV data
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Validate CSV content and structure
        validate_csv_not_empty(df)
        df = validate_and_enhance_csv(df)
        validate_csv_size(df, MAX_CSV_ROWS)
        
        # UNIFIED LLM PROCESSING: All emails now use LLM (both template and AI)
        ai_rows, template_rows = split_dataframe_by_intelligence(df)
        
        # Generate unique request ID for tracking
        request_id = generate_request_id()
        
        print(f"Request {request_id}: Processing {len(df)} rows (ALL with LLM)")
        print(f"   Template LLM emails: {len(template_rows)} (background)")
        print(f"   AI research emails: {len(ai_rows)} (background)")
        
        # Create database records and placeholders for 3-column generation
        all_placeholders, all_uuid_mapping = create_database_records_and_placeholders(
            request_id=request_id,
            original_filename=file.filename,
            df=df,
            template_rows=template_rows,
            ai_rows=ai_rows,
            file_size_bytes=len(contents),
            template_type=template_type
        )
        
        # Add 3 email columns with placeholders
        template_types = [TemplateType.LEXI, TemplateType.LUCAS, TemplateType.NETWORKING]
        template_columns = ['lexi_email', 'lucas_email', 'networking_email']
        
        for i, row in df.iterrows():
            # Get UUID for this row
            row_uuid = all_uuid_mapping.get(row.name, str(uuid.uuid4()))
            
            # Add placeholders for all 3 email columns
            for template_type, column_name in zip(template_types, template_columns):
                processing_type = "AI_PROCESSING" if row.get('intelligence', False) else "TEMPLATE_LLM_PROCESSING"
                df.loc[i, column_name] = f"{processing_type}:{row_uuid}_{template_type.value}"

        # Start unified background processing for ALL emails (3-column generation)
        print(f"Starting unified background 3-column LLM processing for request {request_id}")
        asyncio.create_task(
            process_all_emails_background(request_id, df, template_type, all_uuid_mapping)
        )
        
        # Convert dataframe to CSV string
        output = io.StringIO()
        df.to_csv(output, index=False)
        csv_content = output.getvalue()
        
        # Create streaming response
        def iter_csv():
            yield csv_content
        
        # All emails are now processed in background with LLM
        print(f"Request {request_id}: Returning CSV with {len(df)} placeholder UUIDs, all processing in background")
        
        # Calculate final file size for headers
        final_size_mb = calculate_file_size_mb(csv_content)
        
        return StreamingResponse(
            iter_csv(),
            media_type="text/csv; charset=utf-8",
            headers={
                "Content-Disposition": f"attachment; filename=generated_{file.filename}",
                "X-Request-ID": request_id,
                "X-Total-Rows": str(len(df)),
                "X-Background-Template-LLM-Emails": str(len(template_rows)),
                "X-Background-AI-Research-Emails": str(len(ai_rows)),
                "X-Immediate-Emails": "0",  # No immediate emails anymore
                "X-Processing-Method": "unified-llm-background",
                "X-Template-Type": template_type.value if template_type else DEFAULT_TEMPLATE_TYPE.value,
                "X-LLM-Status": "processing",  # All emails use LLM now
                "X-JSON-Logs": f"unified_results_{request_id}_*.json",
                "X-File-Size-MB": str(final_size_mb),
                "X-Original-Filename": file.filename,
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0"
            }
        )
        
    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="CSV file is empty or invalid")
    except pd.errors.ParserError:
        raise HTTPException(status_code=400, detail="Invalid CSV format")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File encoding not supported. Please use UTF-8 encoded CSV")
    except Exception as e:
        # Log the full error for debugging while returning safe message to user
        print(f"Unexpected error in generate_emails: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Internal server error occurred while processing your request. Please try again or contact support if the issue persists."
        )


@app.post("/upload-and-process")
async def upload_and_process(
    file: UploadFile = File(...),
    template_type: Optional[TemplateType] = Form(None, description="Fallback template type when CSV template_type column is empty. Uses sales_outreach if not provided.")
):
    """
    Upload CSV and start processing with immediate JSON response (Frontend-friendly)
    
    This endpoint is optimized for modern frontend applications:
    - Uploads CSV and validates it
    - Creates database records and starts background processing
    - Returns JSON with request_id for immediate SSE connection
    - No CSV download - use /requests/{request_id}/download for that
    
    Perfect for:
    - React/Vue/Angular applications
    - Real-time progress tracking with SSE
    - Modern UX patterns
    
    Returns: JSON with request_id, processing info, and SSE connection details
    """
    
    # Validate file type
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV")
    
    # Read file contents
    contents = await file.read()
    
    try:
        # Parse CSV data
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Validate CSV content and structure
        validate_csv_not_empty(df)
        df = validate_and_enhance_csv(df)
        validate_csv_size(df, MAX_CSV_ROWS)
        
        # Split processing types
        ai_rows, template_rows = split_dataframe_by_intelligence(df)
        
        # Generate unique request ID for tracking
        request_id = generate_request_id()
        
        print(f"Request {request_id}: Processing {len(df)} rows via frontend upload")
        print(f"   Template LLM emails: {len(template_rows)} (background)")
        print(f"   AI research emails: {len(ai_rows)} (background)")
        
        # Create database records and placeholders
        all_placeholders, all_uuid_mapping = create_database_records_and_placeholders(
            request_id=request_id,
            original_filename=file.filename,
            df=df,
            template_rows=template_rows,
            ai_rows=ai_rows,
            file_size_bytes=len(contents),
            template_type=template_type
        )
        
        # Start unified background processing for ALL emails
        print(f"Starting unified background LLM processing for request {request_id}")
        asyncio.create_task(
            process_all_emails_background(request_id, df, template_type, all_uuid_mapping)
        )
        
        # Return JSON response with all necessary info for frontend
        return {
            "success": True,
            "message": "CSV uploaded and processing started successfully",
            "request_id": request_id,
            "processing_info": {
                "total_emails": len(df),
                "template_llm_emails": len(template_rows),
                "ai_research_emails": len(ai_rows),
                "processing_method": "unified-llm-background",
                "estimated_processing_time_minutes": round((len(df) * 4) / 60, 1),
                "status": "processing"
            },
            "file_info": {
                "original_filename": file.filename,
                "size_bytes": len(contents),
                "size_mb": round(len(contents) / (1024 * 1024), 2),
                "total_rows": len(df)
            },
            "streaming": {
                "sse_endpoint": f"/stream/requests/{request_id}",
                "progress_endpoint": f"/requests/{request_id}",
                "emails_endpoint": f"/requests/{request_id}/emails",
                "download_endpoint": f"/requests/{request_id}/download"
            },
            "template_info": get_template_info(template_type),
            "next_steps": [
                "Connect to the SSE endpoint to receive real-time updates",
                "Processing will complete in background - no further action needed",
                "Download final results using the download endpoint when processing completes"
            ]
        }
        
    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="CSV file is empty or invalid")
    except pd.errors.ParserError:
        raise HTTPException(status_code=400, detail="Invalid CSV format")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File encoding not supported. Please use UTF-8 encoded CSV")
    except Exception as e:
        # Log the full error for debugging while returning safe message to user
        print(f"Unexpected error in upload_and_process: {str(e)}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500, 
            detail=f"Internal server error occurred while processing your request. Please try again or contact support if the issue persists."
        )


@app.get("/requests/{request_id}/download")
async def download_csv_with_placeholders(request_id: str):
    """
    Download CSV file with UUID placeholders for a specific request
    
    This endpoint allows downloading the CSV file with placeholders after upload.
    Useful for users who want to save the placeholder file or for backwards compatibility.
    """
    try:
        # Get request info
        request = EmailRequestService.get_request(request_id)
        if not request:
            raise HTTPException(status_code=404, detail="Request not found")
        
        # Get all emails for this request
        emails = GeneratedEmailService.get_emails_for_request(request_id)
        if not emails:
            raise HTTPException(status_code=404, detail="No emails found for this request")
        
        # Create DataFrame from database records with 3-column support
        email_data = []
        for email in emails:
            email_data.append({
                'name': email.name,
                'company': email.company,
                'linkedin_url': email.linkedin_url,
                'intelligence': email.intelligence_used,
                'template_type': email.template_type,
                'lexi_email': email.lexi_email if email.status == 'completed' else f"PROCESSING:{email.placeholder_uuid}",
                'lucas_email': email.lucas_email if email.status == 'completed' else f"PROCESSING:{email.placeholder_uuid}",
                'networking_email': email.networking_email if email.status == 'completed' else f"PROCESSING:{email.placeholder_uuid}"
            })
        
        df = pd.DataFrame(email_data)
        
        # Convert to CSV
        output = io.StringIO()
        df.to_csv(output, index=False)
        csv_content = output.getvalue()
        
        # Create streaming response
        def iter_csv():
            yield csv_content
        
        return StreamingResponse(
            iter_csv(),
            media_type="text/csv; charset=utf-8",
            headers={
                "Content-Disposition": f"attachment; filename=generated_{request.original_filename}",
                "X-Request-ID": request_id,
                "Cache-Control": "no-cache, no-store, must-revalidate"
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating download: {str(e)}")


# Database Query Endpoints

@app.get("/requests/{request_id}")
async def get_request_details(request_id: str):
    """
    Get detailed information about a specific request
    """
    try:
        from .database.services import AnalyticsService
        
        summary = AnalyticsService.get_request_summary(request_id)
        if not summary:
            raise HTTPException(status_code=404, detail=f"Request {request_id} not found")
        
        return {"request_summary": summary, "status": "success"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving request: {str(e)}")


@app.get("/requests/{request_id}/emails")
async def get_request_emails(request_id: str, limit: int = 100):
    """
    Get all emails for a specific request
    """
    try:
        from .database.services import GeneratedEmailService
        
        emails = GeneratedEmailService.get_emails_for_request(request_id)
        if not emails:
            raise HTTPException(status_code=404, detail=f"No emails found for request {request_id}")
        
        # Apply limit
        limited_emails = emails[:limit]
        
        # Format response
        formatted_emails = []
        for email in limited_emails:
            formatted_emails.append({
                "id": email.id,
                "uuid": str(email.placeholder_uuid),
                "name": email.name,
                "company": email.company,
                "generated_email": email.generated_email,
                "status": email.status,
                "processing_type": email.processing_type,
                "total_tokens": email.total_tokens,
                "cost_usd": float(email.cost_usd or 0),
                "created_at": email.created_at.isoformat() if email.created_at else None
            })
        
        return {
            "request_id": request_id,
            "total_emails": len(emails),
            "returned_count": len(formatted_emails),
            "emails": formatted_emails
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving emails: {str(e)}")


@app.get("/emails/{email_uuid}")
async def get_email_by_uuid(email_uuid: str):
    """
    Get a specific email by its UUID
    """
    try:
        from .database.services import GeneratedEmailService
        
        email = GeneratedEmailService.get_email_by_uuid(email_uuid)
        if not email:
            raise HTTPException(status_code=404, detail=f"Email with UUID {email_uuid} not found")
        
        return {
            "id": email.id,
            "request_id": email.request_id,
            "uuid": str(email.placeholder_uuid),
            "name": email.name,
            "company": email.company,
            "generated_email": email.generated_email,
            "lexi_email": email.lexi_email,
            "lucas_email": email.lucas_email,
            "networking_email": email.networking_email,
            "status": email.status,
            "processing_type": email.processing_type,
            "total_tokens": email.total_tokens,
            "cost_usd": float(email.cost_usd or 0),
            "created_at": email.created_at.isoformat() if email.created_at else None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving email: {str(e)}")


@app.get("/analytics/performance")
async def get_performance_analytics(days: int = 7):
    """
    Get system performance analytics
    """
    try:
        from .database.services import AnalyticsService
        
        performance_data = AnalyticsService.get_performance_report(days)
        
        return {
            "period_days": days,
            "performance_data": performance_data
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving analytics: {str(e)}")


@app.get("/requests")
async def get_email_requests(
    page: int = Query(1, ge=1, description="Page number starting from 1"),
    limit: int = Query(10, ge=1, le=50, description="Items per page (max 50)"),
    status: Optional[str] = Query(None, description="Filter by status: processing, completed, failed, partial")
):
    """Get paginated list of email requests"""
    try:
        # Calculate offset
        offset = (page - 1) * limit
        
        # Get requests from database
        db_manager = get_database_manager()
        with db_manager.session_scope() as session:
            from .database.models import EmailRequest
            
            # Build query
            query = session.query(EmailRequest)
            
            # Apply status filter if provided
            if status:
                query = query.filter(EmailRequest.status == status)
            
            # Get total count
            total_count = query.count()
            
            # Apply pagination and ordering
            requests = query.order_by(EmailRequest.created_at.desc()).offset(offset).limit(limit).all()
            
            # Convert to dict format
            request_list = []
            for req in requests:
                # Get real-time email counts from GeneratedEmail table
                email_counts = GeneratedEmailService.get_email_counts_for_request(req.request_id)
                
                # Calculate real-time total cost from individual emails
                from .database.models import GeneratedEmail
                from sqlalchemy import func
                total_cost_result = session.query(func.sum(GeneratedEmail.cost_usd))\
                                         .filter(GeneratedEmail.request_id == req.request_id)\
                                         .scalar()
                actual_total_cost = float(total_cost_result or 0)
                
                request_dict = {
                    'request_id': req.request_id,
                    'original_filename': req.original_filename,
                    'file_size_mb': float(req.file_size_mb) if req.file_size_mb else 0,
                    'total_rows': req.total_rows,
                    'template_llm_rows': req.template_llm_rows,
                    'ai_research_rows': req.ai_research_rows,
                    'status': req.status,
                    'fallback_template_type': req.fallback_template_type,
                    'created_at': req.created_at.isoformat() if req.created_at else None,
                    'processing_completed_at': req.processing_completed_at.isoformat() if req.processing_completed_at else None,
                    'total_processing_time_seconds': float(req.total_processing_time_seconds) if req.total_processing_time_seconds else 0,
                    'estimated_cost_usd': actual_total_cost,  # Use calculated real-time cost
                    'successful_emails': email_counts['completed'],  # Use real-time count
                    'failed_emails': email_counts['failed']  # Use real-time count
                }
                request_list.append(request_dict)
            
            # Calculate pagination info
            total_pages = (total_count + limit - 1) // limit
            has_next = page < total_pages
            has_prev = page > 1
            
            return {
                "requests": request_list,
                "pagination": {
                    "current_page": page,
                    "total_pages": total_pages,
                    "total_items": total_count,
                    "items_per_page": limit,
                    "has_next": has_next,
                    "has_prev": has_prev
                }
            }
            
    except Exception as e:
        logger.error(f"Error fetching requests: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch requests: {str(e)}")


@app.get("/requests/{request_id}/details")
async def get_request_details(request_id: str):
    """Get detailed information about a specific request including email samples"""
    try:
        # Get request details
        request_info = EmailRequestService.get_request(request_id)
        if not request_info:
            raise HTTPException(status_code=404, detail="Request not found")
        
        # Get email counts
        email_counts = GeneratedEmailService.get_email_counts_for_request(request_id)
        
        # Calculate real-time total cost from individual emails
        db_manager = get_database_manager()
        with db_manager.session_scope() as session:
            from .database.models import GeneratedEmail
            from sqlalchemy import func
            total_cost_result = session.query(func.sum(GeneratedEmail.cost_usd))\
                                     .filter(GeneratedEmail.request_id == request_id)\
                                     .scalar()
            actual_total_cost = float(total_cost_result or 0)
        
        # Get sample emails (first 5 successful and failed)
        db_manager = get_database_manager()
        with db_manager.session_scope() as session:
            from .database.models import GeneratedEmail
            
            successful_samples = session.query(GeneratedEmail).filter(
                GeneratedEmail.request_id == request_id,
                GeneratedEmail.status == 'completed'
            ).limit(5).all()
            
            failed_samples = session.query(GeneratedEmail).filter(
                GeneratedEmail.request_id == request_id,
                GeneratedEmail.status == 'failed'
            ).limit(5).all()
            
            # Convert to dict
            successful_emails = []
            for email in successful_samples:
                # Get best available email content (prefer 3-column format, fallback to legacy)
                email_content = None
                if email.lexi_email:
                    email_content = email.lexi_email
                elif email.lucas_email:
                    email_content = email.lucas_email
                elif email.networking_email:
                    email_content = email.networking_email
                elif email.generated_email:
                    email_content = email.generated_email
                
                # Truncate for preview
                if email_content and len(email_content) > 200:
                    email_content = email_content[:200] + '...'
                
                successful_emails.append({
                    'email_id': email.id,
                    'placeholder_uuid': str(email.placeholder_uuid) if email.placeholder_uuid else None,
                    'name': email.name,
                    'company': email.company,
                    'template_type': email.template_type,
                    'generated_email': email_content,
                    'processing_time_seconds': float(email.processing_time_seconds) if email.processing_time_seconds else 0,
                    'cost_usd': float(email.cost_usd) if email.cost_usd else 0
                })
            
            failed_emails = []
            for email in failed_samples:
                failed_emails.append({
                    'email_id': email.id,
                    'placeholder_uuid': str(email.placeholder_uuid) if email.placeholder_uuid else None,
                    'name': email.name,
                    'company': email.company,
                    'template_type': email.template_type,
                    'error_message': email.error_message
                })
        
        return {
            "request": {
                'request_id': request_info.request_id,
                'original_filename': request_info.original_filename,
                'file_size_mb': float(request_info.file_size_mb) if request_info.file_size_mb else 0,
                'total_rows': request_info.total_rows,
                'template_llm_rows': request_info.template_llm_rows,
                'ai_research_rows': request_info.ai_research_rows,
                'status': request_info.status,
                'fallback_template_type': request_info.fallback_template_type,
                'created_at': request_info.created_at.isoformat() if request_info.created_at else None,
                'processing_completed_at': request_info.processing_completed_at.isoformat() if request_info.processing_completed_at else None,
                'total_processing_time_seconds': float(request_info.total_processing_time_seconds) if request_info.total_processing_time_seconds else 0,
                'estimated_cost_usd': actual_total_cost  # Use calculated real-time cost
            },
            "email_counts": email_counts,
            "samples": {
                "successful_emails": successful_emails,
                "failed_emails": failed_emails
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching request details: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch request details: {str(e)}")


@app.get("/stream/requests/{request_id}")
async def stream_request_progress(request_id: str):
    """
    Stream real-time updates for email processing using Server-Sent Events (SSE)
    
    This endpoint provides real-time streaming of email processing progress:
    - Connection establishment confirmation
    - Individual email completions with full content
    - Progress updates with completion percentages
    - Final processing completion notification
    - Error handling and status updates
    
    Frontend Usage:
    ```javascript
    const eventSource = new EventSource(`/stream/requests/${requestId}`);
    
    eventSource.addEventListener('email_completed', (event) => {
        const email = JSON.parse(event.data);
        console.log('New email:', email);
    });
    
    eventSource.addEventListener('progress_update', (event) => {
        const progress = JSON.parse(event.data);
        updateProgressBar(progress.percentage);
    });
    ```
    
    Returns: Server-Sent Events stream with real-time processing updates
    """
    
    async def event_stream() -> AsyncGenerator[str, None]:
        try:
            # Check if request exists
            request = EmailRequestService.get_request(request_id)
            if not request:
                yield f"data: {json.dumps({'type': 'error', 'message': 'Request not found'})}\n\n"
                return
            
            # Send initial connection confirmation
            initial_data = {
                'type': 'connection_established',
                'request_id': request_id,
                'timestamp': pd.Timestamp.now().isoformat(),
                'message': 'SSE connection established successfully'
            }
            yield f"data: {json.dumps(initial_data)}\n\n"
            
            # Track progress to detect changes
            last_completed_count = 0
            last_failed_count = 0
            processing_complete = False
            
            while not processing_complete:
                try:
                    # Get current request status
                    current_request = EmailRequestService.get_request(request_id)
                    if not current_request:
                        break
                    
                    # Get email counts directly from database (bypassing broken analytics)
                    email_counts = GeneratedEmailService.get_email_counts_for_request(request_id)
                    if not email_counts:
                        await asyncio.sleep(1)
                        continue
                    
                    current_completed = email_counts.get('completed', 0)
                    current_failed = email_counts.get('failed', 0)
                    total_emails = email_counts.get('total', 0)
                    
                    # Check for newly completed emails
                    if current_completed > last_completed_count:
                        # Get the newly completed emails
                        new_emails = GeneratedEmailService.get_completed_emails_since_count(
                            request_id, last_completed_count
                        )
                        
                        # Send each completed email as separate event
                        for email in new_emails:
                            # Get best available email content (prefer 3-column format, fallback to legacy)
                            email_content = None
                            if email.lexi_email:
                                email_content = email.lexi_email
                            elif email.lucas_email:
                                email_content = email.lucas_email
                            elif email.networking_email:
                                email_content = email.networking_email
                            elif email.generated_email:
                                email_content = email.generated_email
                            
                            email_data = {
                                'type': 'email_completed',
                                'request_id': request_id,
                                'timestamp': pd.Timestamp.now().isoformat(),
                                'data': {
                                    'email_uuid': str(email.placeholder_uuid),
                                    'name': email.name,
                                    'company': email.company,
                                    'linkedin_url': email.linkedin_url,
                                    'generated_email': email_content,
                                    'lexi_email': email.lexi_email,
                                    'lucas_email': email.lucas_email,
                                    'networking_email': email.networking_email,
                                    'processing_time_seconds': float(email.processing_time_seconds) if email.processing_time_seconds else None,
                                    'template_type': email.template_type,
                                    'intelligence_used': email.intelligence_used,
                                    'cost_usd': float(email.cost_usd) if email.cost_usd else None
                                }
                            }
                            yield f"event: email_completed\ndata: {json.dumps(email_data)}\n\n"
                        
                        last_completed_count = current_completed
                    
                    # Check for newly failed emails
                    if current_failed > last_failed_count:
                        # Get failed emails (simplified version)
                        failed_emails = GeneratedEmailService.get_failed_emails_since_count(
                            request_id, last_failed_count
                        )
                        
                        for failed_email in failed_emails:
                            error_data = {
                                'type': 'email_failed',
                                'request_id': request_id,
                                'timestamp': pd.Timestamp.now().isoformat(),
                                'data': {
                                    'email_uuid': str(failed_email.placeholder_uuid),
                                    'name': failed_email.name,
                                    'company': failed_email.company,
                                    'error_message': failed_email.error_message,
                                    'template_type': failed_email.template_type
                                }
                            }
                            yield f"event: email_failed\ndata: {json.dumps(error_data)}\n\n"
                        
                        last_failed_count = current_failed
                    
                    # Send progress update if there were changes
                    if current_completed > 0 or current_failed > 0:
                        progress_percentage = round(((current_completed + current_failed) / total_emails) * 100, 1) if total_emails > 0 else 0
                        
                        progress_data = {
                            'type': 'progress_update',
                            'request_id': request_id,
                            'timestamp': pd.Timestamp.now().isoformat(),
                            'data': {
                                'completed_emails': current_completed,
                                'failed_emails': current_failed,
                                'total_emails': total_emails,
                                'percentage': progress_percentage,
                                'status': current_request.status,
                                'processing_time_seconds': current_request.total_processing_time_seconds or 0
                            }
                        }
                        yield f"event: progress_update\ndata: {json.dumps(progress_data)}\n\n"
                    
                    # Check if processing is complete
                    if current_request.status in ['completed', 'failed', 'partial']:
                        processing_complete = True
                        
                        final_data = {
                            'type': 'processing_complete',
                            'request_id': request_id,
                            'timestamp': pd.Timestamp.now().isoformat(),
                            'data': {
                                'final_status': current_request.status,
                                'total_emails': total_emails,
                                'successful_emails': current_completed,
                                'failed_emails': current_failed,
                                'total_processing_time_seconds': current_request.total_processing_time_seconds or 0,
                                'estimated_cost_usd': current_request.estimated_cost_usd or 0,
                                'completion_message': f"Processing completed: {current_completed}/{total_emails} emails generated successfully"
                            }
                        }
                        yield f"event: processing_complete\ndata: {json.dumps(final_data)}\n\n"
                        
                        # Send explicit close signal and wait for frontend to process
                        close_data = {
                            'type': 'connection_closing',
                            'request_id': request_id,
                            'timestamp': pd.Timestamp.now().isoformat(),
                            'message': 'Server closing SSE connection - processing complete'
                        }
                        yield f"event: connection_closing\ndata: {json.dumps(close_data)}\n\n"
                        
                        # Extended buffer to ensure frontend receives both messages
                        await asyncio.sleep(3)
                        break
                    
                    # Wait before next check (1 second intervals)
                    await asyncio.sleep(1)
                    
                except Exception as inner_e:
                    error_data = {
                        'type': 'error',
                        'request_id': request_id,
                        'timestamp': pd.Timestamp.now().isoformat(),
                        'message': f'Error during processing: {str(inner_e)}'
                    }
                    yield f"event: error\ndata: {json.dumps(error_data)}\n\n"
                    await asyncio.sleep(2)  # Wait longer on errors
                    
        except Exception as e:
            final_error_data = {
                'type': 'fatal_error',
                'request_id': request_id,
                'timestamp': pd.Timestamp.now().isoformat(),
                'message': f'Fatal error in SSE stream: {str(e)}'
            }
            yield f"event: fatal_error\ndata: {json.dumps(final_error_data)}\n\n"
    
    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no", 
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Cache-Control",
            "Content-Type": "text/event-stream"
        }
    )


# Production runner
if __name__ == "__main__":
    import uvicorn
    
    # Configuration from environment variables
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    workers = int(os.getenv("WORKERS", "1"))
    reload = os.getenv("RELOAD", "false").lower() == "true"
    
    print(f"Starting Batch Email Generator on {host}:{port}")
    print(f"Configuration: Batch size={BATCH_SIZE}, Max rows={MAX_CSV_ROWS}")
    print(f"Workers: {workers}, Reload: {reload}")
    
    uvicorn.run(
        "src.main:app",
        host=host,
        port=port,
        workers=workers if not reload else 1,  # Can't use multiple workers with reload
        reload=reload,
        access_log=True,
        log_level="info"
    )