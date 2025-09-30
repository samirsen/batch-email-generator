"""
Background Processing Module

Handles background processing of AI emails and JSON logging of results.
"""

import time
import json
import uuid
import asyncio
import pandas as pd
from datetime import datetime
from typing import Optional

# Database imports
from .database.services import (
    EmailRequestService, GeneratedEmailService, ProcessingBatchService,
    ProcessingErrorService, SystemMetricService, AnalyticsService
)
from .email_generator import process_ai_dataframe, process_template_dataframe
from .templates import TemplateType


def log_ai_results_to_json(request_id: str, ai_results: dict, original_ai_rows: pd.DataFrame, processing_time: float, ai_uuid_mapping: Optional[dict] = None):
    """
    Log completed AI email results to structured JSON file
    
    Args:
        request_id: Unique identifier for the request
        ai_results: Dictionary of generated email results
        original_ai_rows: Original AI rows DataFrame
        processing_time: Time taken to process the AI emails
        ai_uuid_mapping: Mapping of row indices to UUIDs from CSV placeholders
    """
    try:
        # Prepare structured data
        log_entry = {
            "request_id": request_id,
            "timestamp": datetime.now().isoformat(),
            "processing_time_seconds": round(processing_time, 2),
            "total_ai_emails": len(ai_results),
            "results": []
        }
        
        # Add each AI result
        for _, row in original_ai_rows.iterrows():
            # Use the same UUID that was used in the CSV placeholder
            row_uuid = ai_uuid_mapping.get(row.name) if ai_uuid_mapping else str(uuid.uuid4())
            
            result_data = {
                "uuid": row_uuid,  # Use the same UUID from CSV placeholder
                "name": str(row['name']),
                "company": str(row['company']),
                "linkedin_url": str(row['linkedin_url']),
                "template_type": str(row.get('template_type', '')),
                "generated_email": ai_results.get(row.name, "Error: No result generated"),
                "row_index": int(row.name) if pd.notna(row.name) else None
            }
            log_entry["results"].append(result_data)
        
        # Write to JSON file in root folder
        filename = f"ai_results_{request_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(log_entry, f, indent=2, ensure_ascii=False)
        
        print(f"AI results logged to {filename}")
        
    except Exception as e:
        print(f"Error logging AI results: {str(e)}")


def log_ai_error_to_json(request_id: str, error: Exception, ai_rows: pd.DataFrame):
    """
    Log AI processing errors to JSON file
    
    Args:
        request_id: Unique identifier for the request
        error: The exception that occurred
        ai_rows: Original AI rows DataFrame
    """
    try:
        error_entry = {
            "request_id": request_id,
            "timestamp": datetime.now().isoformat(),
            "error": str(error),
            "status": "failed",
            "total_ai_emails": len(ai_rows)
        }
        
        filename = f"ai_error_{request_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(error_entry, f, indent=2, ensure_ascii=False)
        print(f"AI error logged to {filename}")
    except Exception as log_error:
        print(f"Failed to log AI error: {str(log_error)}")


def log_unified_error_to_json(request_id: str, error: Exception, all_rows: pd.DataFrame):
    """
    Log unified processing errors to JSON file
    
    Args:
        request_id: Unique identifier for the request
        error: The exception that occurred
        all_rows: All rows DataFrame
    """
    try:
        error_entry = {
            "request_id": request_id,
            "timestamp": datetime.now().isoformat(),
            "error": str(error),
            "status": "failed",
            "processing_method": "unified_llm",
            "total_emails": len(all_rows)
        }
        
        filename = f"unified_error_{request_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(error_entry, f, indent=2, ensure_ascii=False)
        print(f"Unified processing error logged to {filename}")
    except Exception as log_error:
        print(f"Failed to log unified processing error: {str(log_error)}")


async def process_ai_emails_background(request_id: str, ai_rows: pd.DataFrame, fallback_template_type: Optional[TemplateType] = None, ai_uuid_mapping: Optional[dict] = None):
    """
    Background processing of AI emails with JSON logging
    
    Args:
        request_id: Unique identifier for the request
        ai_rows: DataFrame containing AI rows to process
        fallback_template_type: Template type to use when row template_type is empty
        ai_uuid_mapping: Mapping of row indices to UUIDs from CSV placeholders
    """
    try:
        print(f"Starting background AI processing for request {request_id}")
        start_time = time.time()
        
        # Process AI emails
        ai_results = await process_ai_dataframe(ai_rows, fallback_template_type)
        
        processing_time = time.time() - start_time
        print(f"Background AI processing completed for request {request_id} in {processing_time:.2f}s")
        
        # Log results to JSON
        log_ai_results_to_json(request_id, ai_results, ai_rows, processing_time, ai_uuid_mapping)
        
    except Exception as e:
        print(f"Background AI processing failed for request {request_id}: {str(e)}")
        
        # Log error to JSON
        log_ai_error_to_json(request_id, e, ai_rows)


async def process_all_emails_background(
    request_id: str, 
    all_rows: pd.DataFrame, 
    fallback_template_type: Optional[TemplateType] = None, 
    uuid_mapping: Optional[dict] = None
):
    """
    Unified background processing for ALL emails with 3-column generation (LEXI, LUCAS, NETWORKING)
    
    Args:
        request_id: Unique identifier for the request
        all_rows: DataFrame containing all rows to process (both AI and template)
        fallback_template_type: Template type to use when row template_type is empty
        uuid_mapping: Mapping of row indices to UUIDs from CSV placeholders
    """
    try:
        print(f"Starting unified background processing for request {request_id}")
        print(f"Generating 3 email variations (LEXI, LUCAS, NETWORKING) for {len(all_rows)} rows...")
        start_time = time.time()
        
        # Template types for 3-column generation
        template_types = [TemplateType.LEXI, TemplateType.LUCAS, TemplateType.NETWORKING]
        template_columns = ['lexi_email', 'lucas_email', 'networking_email']
        
        # Results storage for all 3 templates
        all_results = {}
        
        # Step 1: Research ALL companies in parallel (Parallel AI + GPT together)
        print("Step 1: Researching all companies in parallel (Parallel AI + GPT)...")
        research_tasks = []
        user_infos = []
        
        for i, row in all_rows.iterrows():
            user_info = {
                'name': row.get('name', ''),
                'company': row.get('company', ''),
                'linkedin_url': row.get('linkedin_url', '')
            }
            user_infos.append((i, user_info))
            
            # Create parallel research task - complete Parallel AI + GPT for each company
            from .gpt_enrichment import ParallelAIEnrichment
            enrichment_client = ParallelAIEnrichment()
            task = enrichment_client.get_company_data(
                user_info['company'], 
                user_info['linkedin_url']
            )
            research_tasks.append(task)
        
        # Execute all company research in parallel (each does Parallel AI + GPT)
        company_data_results = await asyncio.gather(*research_tasks, return_exceptions=True)
        print(f"✅ Completed parallel research for {len(company_data_results)} companies")
        
        # Step 2: Generate ALL emails using pre-fetched data
        print("Step 2: Generating all emails using pre-fetched company data...")
        
        async def process_single_row(row_index, user_info, company_data):
            """Process a single row with all 3 email types in parallel"""
            print(f"Processing row {row_index+1}/{len(all_rows)}: {user_info['company']}")
            
            if isinstance(company_data, Exception):
                print(f"❌ Error getting company data for {user_info['company']}: {company_data}")
                company_data = None
            else:
                print(f"✅ Using pre-fetched data for {user_info['company']}")
            
            # Generate all 3 email types in parallel for this row
            email_tasks = []
            for template_type in template_types:
                from .lexie_prompt import get_ai_email_response
                task = get_ai_email_response(
                    company_name=user_info['company'],
                    recipient_name=user_info['name'],
                    template_type=template_type,
                    company_website=user_info['linkedin_url'],
                    company_data=company_data
                )
                email_tasks.append((template_type, task))
            
            # Wait for all 3 emails for this row
            email_results = await asyncio.gather(*[task for _, task in email_tasks], return_exceptions=True)
            
            # Build row results
            row_results = {}
            for (template_type, _), email_content in zip(email_tasks, email_results):
                column_name = template_columns[template_types.index(template_type)]
                
                if isinstance(email_content, Exception):
                    error_msg = f"Error generating {template_type.value} email: {str(email_content)}"
                    row_results[column_name] = error_msg
                    print(f"❌ {error_msg}")
                else:
                    row_results[column_name] = email_content
                    print(f"✅ Generated {template_type.value} email for {user_info['company']}")
                    print(f"📧 EMAIL CONTENT for {user_info['name']} at {user_info['company']} ({template_type.value}):")
                    print("=" * 60)
                    print(email_content)
                    print("=" * 60)
            
            return row_index, row_results
        
        # Process all rows in parallel
        row_tasks = []
        for (row_index, user_info), company_data in zip(user_infos, company_data_results):
            task = process_single_row(row_index, user_info, company_data)
            row_tasks.append(task)
        
        # Execute all row processing in parallel
        row_results_list = await asyncio.gather(*row_tasks)
        
        # Collect results
        for row_index, row_results in row_results_list:
            all_results[row_index] = row_results
        
        processing_time = time.time() - start_time
        print(f"Unified background processing completed for request {request_id} in {processing_time:.2f}s")
        print(f"Generated {len(all_results)} rows × 3 templates = {len(all_results) * 3} total emails")
        
        # Save 3-column results to database using existing function
        from .email_generator import save_batch_to_database
        save_batch_to_database(request_id, all_results, all_rows, uuid_mapping)
        
        # Save results to JSON with 3-column structure
        log_unified_3column_results_to_json(request_id, all_results, all_rows, processing_time, uuid_mapping)
        
        # Update request status
        try:
            EmailRequestService.update_request_status(request_id, 'completed', processing_time)
            print(f"✓ Updated request {request_id} status to completed")
        except Exception as e:
            print(f"✗ Failed to update request status: {str(e)}")
        
    except Exception as e:
        print(f"Unified background processing failed for request {request_id}: {str(e)}")
        
        # Log error to JSON
        log_unified_error_to_json(request_id, e, all_rows)


def log_unified_results_to_database(
    request_id: str, 
    all_results: dict, 
    original_rows: pd.DataFrame, 
    processing_time: float, 
    uuid_mapping: Optional[dict] = None
):
    """
    Save completed unified email results to database
    
    Args:
        request_id: Unique identifier for the request
        all_results: Dictionary of all generated email results (row_index -> email_content)
        original_rows: Original rows DataFrame  
        processing_time: Time taken to process all emails
        uuid_mapping: Mapping of row indices to UUIDs from CSV placeholders
    """
    try:
        successful_count = 0
        failed_count = 0
        total_tokens = 0
        total_cost = 0.0
        
        # Update each email record with results
        for _, row in original_rows.iterrows():
            row_index = row.name
            generated_content = all_results.get(row_index, "")
            
            # Find the corresponding database record by UUID
            row_uuid = uuid_mapping.get(row_index) if uuid_mapping else None
            if row_uuid:
                email_record = GeneratedEmailService.get_email_by_uuid(str(row_uuid))
                if email_record:
                    # Determine if generation was successful
                    is_successful = bool(generated_content and not generated_content.startswith("Error:"))
                    
                    # Estimate tokens and cost based on realistic GPT-4o-mini pricing
                    estimated_tokens = len(generated_content.split()) * 1.3 if generated_content else 0  # Rough token estimate
                    # GPT-4o-mini: ~$0.15/1M input + $0.60/1M output tokens, average ~$0.3/1M tokens
                    estimated_cost = estimated_tokens * 0.0000003  # Realistic cost estimate
                    
                    # Update the email record
                    GeneratedEmailService.update_email_completion(
                        email_id=email_record.id,
                        generated_email=generated_content if is_successful else "",
                        llm_model="gpt-4o-mini",  # Default model
                        prompt_tokens=int(estimated_tokens * 0.8),
                        completion_tokens=int(estimated_tokens * 0.2),
                        processing_time=processing_time / len(original_rows),  # Average time per email
                        cost=estimated_cost,
                        error_message=generated_content if not is_successful else None
                    )
                    
                    if is_successful:
                        successful_count += 1
                        total_tokens += estimated_tokens
                        total_cost += estimated_cost
                    else:
                        failed_count += 1
        
        # Update the request with final results
        EmailRequestService.update_request_completion(
            request_id=request_id,
            status="completed" if failed_count == 0 else ("partial" if successful_count > 0 else "failed"),
            successful_emails=successful_count,
            failed_emails=failed_count,
            total_tokens=int(total_tokens),
            estimated_cost=total_cost,
            processing_time=processing_time
        )
        
        # Record system metrics
        SystemMetricService.record_metric("emails_processed", len(all_results), "count", request_id)
        SystemMetricService.record_metric("processing_time", processing_time, "seconds", request_id)
        SystemMetricService.record_metric("tokens_used", total_tokens, "tokens", request_id)
        SystemMetricService.record_metric("processing_cost", total_cost, "usd", request_id)
        
        print(f"Unified results saved to database: {successful_count} successful, {failed_count} failed")
        
        # Also create a JSON backup for compatibility (optional)
        try:
            log_unified_results_to_json_backup(request_id, all_results, original_rows, processing_time, uuid_mapping)
        except Exception as json_error:
            print(f"Warning: JSON backup failed: {json_error}")
        
    except Exception as e:
        print(f"Error saving unified results to database: {str(e)}")
        # Fall back to JSON logging if database fails
        log_unified_results_to_json_backup(request_id, all_results, original_rows, processing_time, uuid_mapping)


def log_unified_results_to_json_backup(
    request_id: str, 
    all_results: dict, 
    original_rows: pd.DataFrame, 
    processing_time: float, 
    uuid_mapping: Optional[dict] = None
):
    """
    JSON backup logging (fallback when database fails)
    """
    try:
        # Prepare structured data
        log_entry = {
            "request_id": request_id,
            "timestamp": datetime.now().isoformat(),
            "processing_time_seconds": round(processing_time, 2),
            "total_emails": len(all_results),
            "processing_method": "unified_llm",
            "database_fallback": True,  # Indicate this was a fallback
            "results": []
        }
        
        # Add each result
        for _, row in original_rows.iterrows():
            row_uuid = uuid_mapping.get(row.name) if uuid_mapping else str(uuid.uuid4())
            processing_type = "ai_with_research" if row.get('intelligence', False) else "template_llm"
            
            result_data = {
                "uuid": row_uuid,
                "processing_type": processing_type,
                "name": str(row['name']),
                "company": str(row['company']),
                "linkedin_url": str(row['linkedin_url']),
                "template_type": str(row.get('template_type', '')),
                "intelligence_used": bool(row.get('intelligence', False)),
                "generated_email": all_results.get(row.name, "Error: No result generated"),
                "row_index": int(row.name) if pd.notna(row.name) else None
            }
            log_entry["results"].append(result_data)
        
        # Write to JSON file in root folder
        filename = f"unified_results_{request_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(log_entry, f, indent=2, ensure_ascii=False)
        
        print(f"Backup JSON results logged to {filename}")
        
    except Exception as e:
        print(f"Error logging backup JSON results: {str(e)}")


def log_unified_3column_results_to_json(
    request_id: str, 
    all_results: dict, 
    original_rows: pd.DataFrame, 
    processing_time: float, 
    uuid_mapping: Optional[dict] = None
):
    """
    Log 3-column email results to JSON file
    
    Args:
        request_id: Unique identifier for the request
        all_results: Dictionary of results with structure {row_index: {lexi_email: str, lucas_email: str, networking_email: str}}
        original_rows: Original rows DataFrame
        processing_time: Time taken to process all emails
        uuid_mapping: Mapping of row indices to UUIDs from CSV placeholders
    """
    try:
        # Prepare structured data
        log_entry = {
            "request_id": request_id,
            "timestamp": datetime.now().isoformat(),
            "processing_time_seconds": round(processing_time, 2),
            "total_rows": len(all_results),
            "total_emails": len(all_results) * 3,  # 3 emails per row
            "processing_method": "unified_3column_background",
            "template_types": ["LEXI", "LUCAS", "NETWORKING"],
            "results": []
        }
        
        # Add each row result
        for _, row in original_rows.iterrows():
            row_index = row.name
            row_uuid = uuid_mapping.get(row_index) if uuid_mapping else str(uuid.uuid4())
            row_emails = all_results.get(row_index, {})
            
            result_data = {
                "uuid": row_uuid,
                "row_index": int(row_index) if pd.notna(row_index) else None,
                "name": str(row['name']),
                "company": str(row['company']),
                "linkedin_url": str(row['linkedin_url']),
                "intelligence_used": bool(row.get('intelligence', False)),
                "template_type": str(row.get('template_type', '')),
                "emails": {
                    "lexi_email": row_emails.get('lexi_email', 'Error: No result generated'),
                    "lucas_email": row_emails.get('lucas_email', 'Error: No result generated'),
                    "networking_email": row_emails.get('networking_email', 'Error: No result generated')
                }
            }
            log_entry["results"].append(result_data)
        
        # Write to JSON file in root folder
        filename = f"unified_3column_results_{request_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(log_entry, f, indent=2, ensure_ascii=False)
        
        print(f"3-column results logged to {filename}")
        
    except Exception as e:
        print(f"Error logging 3-column results: {str(e)}")


def create_ai_placeholders(ai_rows: pd.DataFrame) -> tuple[dict, dict]:
    """
    Create UUID placeholders for AI emails and return mapping
    
    Args:
        ai_rows: DataFrame containing AI rows
        
    Returns:
        Tuple of (ai_placeholders dict, ai_uuid_mapping dict)
    """
    ai_placeholders = {}
    ai_uuid_mapping = {}
    
    if not ai_rows.empty:
        print(f"Adding placeholder UUIDs for {len(ai_rows)} AI emails...")
        for _, row in ai_rows.iterrows():
            placeholder_uuid = str(uuid.uuid4())
            ai_placeholders[row.name] = f"AI_PROCESSING:{placeholder_uuid}"
            ai_uuid_mapping[row.name] = placeholder_uuid  # Store for JSON logging
    
    return ai_placeholders, ai_uuid_mapping


def create_template_placeholders(template_rows: pd.DataFrame) -> tuple[dict, dict]:
    """
    Create UUID placeholders for template LLM emails and return mapping
    
    Args:
        template_rows: DataFrame containing template rows (now LLM processed)
        
    Returns:
        Tuple of (template_placeholders dict, template_uuid_mapping dict)
    """
    template_placeholders = {}
    template_uuid_mapping = {}
    
    if not template_rows.empty:
        print(f"Adding placeholder UUIDs for {len(template_rows)} template LLM emails...")
        for _, row in template_rows.iterrows():
            placeholder_uuid = str(uuid.uuid4())
            template_placeholders[row.name] = f"TEMPLATE_LLM_PROCESSING:{placeholder_uuid}"
            template_uuid_mapping[row.name] = placeholder_uuid
    
    return template_placeholders, template_uuid_mapping


def create_all_placeholders(all_rows: pd.DataFrame) -> tuple[dict, dict]:
    """
    Create UUID placeholders for ALL emails (both AI and template now use LLM)
    
    Args:
        all_rows: DataFrame containing all rows to process with LLM
        
    Returns:
        Tuple of (all_placeholders dict, all_uuid_mapping dict)
    """
    all_placeholders = {}
    all_uuid_mapping = {}
    
    if not all_rows.empty:
        print(f"Adding placeholder UUIDs for {len(all_rows)} emails (all LLM processed)...")
        for _, row in all_rows.iterrows():
            placeholder_uuid = str(uuid.uuid4())
            
            # Determine processing type based on intelligence column
            processing_type = "AI_PROCESSING" if row.get('intelligence', False) else "TEMPLATE_LLM_PROCESSING"
            all_placeholders[row.name] = f"{processing_type}:{placeholder_uuid}"
            all_uuid_mapping[row.name] = placeholder_uuid
    
    return all_placeholders, all_uuid_mapping
