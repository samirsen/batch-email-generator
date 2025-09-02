"""
Utility Functions Module

Common helper functions used across the Batch Email Generator application.
"""

import pandas as pd
from typing import List
import random


def create_batches(df: pd.DataFrame, batch_size: int) -> List[pd.DataFrame]:
    """
    Split DataFrame into batches of specified size
    
    Args:
        df: DataFrame to split into batches
        batch_size: Number of rows per batch
        
    Returns:
        List of DataFrame batches
    """
    batches = []
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i + batch_size].copy()
        batches.append(batch)
    return batches


def merge_results_in_order(original_df: pd.DataFrame, results: dict) -> List[str]:
    """
    Merge results back in original DataFrame order
    
    Args:
        original_df: Original DataFrame with correct row order
        results: Dictionary mapping row indices to email content
        
    Returns:
        List of email strings in original order
    """
    ordered_emails = []
    for _, row in original_df.iterrows():
        email = results.get(row.name, "Error: No result generated")
        ordered_emails.append(email)
    
    return ordered_emails


def calculate_file_size_mb(content: str) -> float:
    """
    Calculate file size in MB from string content
    
    Args:
        content: String content to measure
        
    Returns:
        File size in megabytes
    """
    size_bytes = len(content.encode('utf-8'))
    return round(size_bytes / (1024 * 1024), 2)


def format_processing_stats(total_rows: int, ai_rows: int, template_rows: int) -> dict:
    """
    Format processing statistics for responses
    
    Args:
        total_rows: Total number of rows processed
        ai_rows: Number of AI-processed rows
        template_rows: Number of template-processed rows
        
    Returns:
        Dictionary with formatted statistics
    """
    return {
        "total_rows": total_rows,
        "ai_rows": ai_rows,
        "template_rows": template_rows,
        "ai_percentage": round((ai_rows / total_rows) * 100, 1) if total_rows > 0 else 0,
        "template_percentage": round((template_rows / total_rows) * 100, 1) if total_rows > 0 else 0
    }


def generate_request_id() -> str:
    """
    Generate a short, unique request ID for tracking
    
    Returns:
        8-character UUID string
    """
    import uuid
    return str(uuid.uuid4())[:8]


def generate_placeholder_uuid() -> str:
    """
    Generate a full UUID for AI email placeholders
    
    Returns:
        Full UUID string
    """
    import uuid
    return str(uuid.uuid4())


# Agent information for email generation
AGENT_NAMES = [
    "Sarah Chen",
    "Alex Rodriguez",
    "Emily Johnson",
    "Michael Thompson", 
    "Jessica Williams",
    "David Kim",
    "Rachel Martinez",
    "James Anderson",
    "Amanda Davis",
    "Ryan O'Connor",
    "Lisa Zhang",
    "Mark Stevens",
    "Nicole Brown",
    "Chris Wilson",
    "Samantha Taylor"
]

COMPANY_NAME = "FlairLabs Inc."


def get_random_agent_info() -> dict:
    """
    Get random agent information for email generation
    
    Returns:
        Dictionary with agent name and company information
    """
    return {
        "agent_name": random.choice(AGENT_NAMES),
        "company_name": COMPANY_NAME
    }
