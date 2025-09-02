"""
CSV Processing Module

Handles CSV validation, enhancement, and processing logic for the Batch Email Generator.
"""

import pandas as pd
from fastapi import HTTPException
from typing import List


def validate_and_enhance_csv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate CSV columns and add missing optional columns with defaults
    
    Required columns: name, company, linkedin_url
    Optional columns: intelligence (defaults to False), template_type (defaults to None)
    
    Args:
        df: Input DataFrame from CSV
        
    Returns:
        Enhanced DataFrame with all required columns
        
    Raises:
        HTTPException: If required columns are missing
    """
    # Validate required columns
    required_columns = ['name', 'company', 'linkedin_url']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise HTTPException(
            status_code=400, 
            detail=f"Missing required columns: {', '.join(missing_columns)}. Required: {', '.join(required_columns)}"
        )
    
    # Add optional columns with defaults if missing
    if 'intelligence' not in df.columns:
        df['intelligence'] = False
    else:
        # Convert intelligence column to boolean, handling various formats
        df['intelligence'] = df['intelligence'].astype(str).str.lower().isin(['true', '1', 'yes', 'y'])
    
    if 'template_type' not in df.columns:
        df['template_type'] = ''  # Will use default template
    else:
        # Clean template_type values - handle NaN and empty strings
        df['template_type'] = df['template_type'].fillna('')  # Replace NaN with empty string
        df['template_type'] = df['template_type'].astype(str)  # Ensure string type
        df['template_type'] = df['template_type'].replace('nan', '')  # Handle string 'nan'
    
    return df


def get_csv_info(df: pd.DataFrame) -> dict:
    """
    Get information about the CSV structure and content
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary with CSV metadata
    """
    ai_rows = df[df['intelligence'] == True]
    template_rows = df[df['intelligence'] == False]
    
    return {
        "total_rows": len(df),
        "ai_rows": len(ai_rows),
        "template_rows": len(template_rows),
        "columns": list(df.columns),
        "ai_percentage": round((len(ai_rows) / len(df)) * 100, 1) if len(df) > 0 else 0
    }


def split_dataframe_by_intelligence(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split DataFrame into AI and template processing groups
    
    Args:
        df: Input DataFrame with intelligence column
        
    Returns:
        Tuple of (ai_rows, template_rows) DataFrames
    """
    ai_rows = df[df['intelligence'] == True].copy()
    template_rows = df[df['intelligence'] == False].copy()
    
    return ai_rows, template_rows


def validate_csv_size(df: pd.DataFrame, max_rows: int) -> None:
    """
    Validate CSV size against maximum allowed rows
    
    Args:
        df: Input DataFrame
        max_rows: Maximum allowed rows
        
    Raises:
        HTTPException: If CSV exceeds size limit
    """
    if len(df) > max_rows:
        raise HTTPException(
            status_code=400, 
            detail=f"CSV too large. Maximum allowed rows: {max_rows}, received: {len(df)}"
        )


def validate_csv_not_empty(df: pd.DataFrame) -> None:
    """
    Validate that CSV is not empty
    
    Args:
        df: Input DataFrame
        
    Raises:
        HTTPException: If CSV is empty
    """
    if df.empty:
        raise HTTPException(status_code=400, detail="CSV file is empty")
