"""
SixtyFour API Integration Module

This module provides integration with the SixtyFour API for company enrichment data.
"""

import os
import aiohttp
import json
from typing import Dict, Any, Optional

# SixtyFour API configuration
SIXTYFOUR_API_KEY = os.getenv("SIXTYFOUR_API_KEY", "api_rsaNdiPCpBrpGPUMqLx43mMEqWmhLZNN")
SIXTYFOUR_API_BASE_URL = os.getenv("SIXTYFOUR_API_BASE_URL", "https://api.sixtyfour.ai")

class SixtyFourAPI:
    """
    Client for the SixtyFour API for company enrichment data
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the SixtyFour API client
        
        Args:
            api_key: SixtyFour API key (defaults to environment variable)
        """
        self.api_key = api_key or SIXTYFOUR_API_KEY
        self.base_url = SIXTYFOUR_API_BASE_URL
        
    async def get_company_data(self, company_name: str) -> Dict[str, Any]:
        """
        Get company enrichment data from SixtyFour API
        
        Args:
            company_name: Name of the company to look up
            
        Returns:
            Dictionary with company data including industry/vertical
        """
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            # Endpoint for company enrichment (adjust based on actual API documentation)
            endpoint = f"{self.base_url}/v1/companies/enrich"
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    endpoint,
                    headers=headers,
                    json={"company_name": company_name}
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        error_text = await response.text()
                        print(f"SixtyFour API error ({response.status}): {error_text}")
                        return {"error": f"API error: {response.status}", "company_name": company_name}
                        
        except Exception as e:
            print(f"Error calling SixtyFour API: {str(e)}")
            return {"error": str(e), "company_name": company_name}
    
    def extract_company_vertical(self, company_data: Dict[str, Any]) -> str:
        """
        Extract company vertical/industry from API response
        
        Args:
            company_data: Company data from SixtyFour API
            
        Returns:
            Company vertical as string
        """
        # Extract industry/vertical from API response
        # Adjust this based on the actual API response structure
        if "error" in company_data:
            # Fallback to a generic vertical if API call failed
            return "technology"
            
        # Try to extract vertical from different possible fields
        # This is a guess at the API structure - adjust based on actual API docs
        vertical = company_data.get("industry") or company_data.get("vertical") or company_data.get("sector")
        
        if not vertical and "categories" in company_data:
            categories = company_data.get("categories", [])
            if categories and len(categories) > 0:
                vertical = categories[0]
                
        # Default fallback
        return vertical or "technology"
    
    def is_app_layer_company(self, company_data: Dict[str, Any]) -> bool:
        """
        Determine if a company is an app-layer company based on its vertical/industry
        
        Args:
            company_data: Company data from SixtyFour API
            
        Returns:
            Boolean indicating if it's an app-layer company
        """
        vertical = self.extract_company_vertical(company_data).lower()
        
        # App-layer indicators in the vertical/industry
        app_layer_keywords = [
            "software", "saas", "application", "app", "platform", "tech", 
            "technology", "digital", "cloud", "enterprise software"
        ]
        
        return any(keyword in vertical for keyword in app_layer_keywords)
    
    def determine_one_liner(self, company_data: Dict[str, Any]) -> str:
        """
        Determine the appropriate one-liner based on company data
        
        Args:
            company_data: Company data from SixtyFour API
            
        Returns:
            One-liner string for the email
        """
        # Default one-liner
        default_one_liner = "Congrats on everything to-date."
        
        # Check for signals in the company data
        # This is a placeholder - adjust based on actual API response structure
        if "funding" in company_data and company_data.get("funding", {}).get("recent", False):
            return "Congrats on the recent round announcement."
            
        if "product" in company_data and company_data.get("product", {}).get("recent_launch", False):
            return "Congrats on the new launch — exciting milestone."
            
        if "reputation" in company_data and company_data.get("reputation", {}).get("strong", False):
            return "You've built a strong reputation in the space over time."
            
        # Default to momentum
        return default_one_liner


# Module-level convenience functions
async def get_company_enrichment(company_name: str) -> Dict[str, Any]:
    """
    Convenience function to get company enrichment data
    
    Args:
        company_name: Name of the company to look up
        
    Returns:
        Dictionary with company data
    """
    client = SixtyFourAPI()
    return await client.get_company_data(company_name)


async def get_email_variables(company_name: str) -> Dict[str, Any]:
    """
    Get email template variables based on company data
    
    Args:
        company_name: Name of the company to look up
        
    Returns:
        Dictionary with template variables
    """
    client = SixtyFourAPI()
    company_data = await client.get_company_data(company_name)
    
    # Extract company vertical
    company_vertical = client.extract_company_vertical(company_data)
    
    # Determine if it's an app-layer company
    app_layer = client.is_app_layer_company(company_data)
    
    # Determine one-liner based on company signals
    one_liner = client.determine_one_liner(company_data)
    
    # Sample portfolio companies based on vertical
    portfolio_companies = get_portfolio_companies_by_vertical(company_vertical)
    
    return {
        "company_vertical": company_vertical,
        "app_layer": app_layer,
        "one_liner": one_liner,
        "portfolio_companies": portfolio_companies,
        "include_tldr": False,  # Default to not include TLDR
        "tldr_block": ""  # Empty TLDR block by default
    }


def get_portfolio_companies_by_vertical(vertical: str) -> str:
    """
    Get relevant portfolio companies based on company vertical
    
    Args:
        vertical: Company vertical/industry
        
    Returns:
        String of relevant portfolio companies
    """
    # Map of verticals to relevant portfolio companies
    vertical_to_companies = {
        "ai": "OpenAI, Anthropic, and Hugging Face",
        "fintech": "Robinhood, Stripe, and Plaid",
        "healthcare": "Ro, Calm, and Forward",
        "enterprise": "Airtable, Notion, and Slack",
        "consumer": "Airbnb, Spotify, and Uber",
        "ecommerce": "Shopify, Faire, and Bolt",
        "crypto": "Coinbase, FTX, and Alchemy",
        "security": "Cloudflare, Auth0, and Snyk",
        "technology": "GitHub, Figma, and Vercel",
        "software": "Zoom, Asana, and Monday.com",
        "saas": "Salesforce, HubSpot, and Zendesk"
    }
    
    # Normalize vertical and find matching portfolio companies
    vertical_lower = vertical.lower()
    
    for key, companies in vertical_to_companies.items():
        if key in vertical_lower:
            return companies
    
    # Default portfolio companies if no match
    return "Airbnb, Spotify, and Uber"

