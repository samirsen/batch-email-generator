from datetime import datetime
import os
import aiohttp
import json
from typing import Dict, Any, Optional

SIXTYFOUR_API_KEY = os.getenv("SIXTYFOUR_API_KEY", "")
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
        
    async def get_company_data(self, company_name: str, company_website: str = None) -> Dict[str, Any]:
        """
        Get company enrichment data from SixtyFour API
        
        Args:
            company_name: Name of the company to look up
            company_website: Website URL for better context (optional)
            
        Returns:
            Dictionary with company data including industry/vertical
        """
        try:
            if not self.api_key:
                return {"error": "No API key configured", "company_name": company_name}
                
            headers = {
                "x-api-key": self.api_key,
                "Content-Type": "application/json"
            }
            
            
            endpoint = f"{self.base_url}/enrich-company"
            
            target_company = {
                "company_name": company_name
            }
            if company_website:
                target_company["website"] = company_website
            
            struct = {
                "industry": "Primary industry or sector",
                "vertical": "Business vertical or category",
                "company_description": "Brief description of what the company does",
                "founded_year": {"description": "Year the company was founded", "type": "int"},
                "recent_news": {"description": "Recent press or announcements", "type": "list[string]"},
                "funding_rounds": {"description": "Funding rounds and amounts", "type": "list[string]"},
                "linkedin_url": "LinkedIn company page",
            }
            
            payload = {
                "target_company": target_company,
                "struct": struct,
                "find_people": False
            }
            
            
            # The API can take 5-10 minutes to respond
            timeout = aiohttp.ClientTimeout(total=600, sock_read=600, sock_connect=30)
            
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    endpoint,
                    headers=headers,
                    json=payload
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get("structured_data", {})
                    else:
                        error_text = await response.text()
                        return {"error": f"API error: {response.status}", "company_name": company_name}
                        
        except Exception as e:
            # Return minimal fallback data so email generation can continue
            return {
                "error": str(e), 
                "company_name": company_name,
                "industry": "technology",
                "vertical": "software"
            }
    
    def extract_company_vertical(self, company_data: Dict[str, Any]) -> str:
        """
        Extract company vertical/industry from API response
        
        Args:
            company_data: Company data from SixtyFour API (structured_data portion)
            
        Returns:
            Company vertical as string
        """
        if "error" in company_data:
            # Fallback to a generic vertical if API call failed
            return "technology"
            
        vertical = company_data.get("industry") or company_data.get("vertical")
        
        if not vertical and "company_description" in company_data:
            description = company_data.get("company_description", "").lower()
            if any(keyword in description for keyword in ["ai", "artificial intelligence", "machine learning"]):
                vertical = "ai"
            elif any(keyword in description for keyword in ["fintech", "financial", "banking", "payments"]):
                vertical = "fintech"
            elif any(keyword in description for keyword in ["healthcare", "medical", "health"]):
                vertical = "healthcare"
            elif any(keyword in description for keyword in ["enterprise", "b2b", "business"]):
                vertical = "enterprise"
            elif any(keyword in description for keyword in ["consumer", "b2c", "retail"]):
                vertical = "consumer"
            elif any(keyword in description for keyword in ["ecommerce", "e-commerce", "marketplace"]):
                vertical = "ecommerce"
            elif any(keyword in description for keyword in ["crypto", "blockchain", "web3"]):
                vertical = "crypto"
            elif any(keyword in description for keyword in ["security", "cybersecurity", "cyber"]):
                vertical = "security"
            elif any(keyword in description for keyword in ["software", "saas", "platform"]):
                vertical = "software"
                
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
        
        app_layer_keywords = [
            "software", "saas", "application", "app", "platform", "tech", 
            "technology", "digital", "cloud", "enterprise software"
        ]
        
        return any(keyword in vertical for keyword in app_layer_keywords)
    
    def determine_one_liner(self, company_data: Dict[str, Any]) -> str:
        """
        Determine the appropriate one-liner based on company data
        
        Args:
            company_data: Company data from SixtyFour API (structured_data portion)
            
        Returns:
            One-liner string for the email
        """
        default_one_liner = "Congrats on everything to-date."
        
        
        funding_rounds = company_data.get("funding_rounds", [])
        if funding_rounds and len(funding_rounds) > 0:
            # Check if any funding round mentions recent keywords
            recent_funding_keywords = ["2025","2024", "series", "seed", "round", "raised"]
            for round_info in funding_rounds:
                if isinstance(round_info, str) and any(keyword in round_info.lower() for keyword in recent_funding_keywords):
                    return "Congrats on the recent round announcement."
        
        recent_news = company_data.get("recent_news", [])
        if recent_news and len(recent_news) > 0:
            launch_keywords = ["launch", "release", "announce", "milestone", "expansion"]
            for news_item in recent_news:
                if isinstance(news_item, str) and any(keyword in news_item.lower() for keyword in launch_keywords):
                    return "Congrats on the new launch — exciting milestone."
        
        founded_year = company_data.get("founded_year")
        if founded_year and isinstance(founded_year, int):
            current_year = datetime.now().year
            company_age = current_year - founded_year
            if company_age >= 5:
                return "You've built a strong reputation in the space over time."
        
        num_employees = company_data.get("num_employees")
        if num_employees and isinstance(num_employees, int) and num_employees >= 100:
            return "You've built a strong reputation in the space over time."
            
        return default_one_liner


async def get_company_enrichment(company_name: str, company_website: str = None) -> Dict[str, Any]:
    """
    Convenience function to get company enrichment data
    
    Args:
        company_name: Name of the company to look up
        company_website: Website URL for better context (optional)
        
    Returns:
        Dictionary with company data
    """
    client = SixtyFourAPI()
    return await client.get_company_data(company_name, company_website)


async def get_email_variables(company_name: str, company_website: str = None) -> Dict[str, Any]:
    """
    Get email template variables based on company data
    
    Args:
        company_name: Name of the company to look up
        company_website: Website URL for better context (optional)
        
    Returns:
        Dictionary with template variables
    """
    client = SixtyFourAPI()
    company_data = await client.get_company_data(company_name, company_website)
    
    company_vertical = client.extract_company_vertical(company_data)
    
    app_layer = client.is_app_layer_company(company_data)
    
    one_liner = client.determine_one_liner(company_data)
    
    portfolio_companies = get_portfolio_companies_by_vertical(company_vertical)
    
    return {
        "company_vertical": company_vertical,
        "app_layer": app_layer,
        "one_liner": one_liner,
        "portfolio_companies": portfolio_companies,
        "include_tldr": True,
        "tldr_block": ""
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

