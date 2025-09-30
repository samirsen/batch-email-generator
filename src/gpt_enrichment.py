import os
import openai
import json
import aiohttp
import asyncio
from typing import Dict, Any, Optional
from datetime import datetime

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
PARALLEL_AI_API_KEY = os.getenv("PARALLEL_AI_API_KEY", "")





class ParallelAIEnrichment:
    """
    Company enrichment using Parallel AI for real-time web scraping
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Parallel AI enrichment client
        
        Args:
            api_key: Parallel AI API key (defaults to environment variable)
        """
        self.api_key = api_key or PARALLEL_AI_API_KEY
        self.base_url = "https://api.parallel.ai/v1beta/search"
    
    async def get_company_data(self, company_name: str, company_website: str = None, linkedin_url: str = None) -> Dict[str, Any]:
        """
        Get company enrichment data using Parallel AI
        
        Args:
            company_name: Name of the company to research
            company_website: Website URL for better context (optional)
            linkedin_url: LinkedIn URL for the company (optional)
            
        Returns:
            Dictionary with company data in SixtyFour API format
        """
        try:
            if not self.api_key:
                print("ERROR: No Parallel AI API key found!")
                return {"error": "No API key configured", "company_name": company_name}
            
            # Build the objective for Parallel AI
            objective = self._build_objective(company_name, company_website, linkedin_url)
            
            print(f"Researching {company_name} using Parallel AI...")
            
            # Call Parallel AI API
            response = await self._call_parallel_ai_api(objective)
            
            if response:
                return await self._format_parallel_response(response, company_name)
            else:
                return self._fallback_response(company_name)
                
        except Exception as e:
            print(f"Error calling Parallel AI API: {str(e)}")
            return self._fallback_response(company_name, str(e))
    
    def _build_objective(self, company_name: str, company_website: str = None, linkedin_url: str = None) -> str:
        """Build the objective string for Parallel AI to match expected format"""
        
        # Build sources list similar to your example
        sources = []
        if company_website:
            sources.append(company_website)
        if linkedin_url:
            sources.append(linkedin_url)
        
        # Format sources like "scrape the linkedin and website"
        sources_text = ""
        if sources:
            if len(sources) == 1:
                sources_text = f"scrpe the {sources[0]} and "
            else:
                sources_text = f"scrpe the linedin and website {','.join(sources)} and "
        
        # Match your exact objective format from the example
        objective = f'"Provide company profile for {company_name} including company_name, industry, vertical, company_description, founded_year, recent_news, funding_rounds"\nscrpe the linedin adn website {company_name},{company_website or ""},{linkedin_url or ""}'
        
        return objective
    
    async def _call_parallel_ai_api(self, objective: str) -> Optional[Dict[str, Any]]:
        """Call Parallel AI API with the research objective"""
        try:
            headers = {
                "Content-Type": "application/json",
                "x-api-key": self.api_key
            }
            
            payload = {
                "search_queries": [],
                "processor": "base", 
                "objective": objective
            }
            
            print(f"Parallel AI payload: {json.dumps(payload, indent=2)}")
            
            timeout = aiohttp.ClientTimeout(total=120)
            
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    self.base_url,
                    headers=headers,
                    json=payload
                ) as response:
                    print(f"Parallel AI response status: {response.status}")
                    if response.status == 200:
                        data = await response.json()
                        print("Parallel AI response:", str(data)[:100] + ("..." if len(str(data)) > 100 else ""))
                        return data
                    else:
                        error_text = await response.text()
                        print(f"Parallel AI error ({response.status}): {error_text}")
                        return None
                        
        except Exception as e:
            print(f"Parallel AI API error: {e}")
            return None
    
    async def _format_parallel_response(self, parallel_response: Dict[str, Any], company_name: str) -> Dict[str, Any]:
        """Format Parallel AI response to match SixtyFour API structure"""
        
        # Extract the raw content from Parallel AI response
        raw_content = self._extract_raw_content(parallel_response)
        
        # Use GPT to structure the raw content
        structured_response = await self._process_with_gpt(raw_content, company_name)
        
        if structured_response:
            return structured_response
        else:
            # Fallback to basic extraction if GPT fails
            structured_data = self._extract_structured_data(raw_content, company_name)
            return {
                "notes": raw_content[:1000] + "..." if len(raw_content) > 1000 else raw_content,
                "structured_data": structured_data,
                "findings": parallel_response.get("findings", []),
                "references": parallel_response.get("references", {}),
                "confidence_score": 6.0  # Lower confidence for basic extraction
            }
    
    def _extract_raw_content(self, parallel_response: Dict[str, Any]) -> str:
        """Extract raw content from Parallel AI response"""
        content_parts = []
        
        # Handle different response structures
        if "results" in parallel_response:
            for result in parallel_response["results"]:
                if "excerpts" in result:
                    content_parts.extend(result["excerpts"])
                if "title" in result and result["title"]:
                    content_parts.append(f"Title: {result['title']}")
                if "url" in result:
                    content_parts.append(f"Source: {result['url']}")
        
        if "result" in parallel_response:
            content_parts.append(str(parallel_response["result"]))
        elif "content" in parallel_response:
            content_parts.append(str(parallel_response["content"]))
        elif "response" in parallel_response:
            content_parts.append(str(parallel_response["response"]))
        
        # If nothing found, stringify the whole response
        if not content_parts:
            content_parts.append(str(parallel_response))
        
        return "\n\n".join(content_parts)
    
    async def _process_with_gpt(self, raw_content: str, company_name: str) -> Optional[Dict[str, Any]]:
        """Process Parallel AI raw content with GPT to create structured response"""
        try:
            if not OPENAI_API_KEY:
                print("No OpenAI key available for GPT post-processing")
                return None
            
            # Truncate content if too long to avoid token limits
            if len(raw_content) > 8000:
                raw_content = raw_content[:8000] + "..."
            
            prompt = f"""
You are a business researcher. I have raw web scraping data about a company "{company_name}".
Please analyze this data and extract structured company information in the exact JSON format below.

Raw scraped data:
{raw_content}

Please return a JSON object with this exact structure:

{{
  "notes": "Comprehensive analysis of the company based on the scraped data. Include business model, key products/services, market position, and any notable information found.",
  "structured_data": {{
    "industry": "Primary industry sector",
    "vertical": "Specific business vertical or category",
    "company_description": "2-3 sentence description of what the company does",
    "founded_year": 2020,
    "recent_news": "Recent developments, launches, or news found in the data",
    "funding_rounds": "Funding information with year if found (e.g., 'Latest Deal Type: Series A, Latest Deal Amount: $5M, Year: 2024'), empty string otherwise",
    "linkedin_url": "LinkedIn URL if found in the data, empty string otherwise",
    "company_name": "{company_name}",
    "website": "Company website URL if found, empty string otherwise"
  }},
  "findings": [],
  "references": {{}},
  "confidence_score": 8.0
}}

Guidelines:
- Extract factual information only from the provided data
- If information is not available in the scraped data, use empty string or null
- For founded_year, use integer or null if not found
- Set confidence_score based on data quality (1-10 scale)
- Be professional and focus on business-relevant information
- ALWAYS populate references object with actual source URLs found in the scraped data
- If no specific sources available, use empty object {{}}
"""

            # Make async call to GPT
            import openai
            
            print(f"🔄 Attempting GPT post-processing for {company_name}...")
            
            client = openai.AsyncOpenAI(
                api_key=OPENAI_API_KEY,
                timeout=30.0
            )
            
            response = await client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                messages=[
                    {
                        "role": "system",
                        "content": "You are a professional business data analyst. Extract structured company information from raw web scraping data."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=1500
            )
            
            raw_content = response.choices[0].message.content.strip()
            
            print(f"📥 RAW GPT RESPONSE for {company_name}:")
            print("=" * 50)
            print(raw_content)
            print("=" * 50)
            print()
            
            # Extract JSON from response
            content = raw_content
            if content.startswith("```json"):
                content = content[7:-3].strip()
            elif content.startswith("```"):
                content = content[3:-3].strip()
            
            structured_response = json.loads(content)
            print(f"✅ GPT successfully structured Parallel AI data for {company_name}")
            
            return structured_response
            
        except json.JSONDecodeError as e:
            print(f"Failed to parse GPT JSON response: {e}")
            return None
            
        except Exception as e:
            print(f"Error in GPT post-processing: {e}")
            return None
    
    def _extract_structured_data(self, content: str, company_name: str) -> Dict[str, Any]:
        """Extract structured data from Parallel AI content"""
        
        # This is a simple extraction - could be enhanced with better parsing
        content_lower = content.lower()
        
        # Try to extract industry/vertical
        industry = "Technology"
        vertical = "Software"
        
        if "fintech" in content_lower or "financial" in content_lower:
            industry = "Financial Services"
            vertical = "Fintech"
        elif "healthcare" in content_lower or "medical" in content_lower:
            industry = "Healthcare"
            vertical = "Healthcare Technology"
        elif "ai" in content_lower or "artificial intelligence" in content_lower:
            industry = "Technology"
            vertical = "Artificial Intelligence"
        elif "saas" in content_lower or "software" in content_lower:
            industry = "Technology" 
            vertical = "Software as a Service"
        
        # Try to extract founded year
        founded_year = None
        import re
        year_match = re.search(r"founded in (\d{4})|established (\d{4})|since (\d{4})", content_lower)
        if year_match:
            founded_year = int(year_match.group(1) or year_match.group(2) or year_match.group(3))
        
        # Extract company description (first 2-3 sentences)
        sentences = content.split('. ')
        description = '. '.join(sentences[:2]) + '.' if sentences else f"{company_name} is a technology company."
        
        return {
            "industry": industry,
            "vertical": vertical,
            "company_description": description[:500],  # Limit length
            "founded_year": founded_year,
            "recent_news": "Recent company activity and developments from web research.",
            "funding_rounds": "",  # Would need specific extraction logic
            "linkedin_url": "",
            "company_name": company_name,
            "website": ""
        }
    
    def _fallback_response(self, company_name: str, error_msg: str = None) -> Dict[str, Any]:
        """Generate fallback response when API fails"""
        return {
            "notes": f"Unable to research {company_name} using Parallel AI.",
            "structured_data": {
                "industry": "Technology",
                "vertical": "Software",
                "company_description": f"{company_name} is a technology company.",
                "founded_year": None,
                "recent_news": "No recent news available.",
                "funding_rounds": "",
                "linkedin_url": "",
                "company_name": company_name,
                "website": ""
            },
            "findings": [],
            "references": {},
            "confidence_score": 1.0,
            "error": error_msg or "Parallel AI call failed"
        }


# Convenience functions to match existing SixtyFour API interface


# Parallel AI convenience functions
async def get_company_enrichment_parallel(company_name: str, company_website: str = None, linkedin_url: str = None) -> Dict[str, Any]:
    """
    Get company enrichment data using Parallel AI (alternative to SixtyFour API)
    
    Args:
        company_name: Name of the company to research
        company_website: Website URL for better context (optional)
        linkedin_url: LinkedIn URL for the company (optional)
        
    Returns:
        Dictionary with company data in SixtyFour format
    """
    client = ParallelAIEnrichment()
    return await client.get_company_data(company_name, company_website, linkedin_url)


async def get_email_variables_parallel(company_name: str, company_website: str = None, linkedin_url: str = None) -> Dict[str, Any]:
    """
    Get email template variables using Parallel AI-based company research
    
    Args:
        company_name: Name of the company to research
        company_website: Website URL for better context (optional)
        linkedin_url: LinkedIn URL for the company (optional)
        
    Returns:
        Dictionary with template variables
    """
    from sixtyfour_api import SixtyFourAPI, get_portfolio_companies_by_vertical
    
    # Get company data using Parallel AI
    enrichment_client = ParallelAIEnrichment()
    full_response = await enrichment_client.get_company_data(company_name, company_website, linkedin_url)
    
    # Extract structured_data (this matches what SixtyFour API returns)
    company_data = full_response.get("structured_data", {})
    
    # Use existing SixtyFour logic for processing the data
    sixtyfour_client = SixtyFourAPI()
    company_vertical = sixtyfour_client.extract_company_vertical(company_data)
    app_layer = sixtyfour_client.is_app_layer_company(company_data)
    one_liner = sixtyfour_client.determine_one_liner(company_data)
    portfolio_companies = get_portfolio_companies_by_vertical(company_vertical)
    
    return {
        "company_vertical": company_vertical,
        "app_layer": app_layer,
        "one_liner": one_liner,
        "portfolio_companies": portfolio_companies,
        "include_tldr": True,
        "tldr_block": ""
    }


