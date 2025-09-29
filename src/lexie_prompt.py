"""
Generic AI Email Generation Prompt

AI prompt instructions for generating any template content using templates from templates.py
and enriched company data from Parallel AI
"""

import os
from openai import AsyncOpenAI

try:
    from .templates import TemplateType, get_template_content
    from .sixtyfour_api import get_email_variables, get_portfolio_companies_by_vertical
    from .utils import get_random_agent_info
except ImportError:
    from templates import TemplateType, get_template_content
    from sixtyfour_api import get_email_variables, get_portfolio_companies_by_vertical
    from utils import get_random_agent_info


def format_portfolio_companies(portfolio_companies):
    """Ensure portfolio companies are returned as a clean, comma-separated string"""
    if isinstance(portfolio_companies, list):
        return ", ".join(portfolio_companies)
    if isinstance(portfolio_companies, dict):
        return ", ".join(portfolio_companies.values())
    return str(portfolio_companies)


def get_generic_prompt(company_name: str, recipient_name: str, template_type: TemplateType, company_data: dict = None):
    """Return the AI prompt that incorporates any template and company data"""
    template_content = get_template_content(template_type)

    # Extract company information from raw GPT response
    if company_data and "structured_data" in company_data:
        structured_data = company_data["structured_data"]
        company_vertical = structured_data.get("industry", "technology")
        company_description = structured_data.get("company_description", "")
        funding_rounds = structured_data.get("funding_rounds", "")
        recent_news = structured_data.get("recent_news", "")
        founded_year = structured_data.get("founded_year", "")
        findings = company_data.get("findings", [])
        references = company_data.get("references", {})
    else:
        company_vertical = "technology"
        company_description = ""
        funding_rounds = ""
        recent_news = ""
        founded_year = ""
        findings = []
        references = {}
    
    portfolio_companies = get_portfolio_companies_by_vertical(company_vertical)
    portfolio_companies_str = format_portfolio_companies(portfolio_companies)
    
    # Get agent information
    agent_info = get_random_agent_info()
    agent_name = agent_info["agent_name"]

    return f"""
You are an expert email writer specializing in personalized outreach emails. Use the following template and company information to create a highly personalized email:

TEMPLATE:
{template_content}

COMPANY INFORMATION:
- Company Name: {company_name}
- Industry/Vertical: {company_vertical}
- Company Description: {company_description}
- Founded Year: {founded_year}
- Funding Rounds: {funding_rounds}
- Recent News: {recent_news}
- Key Findings: {', '.join(findings) if findings else 'None'}
- Research Sources: {', '.join(references.values()) if references else 'None'}
- Relevant Portfolio Companies: {portfolio_companies_str}

Based on the recipient's information and company data, generate content for each template variable:
- {{{{name}}}}: Use '{recipient_name}'
- {{{{company}}}}: Use '{company_name}' 
- {{{{agent_name}}}}: Use '{agent_name}'
For templates with additional variables, use the enriched company data to personalize:
- Use the company description, funding info, and industry to make the email relevant
- Reference specific company achievements or recent developments when appropriate
- Match the tone and style of the template provided

Guidelines:
- Keep tone professional and appropriate for the template style
- Use the enriched company data to add personalization where possible
- Ensure all template variables are properly filled
- Make the email feel genuine and well-researched
- Follow the structure and style of the provided template

Generate the complete email by replacing all template variables with appropriate content.
"""


async def get_ai_email_response(company_name: str, recipient_name: str, template_type: TemplateType, company_website: str = None):
    """Get OpenAI response using any template with company enrichment"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    client = AsyncOpenAI(api_key=api_key)

    # Try to get raw enriched company data from Parallel AI
    company_data = None
    if company_website:
        try:
            try:
                from .gpt_enrichment import ParallelAIEnrichment
            except ImportError:
                from src.gpt_enrichment import ParallelAIEnrichment
            enrichment_client = ParallelAIEnrichment()
            company_data = await enrichment_client.get_company_data(company_name, company_website)
        except Exception as e:
            print(f"Error getting company enrichment: {e}")

    # Get the prompt
    prompt = get_generic_prompt(company_name, recipient_name, template_type, company_data)

    try:
        print(f"🔄 Making OpenAI API call for {recipient_name} at {company_name}")
        print(f"🔑 Using model: {os.getenv('OPENAI_MODEL', 'gpt-4o-mini')}")
        print(f"🔑 Max tokens: {os.getenv('OPENAI_MAX_TOKENS', '1000')}")
        print(f"🔑 Temperature: {os.getenv('OPENAI_TEMPERATURE', '0.4')}")
        
        response = await client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Generate a personalized outreach email for {recipient_name} at {company_name}. "
                        f"Use the template and company information provided. "
                        f"Replace any {{{{name}}}} placeholders with '{recipient_name}' in the final email.\n\n{prompt}"
                    ),
                }
            ],
            max_tokens=int(os.getenv("OPENAI_MAX_TOKENS", "1000")),
            temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.4")),  # tuned for consistency
        )

        print(f"✅ OpenAI API response received for {recipient_name}")
        
        if response.choices and len(response.choices) > 0:
            email_text = response.choices[0].message.content.strip()
            print(f"📧 Generated email content length: {len(email_text)} characters")
            print(f"📧 Email preview (first 100 chars): {email_text[:100]}...")
            
            # Safety net replacement in case model leaves {{name}} placeholders
            email_text = email_text.replace("{{name}}", recipient_name)
            print(f"✅ Email generation completed for {recipient_name} at {company_name}")
            return email_text
        else:
            print(f"❌ No response choices from OpenAI for {recipient_name}")
            raise ValueError("No response generated from OpenAI")

    except Exception as e:
        print(f"❌ OpenAI API error for {recipient_name} at {company_name}: {str(e)}")
        print(f"❌ Error type: {type(e).__name__}")
        raise Exception(f"OpenAI API error: {str(e)}")


# Backwards compatibility function for Lexie
async def get_lexie_response(company_name: str, recipient_name: str, company_website: str = None):
    """Backwards compatibility wrapper for Lexie template"""
    return await get_ai_email_response(company_name, recipient_name, TemplateType.LEXI, company_website)


