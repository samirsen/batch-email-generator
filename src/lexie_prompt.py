"""
Lexie AI Email Generation Prompt

AI prompt instructions for generating Lexie template content using the actual template from templates.py
and company data from sixtyfour_api.py
"""

import os
from openai import AsyncOpenAI

try:
    from .templates import TemplateType, get_template_content
    from .sixtyfour_api import get_email_variables, get_portfolio_companies_by_vertical
except ImportError:
    from templates import TemplateType, get_template_content
    from sixtyfour_api import get_email_variables, get_portfolio_companies_by_vertical

def get_lexie_prompt(company_name: str, company_data: dict = None):
    """Return the Lexie AI prompt that incorporates the actual template and company data"""
    template_content = get_template_content(TemplateType.LEXI)
    
    # Extract company information for personalization
    company_vertical = company_data.get("company_vertical", "technology") if company_data else "technology"
    portfolio_companies = get_portfolio_companies_by_vertical(company_vertical)
    
    return f"""
You are Lexi, an investor at Sound Ventures, Ashton Kutcher's $1.5bn+ AUM VC firm. Generate a personalized outreach email for startup founders using the following template:

TEMPLATE:
{template_content}

COMPANY INFORMATION:
- Company Name: {company_name}
- Industry/Vertical: {company_vertical}
- Relevant Portfolio Companies: {portfolio_companies}

Based on the recipient's information and company data, generate content for each template variable:
- name: {{{{name}}}}
- opening_line: Brief, personalized greeting referencing something specific about {company_name} or their background
- intro_line: Your role/focus at Sound Ventures (e.g., "I focus on {company_vertical} and emerging tech investments")
- portfolio: Use these relevant portfolio companies: {portfolio_companies}
- personalization_block: Specific reason why you're reaching out to this particular founder/company, mentioning their industry ({company_vertical})
- context_block: Brief context about Sound Ventures' investment focus, especially in {company_vertical} companies
- cta_block: Clear call-to-action for a call or meeting

Guidelines:
- Write as Lexi from Sound Ventures
- Keep tone professional but approachable
- Focus on {company_vertical} and emerging tech opportunities
- Demonstrate genuine interest in their specific company ({company_name})
- Establish credibility through Sound's track record in {company_vertical}
- Use the provided portfolio companies that are relevant to their vertical
- Personalize based on company industry and stage

Generate the complete email by filling in all template variables.
"""

async def get_lexie_response(company_name: str, recipient_name: str, company_website: str = None):
    """Get OpenAI response using the Lexie prompt with company enrichment"""
    # Initialize OpenAI client
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    client = AsyncOpenAI(api_key=api_key)
    
    # Try to get enriched company data
    company_data = None
    if company_website:
        try:
            company_data = await get_email_variables(company_name, company_website)
        except Exception as e:
            print(f"Error getting company enrichment: {e}")
    
    # Get the prompt
    prompt = get_lexie_prompt(company_name, company_data)
    
    try:
        # Make API call to OpenAI
        response = await client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[
                {
                    "role": "user",
                    "content": f"You are Lexi, an investor at Sound Ventures. Generate a personalized outreach email for {recipient_name} at {company_name}. Replace {{{{name}}}} with '{recipient_name}' in the final email.\n\n{prompt}"
                }
            ],
            max_tokens=int(os.getenv("OPENAI_MAX_TOKENS", "1000")),
            temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.7"))
        )
        
        if response.choices and len(response.choices) > 0:
            return response.choices[0].message.content.strip()
        else:
            raise ValueError("No response generated from OpenAI")
            
    except Exception as e:
        raise Exception(f"OpenAI API error: {str(e)}")

if __name__ == "__main__":
    import asyncio
    
    print('=== LEXIE TEMPLATE ===')
    print(get_template_content(TemplateType.LEXI))
    print()
    
    print('=== LEXIE PROMPT FOR NURTURE ===')
    print(get_lexie_prompt('Nurture'))
    print()
    
    print('=== OPENAI RESPONSE FOR NURTURE COMPANY (NISHA) ===')
    async def test_response():
        try:
            print("Starting OpenAI call...")
            # Test with Nurture company data and Nisha's name
            response = await get_lexie_response('Nurture', 'Nisha', 'https://www.linkedin.com/in/nishakochar/')
            print("OpenAI Response received:")
            print("-" * 50)
            print(response)
            print("-" * 50)
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
    
    asyncio.run(test_response())