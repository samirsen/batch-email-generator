"""
Test script for the new generic email generation
"""

import asyncio
import os
from src.lexie_prompt import get_ai_email_response
from src.templates import TemplateType

async def test_generic_email():
    """Test the generic email generation with different templates"""
    
    # Test data
    company_name = "Nurture"
    recipient_name = "Alex"
    company_website = "https://nurture.is"
    
    # Test different templates
    templates_to_test = [
        TemplateType.LEXI,
        TemplateType.LUCAS,
        TemplateType.ZACH,
        TemplateType.SALES_OUTREACH,
        TemplateType.NETWORKING
    ]
    
    for template_type in templates_to_test:
        print(f"\n{'='*60}")
        print(f"TESTING TEMPLATE: {template_type}")
        print(f"{'='*60}")
        
        try:
            # Check if API key is set
            if not os.getenv("OPENAI_API_KEY"):
                print("❌ OPENAI_API_KEY not set - skipping AI generation")
                continue
                
            email_content = await get_ai_email_response(
                company_name=company_name,
                recipient_name=recipient_name,
                template_type=template_type,
                company_website=company_website
            )
            
            print(f"✅ Generated email for {template_type}:")
            print("-" * 40)
            print(email_content)
            
        except Exception as e:
            print(f"❌ Error with {template_type}: {str(e)}")

if __name__ == "__main__":
    asyncio.run(test_generic_email())