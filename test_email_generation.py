"""
Test script for email generation with the new template format
"""

import asyncio
import pandas as pd
from dotenv import load_dotenv
from src.email_generator import generate_single_email
from src.templates import TemplateType

# Load environment variables
load_dotenv()

async def test_email_generation():
    """
    Test email generation with the new template format
    """
    # Create a test row using Nurture company data
    test_row = pd.Series({
        'name': 'Nisha',
        'company': 'Nurture',
        'linkedin_url': 'https://www.linkedin.com/in/nishakochar/',
        'intelligence': True,
        'template_type': 'lexie'
    })
    
    # Generate email
    email_content = await generate_single_email(test_row, TemplateType.LEXI)
    
    # Print the generated email
    print("\n" + "="*50)
    print("GENERATED EMAIL:")
    print("="*50)
    print(email_content)
    print("="*50 + "\n")
    
    return email_content

if __name__ == "__main__":
    # Run the test
    asyncio.run(test_email_generation())

