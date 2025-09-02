"""
AI Email Generator Module

This module provides OpenAI-powered intelligent email generation based on 
fake LinkedIn research data and user information.

Features:
- Template-type aware email generation
- Personalized prompts using fake research data
- OpenAI GPT integration with error handling
- Simplified retry logic and token management
- Industry-specific email personalization
"""

import asyncio
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import json

# OpenAI import
from openai import AsyncOpenAI

from .linkedin_research import ResearchResult, ProfileData, ResearchStatus
from .templates import TemplateType, EMAIL_TEMPLATES
from .utils import get_random_agent_info


class GenerationStatus(Enum):
    """Status of AI email generation operation"""
    SUCCESS = "success"
    FAILED = "failed"
    API_ERROR = "api_error"
    TIMEOUT = "timeout"
    QUOTA_EXCEEDED = "quota_exceeded"
    INVALID_INPUT = "invalid_input"


@dataclass
class GenerationResult:
    """Container for AI email generation results"""
    status: GenerationStatus
    email_content: Optional[str] = None
    error_message: Optional[str] = None
    tokens_used: int = 0
    generation_time_seconds: float = 0.0
    model_used: Optional[str] = None
    fallback_used: bool = False


class AIEmailGenerator:
    """
    Main class for AI-powered email generation
    """
    
    def __init__(self):
        # OpenAI Configuration
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.max_tokens = int(os.getenv("OPENAI_MAX_TOKENS", "1000"))
        self.temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.7"))
        
        # Processing Configuration
        self.max_tokens_per_email = int(os.getenv("MAX_TOKENS_PER_EMAIL", "500"))
        self.max_retries = int(os.getenv("AI_MAX_RETRIES", "2"))
        
        # Initialize OpenAI client
        self._client = None
        if self.api_key:
            self._client = AsyncOpenAI(api_key=self.api_key)
    
    def validate_configuration(self) -> tuple[bool, Optional[str]]:
        """
        Validate AI generator configuration
        
        Returns:
            tuple: (is_valid, error_message)
        """
        if not self.api_key:
            return False, "OpenAI API key not configured (OPENAI_API_KEY environment variable)"
        
        if not self._client:
            return False, "OpenAI client not initialized"
        
        if self.max_tokens <= 0:
            return False, f"Invalid max_tokens configuration: {self.max_tokens}"
        
        if not (0.0 <= self.temperature <= 2.0):
            return False, f"Invalid temperature configuration: {self.temperature}"
        
        return True, None
    
    def build_prompt(self, research_data: ResearchResult, user_info: Dict[str, Any], template_type: str = None) -> str:
        """
        Build OpenAI prompt based on research data, user information, and template type
        
        Args:
            research_data: LinkedIn research results
            user_info: User information from CSV (name, company, linkedin_url)
            template_type: Email template type (sales_outreach, recruitment, etc.)
            
        Returns:
            str: Formatted prompt for OpenAI API
        """
        # Extract user information (recipient)
        name = user_info.get('name', 'Unknown')
        company = user_info.get('company', 'Unknown Company')
        template_type = template_type or 'sales_outreach'
        
        # Get random agent information (sender)
        agent_info = get_random_agent_info()
        agent_name = agent_info['agent_name']
        agent_company = agent_info['company_name']
        
        # Build comprehensive research context
        research_context = ""
        if research_data.status.value == "success" and research_data.profile_data:
            profile = research_data.profile_data
            research_context = f"""
Professional Background:
- Name: {profile.name}
- Current Role: {profile.current_title} at {profile.current_company}
- Location: {profile.location}
- Industry: {profile.industry}
- Experience: {profile.experience_years} years in the field
- Education: {profile.education}
- Key Skills: {', '.join(profile.skills) if profile.skills else 'Not specified'}
"""
            
            # Add recent activity if available
            if profile.recent_posts:
                research_context += f"\nRecent Professional Activity:\n"
                for i, post in enumerate(profile.recent_posts[:2], 1):  # Limit to 2 posts
                    research_context += f"- {post}\n"
        else:
            research_context = f"Limited background information available (Research status: {research_data.status.value})"
        
        # Get template-specific instructions from templates.py
        try:
            template_enum = TemplateType(template_type)
            template_info = EMAIL_TEMPLATES.get(template_enum, EMAIL_TEMPLATES[TemplateType.SALES_OUTREACH])
            instruction = f"Write a {template_info['name'].lower()} email. {template_info['use_case']}. Context: {template_info['description']}"
        except (ValueError, KeyError):
            # Fallback to sales outreach if template type is invalid
            template_info = EMAIL_TEMPLATES[TemplateType.SALES_OUTREACH]
            instruction = f"Write a {template_info['name'].lower()} email. {template_info['use_case']}. Context: {template_info['description']}"
        
        # Build the complete prompt
        prompt = f"""
You are an expert email writer specializing in personalized business outreach emails.

SENDER INFORMATION (YOU):
- Your Name: {agent_name}
- Your Company: {agent_company}

TARGET PERSON PROFILE:
{research_context}

EMAIL TYPE: {template_type.replace('_', ' ').title()}

SPECIFIC INSTRUCTIONS: {instruction}

WRITING GUIDELINES:
1. Use specific details from their professional background to personalize the email
2. Reference their experience level, industry, or skills when relevant
3. Show genuine knowledge of their company and role
4. Maintain a professional but warm tone
5. Include a clear, specific call to action
6. Keep it concise (2-3 paragraphs maximum)
7. Write in first person as if you're reaching out directly
8. IMPORTANT: Use your actual name ({agent_name}) and company ({agent_company}) - DO NOT use placeholders like [Your Name] or [Your Company]
9. Sign the email with your name: {agent_name}

EMAIL FORMAT: Return only the email body content (no subject line, no signature block).

Write the personalized {template_type.replace('_', ' ')} email from {agent_name} at {agent_company}:
"""
        
        return prompt.strip()
    
    def estimate_tokens(self, prompt: str) -> int:
        """
        Estimate token count for a given prompt
        
        Args:
            prompt: Text prompt to estimate
            
        Returns:
            int: Estimated token count (rough approximation)
        """
        # Rough estimation: ~4 characters per token for English text
        return len(prompt) // 4 + 50  # Add buffer for response
    

    
    async def generate_intelligent_email(
        self, 
        research_data: ResearchResult, 
        user_info: Dict[str, Any],
        template_type: str = None
    ) -> GenerationResult:
        """
        Generate intelligent email using OpenAI based on research data
        
        Args:
            research_data: LinkedIn research results
            user_info: User information from CSV
            template_type: Email template type (sales_outreach, recruitment, etc.)
            
        Returns:
            GenerationResult: Generated email and metadata
        """
        start_time = asyncio.get_event_loop().time()
        
        # Validate configuration
        is_valid, error_msg = self.validate_configuration()
        if not is_valid:
            return GenerationResult(
                status=GenerationStatus.FAILED,
                error_message=f"Configuration error: {error_msg}",
                generation_time_seconds=asyncio.get_event_loop().time() - start_time
            )
        
        try:
            # Build prompt with template type
            prompt = self.build_prompt(research_data, user_info, template_type)
            estimated_tokens = self.estimate_tokens(prompt)
            
            # Check token limits
            if estimated_tokens > self.max_tokens_per_email:
                return GenerationResult(
                    status=GenerationStatus.FAILED,
                    error_message=f"Prompt too long: {estimated_tokens} tokens (max: {self.max_tokens_per_email})",
                    generation_time_seconds=asyncio.get_event_loop().time() - start_time
                )
            
            # Call OpenAI API with simplified retry logic
            for attempt in range(self.max_retries + 1):
                try:
                    if attempt > 0:
                        await asyncio.sleep(1.0)  # Simple 1-second delay between retries
                    
                    email_content, tokens_used = await self._call_openai_api(prompt)
                    
                    return GenerationResult(
                        status=GenerationStatus.SUCCESS,
                        email_content=email_content,
                        tokens_used=tokens_used,
                        generation_time_seconds=asyncio.get_event_loop().time() - start_time,
                        model_used=self.model,
                        fallback_used=False
                    )
                    
                except Exception as api_error:
                    if attempt == self.max_retries:
                        # Final attempt failed
                        if "rate limit" in str(api_error).lower():
                            status = GenerationStatus.QUOTA_EXCEEDED
                        else:
                            status = GenerationStatus.API_ERROR
                        
                        return GenerationResult(
                            status=status,
                            error_message=f"API error after {self.max_retries + 1} attempts: {str(api_error)}",
                            generation_time_seconds=asyncio.get_event_loop().time() - start_time
                        )
                    # Continue to next retry
            
        except Exception as e:
            return GenerationResult(
                status=GenerationStatus.FAILED,
                error_message=f"Unexpected error: {str(e)}",
                generation_time_seconds=asyncio.get_event_loop().time() - start_time
            )
    
    async def _call_openai_api(self, prompt: str) -> tuple[str, int]:
        """
        Make API call to OpenAI
        
        Args:
            prompt: Formatted prompt for the API
            
        Returns:
            tuple: (response_text, tokens_used)
        """
        if not self._client:
            raise ValueError("OpenAI client not initialized")
        
        try:
            # Make the API call
            response = await self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are an expert email writer specializing in personalized outreach emails. Write professional, engaging emails that feel personal and authentic."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                max_tokens=self.max_tokens_per_email,
                temperature=self.temperature,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )
            
            # Extract response and token usage
            if response.choices and len(response.choices) > 0:
                email_content = response.choices[0].message.content
                tokens_used = response.usage.total_tokens if response.usage else self.estimate_tokens(prompt)
                
                return email_content.strip(), tokens_used
            else:
                raise ValueError("No response generated from OpenAI")
                
        except Exception as e:
            # Re-raise with more context
            raise Exception(f"OpenAI API error: {str(e)}")


# Module-level convenience functions
async def generate_ai_email(research_data: ResearchResult, user_info: Dict[str, Any], template_type: str = None) -> GenerationResult:
    """
    Convenience function to generate AI email
    
    Args:
        research_data: LinkedIn research results
        user_info: User information from CSV
        template_type: Email template type (sales_outreach, recruitment, etc.)
        
    Returns:
        GenerationResult: Generated email and metadata
    """
    generator = AIEmailGenerator()
    return await generator.generate_intelligent_email(research_data, user_info, template_type)


async def generate_ai_email_from_template(user_info: Dict[str, Any], template_type: str = None) -> GenerationResult:
    """
    Generate LLM-based email using template as base (without LinkedIn research)
    
    Args:
        user_info: User information from CSV (name, company, linkedin_url)
        template_type: Email template type (sales_outreach, recruitment, etc.)
        
    Returns:
        GenerationResult: Generated email and metadata
    """
    generator = AIEmailGenerator()
    
    # Create a minimal research result (no LinkedIn research for template-based)
    minimal_research = ResearchResult(
        status=ResearchStatus.SUCCESS,
        profile_data=ProfileData(
            name=user_info.get('name', ''),
            current_company=user_info.get('company', ''),
            current_title="Professional",  # Generic title
            location="Unknown",
            industry="Business",
            experience_years=5,
            education="Professional Education",
            skills=[],
            recent_posts=[]
        ),
        research_time_seconds=0.0,  # Fixed: use correct parameter name
        error_message=None,
        data_quality_score=0.5  # Added missing parameter
    )
    
    return await generator.generate_intelligent_email(minimal_research, user_info, template_type)


def validate_ai_configuration() -> tuple[bool, Optional[str]]:
    """
    Convenience function to validate AI configuration
    
    Returns:
        tuple: (is_valid, error_message)
    """
    generator = AIEmailGenerator()
    return generator.validate_configuration()
