"""
LinkedIn Research Module

This module provides functionality to generate fake but realistic LinkedIn 
profile data for personalized email generation. Uses industry-aware data 
generation based on company names and creates plausible professional profiles.

Features:
- Industry detection based on company keywords
- Realistic job titles, skills, and education by industry
- Fake but plausible work experience and background
- Random LinkedIn posts and activity
- High-quality fake data for AI prompt enhancement
"""

import asyncio
import random
from typing import Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum
import os

class ResearchStatus(Enum):
    """Status of LinkedIn research operation"""
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    BLOCKED = "blocked"
    INVALID_URL = "invalid_url"


@dataclass
class ProfileData:
    """Container for extracted LinkedIn profile data"""
    name: str
    current_title: Optional[str] = None
    current_company: Optional[str] = None
    location: Optional[str] = None
    industry: Optional[str] = None
    experience_years: Optional[int] = None
    education: Optional[str] = None
    skills: Optional[list[str]] = None
    recent_posts: Optional[list[str]] = None


@dataclass
class ResearchResult:
    """Container for research operation results"""
    status: ResearchStatus
    profile_data: Optional[ProfileData] = None
    error_message: Optional[str] = None
    research_time_seconds: float = 0.0
    data_quality_score: float = 0.0  # 0-1 score indicating data completeness


# Industry-aware fake data for profile generation
INDUSTRY_DATA = {
    'technology': {
        'keywords': ['tech', 'software', 'digital', 'ai', 'data', 'cloud', 'cyber', 'automation', 'platform', 'app'],
        'titles': ['Software Engineer', 'Senior Developer', 'Tech Lead', 'Engineering Manager', 'CTO', 'Product Manager', 'Data Scientist', 'DevOps Engineer', 'Full Stack Developer', 'Solutions Architect'],
        'skills': ['Python', 'JavaScript', 'React', 'AWS', 'Docker', 'Kubernetes', 'Machine Learning', 'SQL', 'Git', 'Node.js', 'TypeScript', 'MongoDB'],
        'degrees': ['Computer Science', 'Software Engineering', 'Information Technology', 'Computer Engineering', 'Data Science'],
        'universities': ['Stanford University', 'MIT', 'UC Berkeley', 'Carnegie Mellon', 'University of Washington', 'Georgia Tech', 'UT Austin']
    },
    'finance': {
        'keywords': ['bank', 'financial', 'capital', 'investment', 'fund', 'trading', 'wealth', 'credit', 'asset', 'insurance'],
        'titles': ['Financial Analyst', 'Investment Manager', 'Portfolio Manager', 'VP Finance', 'Risk Manager', 'Compliance Officer', 'Quantitative Analyst', 'Relationship Manager'],
        'skills': ['Financial Modeling', 'Risk Management', 'Bloomberg Terminal', 'Excel', 'SQL', 'Python', 'Tableau', 'Portfolio Management', 'Derivatives', 'Fixed Income'],
        'degrees': ['Finance', 'Economics', 'Business Administration', 'Mathematics', 'Accounting'],
        'universities': ['Wharton', 'Harvard Business School', 'Chicago Booth', 'NYU Stern', 'Columbia Business School', 'London Business School']
    },
    'healthcare': {
        'keywords': ['health', 'medical', 'pharma', 'biotech', 'clinical', 'hospital', 'wellness', 'therapy', 'diagnostic'],
        'titles': ['Clinical Research Manager', 'Medical Director', 'Healthcare Consultant', 'Biotech Researcher', 'Regulatory Affairs Manager', 'Medical Affairs Manager'],
        'skills': ['Clinical Research', 'GCP', 'FDA Regulations', 'Medical Writing', 'Biostatistics', 'Drug Development', 'Healthcare Analytics'],
        'degrees': ['Medicine', 'Biology', 'Biomedical Engineering', 'Public Health', 'Pharmacy', 'Nursing'],
        'universities': ['Johns Hopkins', 'Harvard Medical School', 'Mayo Clinic', 'UCSF', 'Duke University', 'Vanderbilt University']
    },
    'consulting': {
        'keywords': ['consulting', 'advisory', 'strategy', 'management', 'transformation', 'optimization'],
        'titles': ['Management Consultant', 'Strategy Consultant', 'Senior Consultant', 'Principal', 'Director', 'Partner', 'Business Analyst'],
        'skills': ['Strategy Development', 'Process Improvement', 'Change Management', 'Data Analysis', 'Project Management', 'Stakeholder Management'],
        'degrees': ['MBA', 'Business Administration', 'Economics', 'Engineering', 'Liberal Arts'],
        'universities': ['Harvard Business School', 'Stanford GSB', 'Wharton', 'INSEAD', 'Kellogg', 'MIT Sloan']
    },
    'marketing': {
        'keywords': ['marketing', 'advertising', 'brand', 'digital', 'social', 'campaign', 'creative', 'media'],
        'titles': ['Marketing Manager', 'Digital Marketing Specialist', 'Brand Manager', 'Marketing Director', 'Growth Manager', 'Content Manager'],
        'skills': ['Digital Marketing', 'Google Analytics', 'SEO/SEM', 'Social Media Marketing', 'Content Strategy', 'Brand Management', 'Marketing Automation'],
        'degrees': ['Marketing', 'Communications', 'Business', 'Psychology', 'Journalism'],
        'universities': ['Northwestern', 'University of Pennsylvania', 'NYU', 'USC', 'University of Michigan', 'Boston University']
    },
    'default': {
        'keywords': [],
        'titles': ['Manager', 'Senior Manager', 'Director', 'VP', 'Analyst', 'Specialist', 'Coordinator', 'Associate'],
        'skills': ['Leadership', 'Project Management', 'Communication', 'Analysis', 'Problem Solving', 'Team Management'],
        'degrees': ['Business Administration', 'Liberal Arts', 'Management', 'Economics'],
        'universities': ['State University', 'University of California', 'University of Texas', 'Florida State', 'Ohio State', 'Penn State']
    }
}

LOCATIONS = [
    'San Francisco, CA', 'New York, NY', 'Los Angeles, CA', 'Chicago, IL', 'Boston, MA',
    'Seattle, WA', 'Austin, TX', 'Denver, CO', 'Atlanta, GA', 'Miami, FL',
    'Washington, DC', 'Philadelphia, PA', 'Phoenix, AZ', 'San Diego, CA', 'Dallas, TX'
]


def detect_industry(company_name: str) -> str:
    """Detect industry based on company name keywords"""
    company_lower = company_name.lower()
    
    for industry, data in INDUSTRY_DATA.items():
        if industry == 'default':
            continue
        for keyword in data['keywords']:
            if keyword in company_lower:
                return industry
    
    return 'default'


def generate_fake_profile_data(name: str, company: str) -> ProfileData:
    """Generate realistic fake LinkedIn profile data based on name and company"""
    
    # Detect industry from company name
    industry = detect_industry(company)
    industry_data = INDUSTRY_DATA[industry]
    
    # Generate current title and experience
    current_title = random.choice(industry_data['titles'])
    experience_years = random.randint(3, 15)
    
    # Generate education
    degree = random.choice(industry_data['degrees'])
    university = random.choice(industry_data['universities'])
    education = f"{degree} from {university}"
    
    # Generate skills (4-8 relevant skills)
    num_skills = random.randint(4, 8)
    skills = random.sample(industry_data['skills'], min(num_skills, len(industry_data['skills'])))
    
    # Generate location
    location = random.choice(LOCATIONS)
    
    # Generate recent posts/activity (1-3 posts)
    recent_posts = generate_fake_posts(industry, current_title, company, random.randint(1, 3))
    
    return ProfileData(
        name=name,
        current_title=current_title,
        current_company=company,
        location=location,
        industry=industry.title(),
        experience_years=experience_years,
        education=education,
        skills=skills,
        recent_posts=recent_posts
    )


def generate_fake_posts(industry: str, title: str, company: str, num_posts: int) -> list[str]:
    """Generate fake but realistic LinkedIn posts"""
    
    post_templates = [
        f"Excited to share insights from our latest project at {company}. The team's dedication to innovation continues to inspire me.",
        f"Reflecting on the key trends shaping our industry. Looking forward to the opportunities ahead in {industry}.",
        f"Grateful for the opportunity to lead such an amazing team. Our recent achievements wouldn't be possible without their expertise.",
        f"Just wrapped up an insightful conference on {industry} trends. The future looks bright for our field!",
        f"Proud of what we've accomplished at {company} this quarter. Collaboration and innovation are truly our strengths.",
        f"Sharing some thoughts on the evolving landscape in {industry}. Exciting times ahead for professionals in our space.",
        f"Honored to be part of the {company} family. Every day brings new challenges and opportunities to grow."
    ]
    
    return random.sample(post_templates, min(num_posts, len(post_templates)))


class LinkedInResearcher:
    """
    Main class for LinkedIn profile research
    
    Phase 1: Interface definition and basic validation
    TODO: Implement research strategies in later phases
    """
    
    def __init__(self):
        self.timeout = int(os.getenv("LINKEDIN_SCRAPER_TIMEOUT", "10"))
        self.user_agent = os.getenv("LINKEDIN_USER_AGENT", "Mozilla/5.0 (compatible; EmailGenerator/1.0)")
        self.max_retries = int(os.getenv("AI_MAX_RETRIES", "2"))
    
    def validate_linkedin_url(self, url: str) -> bool:
        """
        Validate if the provided URL is a valid LinkedIn profile URL
        
        Args:
            url: LinkedIn profile URL to validate
            
        Returns:
            bool: True if URL appears to be a valid LinkedIn profile URL
        """
        if not url or not isinstance(url, str):
            return False
        
        # Basic LinkedIn URL validation
        linkedin_patterns = [
            "linkedin.com/in/",
            "www.linkedin.com/in/",
            "https://linkedin.com/in/",
            "https://www.linkedin.com/in/"
        ]
        
        url_lower = url.lower().strip()
        return any(pattern in url_lower for pattern in linkedin_patterns)
    
    async def research_profile(self, name: str, company: str, linkedin_url: str = None) -> ResearchResult:
        """
        Generate fake LinkedIn profile research based on name and company
        
        Args:
            name: Person's full name
            company: Company name 
            linkedin_url: LinkedIn profile URL (optional, used for validation only)
            
        Returns:
            ResearchResult: Container with fake research results and metadata
        """
        start_time = asyncio.get_event_loop().time()
        
        # Validate LinkedIn URL if provided
        if linkedin_url and not self.validate_linkedin_url(linkedin_url):
            return ResearchResult(
                status=ResearchStatus.INVALID_URL,
                error_message=f"Invalid LinkedIn URL format: {linkedin_url}",
                research_time_seconds=asyncio.get_event_loop().time() - start_time
            )
        
        try:
            # Add small random delay to simulate research time
            await asyncio.sleep(random.uniform(0.1, 0.5))
            
            # Generate fake profile data
            profile_data = generate_fake_profile_data(name, company)
            
            return ResearchResult(
                status=ResearchStatus.SUCCESS,
                profile_data=profile_data,
                research_time_seconds=asyncio.get_event_loop().time() - start_time,
                data_quality_score=random.uniform(0.7, 0.95)  # High quality fake data
            )
            
        except Exception as e:
            return ResearchResult(
                status=ResearchStatus.FAILED,
                error_message=f"Profile generation failed: {str(e)}",
                research_time_seconds=asyncio.get_event_loop().time() - start_time,
                data_quality_score=0.0
            )
    

# Module-level convenience functions
async def research_linkedin_profile(name: str, company: str, linkedin_url: str = None) -> ResearchResult:
    """
    Convenience function to generate fake LinkedIn profile research
    
    Args:
        name: Person's full name
        company: Company name
        linkedin_url: LinkedIn profile URL (optional)
        
    Returns:
        ResearchResult: Fake research results and metadata
    """
    researcher = LinkedInResearcher()
    return await researcher.research_profile(name, company, linkedin_url)


def is_valid_linkedin_url(url: str) -> bool:
    """
    Convenience function to validate LinkedIn URL
    
    Args:
        url: URL to validate
        
    Returns:
        bool: True if valid LinkedIn profile URL
    """
    researcher = LinkedInResearcher()
    return researcher.validate_linkedin_url(url)
