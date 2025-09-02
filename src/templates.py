"""
Email template management system for the Batch Email Generator.

This module provides a collection of professional email templates for different
use cases like sales outreach, recruitment, networking, etc.
"""

from enum import Enum
from typing import Dict, Any, Optional


class TemplateType(str, Enum):
    """Available email template types"""
    SALES_OUTREACH = "sales_outreach"
    RECRUITMENT = "recruitment"
    NETWORKING = "networking"
    PARTNERSHIP = "partnership"
    FOLLOW_UP = "follow_up"
    INTRODUCTION = "introduction"
    COLD_EMAIL = "cold_email"
    LUCAS="lucas"
    LEXI="lexie"
    ZACH="zach"


# Default template type
DEFAULT_TEMPLATE_TYPE = TemplateType.ZACH

# Template repository
EMAIL_TEMPLATES = {
    TemplateType.SALES_OUTREACH: {
        "name": "Sales Outreach",
        "description": "Professional sales outreach template for B2B lead generation",
        "use_case": "Initial contact with potential clients to offer services/products",
        "content": """Subject: Quick question about {{company}}'s growth strategy

Hi {{name}},

I've been following {{company}}'s impressive growth and noticed your role there. Your background caught my attention, especially your experience in scaling operations.

I work with companies similar to {{company}} to help them streamline their processes and reduce operational costs by 20-30%. Given your position, I thought you might be interested in a brief conversation about how we've helped similar organizations achieve significant efficiency gains.

Would you be open to a 15-minute call this week? I'd love to share some specific examples relevant to {{company}}'s industry.


Best regards,
{{agent_name}}

P.S. If this isn't a priority right now, I completely understand. Feel free to keep my contact for future reference."""
    },
    
    TemplateType.RECRUITMENT: {
        "name": "Recruitment Outreach",
        "description": "Professional template for reaching out to potential candidates",
        "use_case": "Recruiting talented professionals for open positions",
        "content": """Subject: Exciting opportunity at {{company}} - Would love to connect

Hi {{name}},

I came across your profile and was impressed by your background, particularly your experience at {{company}}. Your skill set aligns perfectly with some exciting opportunities we have.

We're currently looking for talented professionals to join our growing team, and I believe you'd be a great fit for several roles we have open. I'd love to discuss how your experience could contribute to our mission.

Would you be interested in a brief conversation about potential opportunities? I can share more details about the roles and how they might align with your career goals.


Best regards,
{{agent_name}}
Talent Acquisition Team

P.S. Even if you're not actively looking, I'd be happy to keep you in mind for future opportunities."""
    },
    
    TemplateType.NETWORKING: {
        "name": "Professional Networking",
        "description": "Template for building professional relationships and connections",
        "use_case": "Connecting with industry professionals for mutual benefit",
        "content": """Subject: Connecting with a fellow professional from {{company}}

Hi {{name}},

I hope this message finds you well. I came across your profile and was impressed by your work at {{company}}, particularly in your current role.

I'm always interested in connecting with talented professionals in our industry. Your experience and insights would be valuable, and I believe there could be opportunities for mutual collaboration or knowledge sharing.

Would you be open to connecting? I'd love to learn more about your work and share some insights from my own experience.


Best regards,
{{agent_name}}

P.S. No agenda here - just genuinely interested in expanding my professional network with quality connections."""
    },
    
    TemplateType.PARTNERSHIP: {
        "name": "Partnership Proposal",
        "description": "Template for proposing business partnerships and collaborations",
        "use_case": "Reaching out to potential business partners or collaborators",
        "content": """Subject: Partnership opportunity between our companies

Hi {{name}},

I've been researching companies in our space and {{company}} consistently comes up as an industry leader. Your innovative approach to business development particularly caught my attention.

I believe there might be some interesting synergies between our organizations. We've been exploring strategic partnerships that could benefit both parties, and I think {{company}} could be an ideal fit.

I'd love to explore potential collaboration opportunities. Would you be interested in a brief conversation to discuss how our companies might work together?


Best regards,
{{agent_name}}
Business Development

P.S. I'm happy to share some specific ideas I have in mind during our conversation."""
    },
    
    TemplateType.FOLLOW_UP: {
        "name": "Follow-up Email",
        "description": "Template for following up on previous conversations or meetings",
        "use_case": "Following up after initial contact, meetings, or events",
        "content": """Subject: Following up on our conversation about {{company}}

Hi {{name}},

I wanted to follow up on our previous conversation about {{company}} and the opportunities we discussed.

I've been thinking about the points you raised, and I believe we can address the specific challenges you mentioned. I'd like to share some additional insights that might be relevant to your situation.

Would you have time for a brief follow-up call this week? I can provide more concrete examples of how we've helped companies similar to {{company}}.


Best regards,
{{agent_name}}

P.S. Thank you for your time during our last conversation - I found your insights very valuable."""
    },
    
    TemplateType.INTRODUCTION: {
        "name": "Introduction Email",
        "description": "Template for introducing yourself and your services",
        "use_case": "Initial introduction to new contacts or warm leads",
        "content": """Subject: Introduction - Helping {{company}} achieve its goals

Hi {{name}},

I hope this email finds you well. My name is {{agent_name}}, and I wanted to introduce myself and my work.

I help companies like {{company}} optimize their operations and achieve sustainable growth. Based on what I've seen about {{company}}'s trajectory, I believe there might be some interesting opportunities to collaborate.

I'd love to learn more about your current initiatives and see if there's a way I can contribute to {{company}}'s continued success.

Would you be open to a brief introductory conversation? I promise to keep it concise and valuable.


Best regards,
{{agent_name}}

P.S. I'm genuinely interested in learning about your business, not just pitching services."""
    },
    
    TemplateType.COLD_EMAIL: {
        "name": "Cold Email Outreach",
        "description": "Generic cold email template for initial contact",
        "use_case": "First-time contact with prospects who don't know you",
        "content": """Subject: Quick question about {{company}}'s growth strategy

Hi {{name}},

I hope you're having a great week. I've been researching companies in your industry and {{company}} stands out for your innovative market positioning.

I'm reaching out because I help organizations like {{company}} with operational optimization and growth acceleration. I've noticed that many companies in your space are dealing with scaling challenges, and I'm curious if this is something {{company}} is experiencing as well.

If so, I'd love to share a quick insight that has helped similar companies. Would you be open to a brief 10-minute conversation?

No worries if the timing isn't right - I understand how busy things can get.

Best regards,
{{agent_name}}
"""
    },

    TemplateType.ZACH: {
        "name": "Zach Template",
        "description": "Zach template for initial contact",
        "use_case": "Zach template for initial contact",
        "content": """Subject: Reaching out from Ashton Kutcher's Sound Ventures

{{ name }},

Hope you're doing well. {{ intro_line }}. {{ personalization_block }}

{{ optional_context_block }}

Even if you're not raising right now, would love to find a way to be helpful. {{ cta_line }}

- Zach
"""
    },

    TemplateType.LEXI: {
        "name": "Lexie Template",
        "description": "Lexie template for initial contact",
        "use_case": "Lexie template for initial contact",
        "content": """Subject: Reaching out from Ashton Kutcher's Sound Ventures

Hi {{ name }},

{{ opening_line }}

{{ intro_line }} We've been early investors in {{ portfolio }}.

{{ personalization_block }}

A bit on Sound for context - {{ context_block }}

{{ cta_block }}

Looking forward to it,
Lexi
"""
    },

    TemplateType.LUCAS: {
        "name": "Lucas Template",
        "description": "Lucas template for initial contact",
        "use_case": "Lucas template for initial contact",
        "content": """Subject: Reaching out from Ashton Kutcher's Sound Ventures

{{ name }},

{% if app_layer -%}
I lead app-layer investing at Sound Ventures, a $1.5bn+ AUM VC firm led by Ashton Kutcher. 
We've been early investors in generational companies like {{ portfolio }}.
{%- else -%}
I’m an investor at Sound Ventures, a $1.5bn+ AUM VC firm led by Ashton Kutcher. 
We've been early investors in generational companies like {{ portfolio }}.
{%- endif %}

AI x {{ company_vertical }} is our #1 focus currently and we’ve heard exceptionally positive things about you/{{ company_name }}. {{ one_liner }}

{% if include_tldr %}
{{ tldr_block }}
{% endif %}

I’d love to hop on a call to learn about where you’re headed and introduce you to Sound / our less traditional value-add{{ " in this space" if add_space_clause else "" }}. 
Let me know if you’d be open to it and if so when works for you!

Thanks,
Lucas
"""
    }
}


def get_template_content(template_type: Optional[TemplateType] = None) -> str:
    """Get the content of a specific template type"""
    if template_type is None:
        template_type = DEFAULT_TEMPLATE_TYPE
    
    if template_type not in EMAIL_TEMPLATES:
        raise ValueError(f"Template type '{template_type}' not found. Available types: {list(EMAIL_TEMPLATES.keys())}")
    
    return EMAIL_TEMPLATES[template_type]["content"]


def get_template_info(template_type: Optional[TemplateType] = None) -> Dict[str, Any]:
    """Get information about a specific template"""
    if template_type is None:
        template_type = DEFAULT_TEMPLATE_TYPE
    
    if template_type not in EMAIL_TEMPLATES:
        raise ValueError(f"Template type '{template_type}' not found. Available types: {list(EMAIL_TEMPLATES.keys())}")
    
    template_data = EMAIL_TEMPLATES[template_type].copy()
    return {
        "template_type": template_type.value,
        "name": template_data["name"],
        "description": template_data["description"],
        "use_case": template_data["use_case"],
        "variables": ["name", "company", "linkedin_url"],
        "content_preview": template_data["content"][:200] + "..." if len(template_data["content"]) > 200 else template_data["content"]
    }


def get_all_templates() -> Dict[str, Dict[str, Any]]:
    """Get information about all available templates"""
    return {
        template_type.value: {
            "name": template_data["name"],
            "description": template_data["description"],
            "use_case": template_data["use_case"]
        }
        for template_type, template_data in EMAIL_TEMPLATES.items()
    }


def validate_template_variables(template_content: str) -> bool:
    """Validate that a template contains the required variables"""
    required_vars = ["{{name}}", "{{company}}", "{{linkedin_url}}"]
    return all(var in template_content for var in required_vars)


def get_template_statistics() -> Dict[str, Any]:
    """Get statistics about the template repository"""
    return {
        "total_templates": len(EMAIL_TEMPLATES),
        "available_types": [t.value for t in TemplateType],
        "default_template": DEFAULT_TEMPLATE_TYPE.value,
        "required_variables": ["name", "company", "linkedin_url"]
    }