from typing import Dict, List, Any

OBJECTIVES: List[Dict[str, Any]] = [
    {
        "id": "company_awareness",
        "index": 1,
        "title": "Is the company aware of this position?",
        "target_fact": "company_reached",
        "default_question": "Hello, I'm calling to verify an open job position at {company_name}. Is this {company_name} HR or recruitment office?",
        "allowed_questions": [
            "Hello, am I speaking with the HR or recruiting team at {company_name}?",
            "Hi, is this the official recruitment line for {company_name}?",
            "Good day! Could you confirm if this is {company_name}?"
        ]
    },
    {
        "id": "role_active",
        "index": 2,
        "title": "Is the specific role actively hiring?",
        "target_fact": "role_actively_hiring",
        "default_question": "Could you confirm if {company_name} is actively hiring for the position of {job_role}?",
        "allowed_questions": [
            "Is the position of {job_role} currently open for hiring at {company_name}?",
            "Are you currently accepting applications for the {job_role} position?",
            "Can you verify whether the {job_role} opening is legitimate and active?"
        ]
    },
    {
        "id": "recruiter_legitimacy",
        "index": 3,
        "title": "Is the recruiter legitimate/recognized?",
        "target_fact": "recruiter_recognized",
        "default_question": "Does a recruiter named {recruiter_name} represent {company_name} for this position?",
        "allowed_questions": [
            "Can you confirm if {recruiter_name} is an authorized recruiter for {company_name}?",
            "Is {recruiter_name} associated with your HR or talent acquisition department?",
            "Have candidates been contacted by {recruiter_name} on behalf of {company_name}?"
        ]
    },
    {
        "id": "application_process",
        "index": 4,
        "title": "What is the official application process?",
        "target_fact": "application_channel_verified",
        "default_question": "Could you share the official application process for applying to the {job_role} role?",
        "allowed_questions": [
            "What is the standard procedure for submitting an application for {job_role}?",
            "How do candidates formally submit their resume or application for this role?",
            "Should applicants apply directly through your website careers portal?"
        ]
    },
    {
        "id": "payment_request",
        "index": 5,
        "title": "Is any payment being requested?",
        "target_fact": "payment_requested",
        "default_question": "Does {company_name} ever charge application, training, or equipment processing fees to candidates?",
        "allowed_questions": [
            "Is there any application fee, security deposit, or equipment fee required from applicants?",
            "Just to clarify, candidates do not need to pay anything during hiring, right?",
            "Does {company_name} require any payment for background checks or onboarding materials?"
        ]
    },
    {
        "id": "official_channel",
        "index": 6,
        "title": "What official contact/application channel should applicants use?",
        "target_fact": "application_channel_verified",
        "default_question": "What official website URL or email address should candidates use to verify or submit applications?",
        "allowed_questions": [
            "Could you provide the official domain or email address where candidates should send inquiries?",
            "What is the official careers web page for {company_name}?",
            "Where can applicants double-check the authenticity of job offers from {company_name}?"
        ]
    }
]

def get_objective_by_index(index: int) -> Dict[str, Any]:
    for obj in OBJECTIVES:
        if obj["index"] == index:
            return obj
    return OBJECTIVES[-1]

def format_question_for_context(objective_index: int, context_vars: Dict[str, str]) -> str:
    obj = get_objective_by_index(objective_index)
    company = context_vars.get("company_name") or "your company"
    role = context_vars.get("job_role") or "this role"
    recruiter = context_vars.get("recruiter_name") or "the recruiter"
    
    question_template = obj["default_question"]
    return question_template.format(company_name=company, job_role=role, recruiter_name=recruiter)
