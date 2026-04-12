from datetime import datetime

from openai import AsyncOpenAI
from agents import OpenAIChatCompletionsModel, set_tracing_disabled
from pydantic import BaseModel, Field
from typing import Literal
from agents.tracing import set_tracing_disabled
from agents import (Agent, Runner, ModelSettings, function_tool)


set_tracing_disabled(True)

def get_ollama_model():
    client = AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    local_model = OpenAIChatCompletionsModel(
        model="qwen2.5-coder:3b", openai_client=client
    )
    return local_model

MODEL = get_ollama_model()

classifier_agent = Agent(
    name="Ticket Classifier",
    instructions="""
    You classify customer support messages.
    Analyze the message and return structured classification data.
    Be accurate — wrong classification wastes everyone's time.
    
    Priority guide:
    - P1-critical: Service down, data loss, security issue
    - P2-high: Feature broken, payment failure, angry customer
    - P3-medium: Bug report, how-to question, feature request
    - P4-low: General feedback, suggestions
    """,
    model=MODEL,  
    output_type=TicketClassification,         # Forces structured JSON output
    model_settings=ModelSettings(temperature=0.1),  # Low temp = consistent results
)

def support_instructions(context, agent):
    """Dynamic instructions that change based on current state."""
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
    return f"""
    You are an AI customer support agent for "CloudSync" — a SaaS platform.
    Current time: {current_time}

    YOUR WORKFLOW (follow this order):
    1. Greet the customer professionally.
    2. Look up their account using lookup_customer if they provide an email.
    3. Understand their issue — search_knowledge_base for known solutions.
    4. If it's a service issue, check_service_status to see if it's a known outage.
    5. If you can solve it, provide the solution clearly.
    6. If you can't solve it, create_ticket to escalate.
    7. Always end with: "Is there anything else I can help with?"

    YOUR RULES:
    - Be empathetic, professional, and concise.
    - If the customer is angry, acknowledge their frustration first.
    - ALWAYS use tools to get real data — never make up information.
    - For billing issues on Enterprise plans, always escalate (create a ticket).
    - Include ticket ID when creating tickets.
    """

@function_tool
def lookup_customer(email: str) -> str:
    """Look up customer details by their email address."""
    # In production: query your CRM/database
    customers = {
        "ahmed@example.com": {
            "name": "Ahmed Hassan",
            "plan": "Enterprise",
            "since": "2023-01",
            "mrr": "$299/mo",
            "tickets_open": 2,
        },
        "sara@startup.io": {
            "name": "Sara Khan",
            "plan": "Pro",
            "since": "2024-06",
            "mrr": "$49/mo",
            "tickets_open": 0,
        },
    }
    customer = customers.get(email.lower())
    if not customer:
        return f"No customer found with email: {email}"
    return (
        f"Customer: {customer['name']}\n"
        f"Plan: {customer['plan']} ({customer['mrr']})\n"
        f"Customer since: {customer['since']}\n"
        f"Open tickets: {customer['tickets_open']}"
    )

@function_tool
def check_service_status(service: str) -> str:
    """Check the current status of a company service/feature."""
    statuses = {
        "api":       "Operational (99.98% uptime, 45ms avg latency)",
        "dashboard": "Degraded (slow loading, team investigating)",
        "billing":   "Operational",
        "auth":      "Operational",
    }
    status = statuses.get(service.lower())
    if not status:
        return f"Unknown service: {service}. Available: {list(statuses.keys())}"
    return f"{service}: {status}"

@function_tool
def create_ticket(
    customer_email: str,
    category: str,
    priority: str,
    description: str,
) -> str:
    """Create a support ticket in the system."""
    ticket_id = f"TKT-{abs(hash(description)) % 100000:05d}"
    return (
        f"Ticket created!\n"
        f"ID: {ticket_id}\n"
        f"Customer: {customer_email}\n"
        f"Category: {category} | Priority: {priority}\n"
        f"Description: {description[:100]}...\n"
        f"Created: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    )

@function_tool
def search_knowledge_base(query: str) -> str:
    """Search the internal knowledge base for help articles."""
    articles = {
        "password": "Article #101: Reset password at settings > security > change password. "
                    "If locked out, use 'Forgot Password' on login page.",
        "billing":  "Article #201: Invoices are generated on the 1st of each month. "
                    "Change plan at settings > billing > change plan.",
        "api":      "Article #301: API docs at docs.example.com/api. "
                    "Rate limit: 1000 req/min (Pro), 10000 req/min (Enterprise).",
        "export":   "Article #401: Export data at settings > data > export. "
                    "Supports CSV, JSON, and PDF formats.",
    }
    for key, article in articles.items():
        if key in query.lower():
            return article
    return "No relevant articles found. Escalate to human agent."

agent = Agent(
    name="CloudSync Support", 
    instructions=support_instructions, 
    model=MODEL, 
    model_settings=ModelSettings(
    temperature=0.3,
    max_tokens=1000,
    ),
    tools=[
        lookup_customer,
        check_service_status,
        create_ticket,
        search_knowledge_base,
        classifier_agent.as_tool(
        tool_name="classify_ticket",
        tool_description="Classify a customer message into category, priority, and severity"
        ),
    ]
)

query = "Hi"
result = Runner.run_sync(agent, query)
print(result.final_output)