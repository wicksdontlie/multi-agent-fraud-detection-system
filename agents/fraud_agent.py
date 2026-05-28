from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
client = OpenAI(
    api_key=os.getenv("open_router_api_key"),
    base_url="https://openrouter.ai/api/v1"
)
def fraud_agent(context, query):

    prompt = f"""
You are a Fraud Detection Expert.

Answer the user's question using ONLY the context.

Context:
{context}

Question:
{query}

Answer:
"""

    response = client.chat.completions.create(
        model="openai/gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content
