from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
client = OpenAI(
    api_key=os.getenv("open_ai_api_key"),
    base_url="https://openrouter.ai/api/v1"
)

def risk_agent(answer):

    prompt = f"""
Analyze the fraud risk level.

Response:
{answer}

Give:
- Risk Level
- Explanation
- Recommendation
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