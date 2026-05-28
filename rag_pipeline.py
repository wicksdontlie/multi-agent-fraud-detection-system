from openai import OpenAI
from retriever import retrieve
from dotenv import load_dotenv
import os

load_dotenv()

# =====================================
# OpenRouter Client
# =====================================

client = OpenAI(
    api_key=os.getenv("open_router_openai_api_key"),
    base_url="https://openrouter.ai/api/v1"
)

# =====================================
# RAG Function
# =====================================

def ask_rag(query):

    print("Retrieving chunks...")

    results = retrieve(query)

    print("Chunks retrieved")

    context = "\n\n".join(results)

    prompt = f"""
You are a Fraud Detection and AML Expert.

Answer ONLY from the provided context.

If answer is not found in context,
say:
'I could not find the answer in the documents.'

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:
"""

    print("Sending request to OpenRouter...")

    response = client.chat.completions.create(
        model="openai/gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    print("Response received")

    return response.choices[0].message.content

# =====================================
# TEST
# =====================================

query = "What is money laundering?"

answer = ask_rag(query)

print("\nANSWER:\n")

print(answer)