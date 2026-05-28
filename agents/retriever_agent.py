import sys
import os

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from retriever import retrieve


def retriever_agent(query):

    results = retrieve(query)

    context = "\n\n".join(results)

    return context