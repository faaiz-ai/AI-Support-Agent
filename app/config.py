"""
Central configuration for the AI Support Troubleshooting Agent.

Holds environment loading, the shared LLM instance, and the shared
checkpointer used by the parent graph and all subgraphs.
"""

import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

# Fail fast with a clear message if the API key is missing, instead
# of letting ChatGroq raise a confusing error later on.
if not os.getenv("GROQ_API_KEY"):
    raise RuntimeError(
        "GROQ_API_KEY is not set. Add it to your .env file, e.g.:\n"
        "GROQ_API_KEY=your_key_here"
    )

# Shared LLM instance used by every node in the graph.
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.1,
    reasoning_effort="medium",
)

# Shared in-memory checkpointer used by the parent graph and all
# category subgraphs so that interrupts / human-in-the-loop flows
# work consistently across the whole application.
checkpointer = InMemorySaver()
