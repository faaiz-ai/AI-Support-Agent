"""
General-answer node.

Handles requests classified as "general_question" by the UNDERSTAND
step (i.e. informational questions rather than support issues).
Generates a direct, helpful answer using the shared LLM instead of
routing the user through the troubleshooting workflow.
"""

from app.config import llm
from app.graph.state import SupportState

general_answer_prompt = """
You are the GENERAL ANSWER step of an AI technical-support agent.

The user asked an informational question rather than reporting a
problem. Answer it directly, clearly, and concisely.

USER QUESTION:
{user_message}

QUESTION SUMMARY:
{problem_summary}

Guidelines:
- Answer the question directly; do not ask clarifying questions.
- Keep the answer practical and technically accurate.
- Do not treat this as a troubleshooting issue or suggest
  diagnostic steps unless the user explicitly asked for them.
"""


def general_answer(state: SupportState) -> dict:
    prompt = general_answer_prompt.format(
        user_message=state["user_message"],
        problem_summary=state.get("problem_summary", "")
    )

    response = llm.invoke(prompt)

    print("\n========== GENERAL ANSWER ==========")
    print("QUESTION:", state["user_message"])
    print("ANSWER:", response.content)
    print("=====================================\n")

    return {
        "current_solution": response.content
    }
