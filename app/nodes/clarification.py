"""
Clarification node.

Used when the UNDERSTAND step determines that a support issue is
too vague to classify. Interrupts the graph and waits for the user
to supply the missing basic information.
"""

from langgraph.types import interrupt

from app.graph.state import SupportState


def ask_clarification(state: SupportState) -> dict:
    """
    Ask the user for the basic information required
    to understand and classify their support issue.
    """

    question = (
        f"I need a little more information to understand your problem. "
        f"{state['missing_information']}"
    )

    user_response = interrupt(question)

    return {
        "clarification": user_response
    }
