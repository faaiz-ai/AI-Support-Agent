"""
Shared state definition for the AI Support Troubleshooting Agent.

This state is threaded through the parent graph and every
category-specific subgraph (Docker, Network, Application, Account).
"""

from typing import TypedDict, Annotated
import operator


class SupportState(TypedDict, total=False):
    user_message: str
    original_user_message: str

    request_type: str
    problem_summary: str
    can_classify: bool
    missing_information: str
    clarification: str

    category: str

    problem_queue: list[dict]
    current_problem: str

    docker_diagnosis: str
    network_diagnosis: str
    application_diagnosis: str
    account_diagnosis: str

    diagnostic_feedback: Annotated[list[str], operator.add]
    diagnostic_question: str
    needs_more_information: bool

    attempted_solutions: Annotated[list[str], operator.add]
    current_solution: str
    solution_feedback: str
    solution_worked: bool
    current_action_type: str

    verified_conditions: list[str]
