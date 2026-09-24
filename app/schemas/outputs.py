"""
Pydantic structured-output schemas used by every LLM-backed node
across the parent graph and the Docker / Network / Application /
Account subgraphs.
"""

from typing import Literal
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Understand step
# ---------------------------------------------------------------------------
class UnderstandOutput(BaseModel):
    request_type: str = Field(
        description=(
            "The type of request. Must be either "
            "'support_issue' or 'general_question'. "
            "Use 'support_issue' when the user is reporting "
            "a technical problem that requires troubleshooting. "
            "Use 'general_question' when the user is asking "
            "for general information rather than reporting a problem."
        )
    )

    problem_summary: str = Field(
        description=(
            "A concise and accurate summary of the user's "
            "main problem or question."
        )
    )

    can_classify: bool = Field(
        description=(
            "For support issues, whether the problem is clear "
            "enough to identify the appropriate support category. "
            "For general questions, always set this to True."
        )
    )

    missing_information: str = Field(
        description=(
            "Basic information needed only when a support issue "
            "is too vague to classify. Return an empty string "
            "for general questions and for support issues that "
            "can already be classified."
        )
    )


# ---------------------------------------------------------------------------
# Classify step
# ---------------------------------------------------------------------------
class ClassifiedProblem(BaseModel):
    summary: str = Field(
        description="A short summary of one distinct technical problem."
    )

    category: Literal[
        "Docker",
        "Network",
        "Application",
        "Account",
    ] = Field(
        description="The support category for this technical problem."
    )


class ClassifyOutput(BaseModel):
    problems: list[ClassifiedProblem] = Field(
        description=(
            "A list of all distinct technical problems mentioned by the user. "
            "Create one item for each independent problem."
        )
    )


# ---------------------------------------------------------------------------
# Shared solution-feedback evaluation
# ---------------------------------------------------------------------------
class SolutionFeedbackOutput(BaseModel):
    worked: bool = Field(
        description=(
            "Whether the previously suggested troubleshooting "
            "solution successfully solved the user's problem."
        )
    )

    explanation: str = Field(
        description=(
            "Brief explanation of what the user's feedback indicates."
        )
    )


# ---------------------------------------------------------------------------
# Docker subgraph
# ---------------------------------------------------------------------------
class DockerDiagnosisOutput(BaseModel):
    diagnosis: str
    diagnostic_question: str
    needs_more_information: bool
    new_verified_conditions: list[str]


class DockerTroubleshootingOutput(BaseModel):
    solution: str
    reasoning: str


# ---------------------------------------------------------------------------
# Network subgraph
# ---------------------------------------------------------------------------
class NetworkDiagnosisOutput(BaseModel):
    diagnosis: str
    diagnostic_question: str
    needs_more_information: bool
    new_verified_conditions: list[str]


class NetworkTroubleshootingOutput(BaseModel):
    solution: str = Field(
        description=(
            "Exactly ONE troubleshooting action. "
            "It must be a single action the user can perform now. "
            "Do not include follow-up actions, alternatives, "
            "conditional instructions, or multiple commands."
        )
    )

    reasoning: str = Field(
        description=(
            "Briefly explain why this single troubleshooting "
            "action is appropriate based on the current problem "
            "and previous feedback."
        )
    )


# ---------------------------------------------------------------------------
# Application subgraph
# ---------------------------------------------------------------------------
class ApplicationDiagnosisOutput(BaseModel):
    diagnosis: str
    diagnostic_question: str
    needs_more_information: bool
    new_verified_conditions: list[str]


class ApplicationTroubleshootingOutput(BaseModel):
    solution: str = Field(
        description=(
            "Exactly ONE troubleshooting action. "
            "It must be a single action the user can perform now. "
            "Do not include follow-up actions, alternatives, "
            "conditional instructions, or multiple commands."
        )
    )

    reasoning: str = Field(
        description=(
            "Briefly explain why this single troubleshooting "
            "action is appropriate based on the current problem "
            "and previous feedback."
        )
    )


# ---------------------------------------------------------------------------
# Account subgraph
# ---------------------------------------------------------------------------
class AccountDiagnosisOutput(BaseModel):
    diagnosis: str
    diagnostic_question: str
    needs_more_information: bool
    new_verified_conditions: list[str]


class AccountTroubleshootingOutput(BaseModel):
    solution: str = Field(
        description=(
            "Exactly ONE troubleshooting action. "
            "It must be a single action the user can perform now. "
            "Do not include follow-up actions, alternatives, "
            "conditional instructions, or multiple commands."
        )
    )

    reasoning: str = Field(
        description=(
            "Briefly explain why this single troubleshooting "
            "action is appropriate based on the current problem "
            "and previous feedback."
        )
    )
