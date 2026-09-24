from app.config import llm
from app.graph.state import SupportState
from app.schemas.outputs import ClassifyOutput


def classify(state: SupportState) -> dict:
    structured_llm = llm.with_structured_output(
        ClassifyOutput,
        method="json_schema"
    )

    prompt = f"""
You are the CLASSIFY step of an AI technical support
troubleshooting workflow.

The user's message may contain ONE or MORE independent
technical support problems.

Your job is to:

1. Identify every distinct technical problem.
2. Create one problem entry for each independent problem.
3. Give each problem a short summary.
4. Assign exactly ONE category to each problem.

Supported categories:

- Docker
- Network
- Application
- Account

Do NOT diagnose the problems.

Do NOT provide solutions.

Do NOT merge independent problems into one problem.

If the user mentions multiple problems, return multiple
problem entries.

USER PROBLEM:

{state["problem_summary"]}

ORIGINAL USER MESSAGE:

{state["user_message"]}

CLARIFICATION:

{state.get("clarification", "")}

Return ONLY the structured output.
"""

    response = structured_llm.invoke(prompt)

    problems = [
        {
            "summary": problem.summary,
            "category": problem.category
        }
        for problem in response.problems
    ]

    return {
        "problem_queue": problems,
        "category": problems[0]["category"] if problems else "",
    }
