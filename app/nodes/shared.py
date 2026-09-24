"""
Nodes shared across every category subgraph (Docker, Network,
Application, Account): recording an attempted solution and
evaluating the user's feedback about whether it worked.
"""

from app.config import llm
from app.graph.state import SupportState
from app.schemas.outputs import SolutionFeedbackOutput

feedback_prompt = """
You are evaluating the user's feedback about a troubleshooting action.

Original problem:
{problem_summary}

Troubleshooting action that was suggested:
{current_solution}

User's feedback:
{solution_feedback}

Determine whether the troubleshooting action successfully solved
the original problem.

Rules:
- worked = True ONLY if the problem was successfully resolved.
- worked = False if the problem remains unresolved.
- If the user reports an error, failure, or incomplete result, use False.
- Do not assume success if the user's feedback is unclear.
"""

feedback_llm = llm.with_structured_output(
    SolutionFeedbackOutput
)


def record_attempt(state: SupportState) -> dict:

    attempted = state.get("attempted_solutions", [])
    current = state.get("current_solution", "")

    print("\n========== RECORD ATTEMPT ==========")
    print("CURRENT SOLUTION:", current)
    print("BEFORE:", attempted)

    new_attempts = attempted + [current]

    print("AFTER:", new_attempts)
    print("====================================")

    return {
        "attempted_solutions": [current]
    }


def process_feedback(state: SupportState) -> dict:

    prompt = feedback_prompt.format(
        problem_summary=state["problem_summary"],
        current_solution=state["current_solution"],
        solution_feedback=state["solution_feedback"]
    )

    result = feedback_llm.invoke(prompt)

    feedback = state.get("solution_feedback", "").strip()

    return {
        "solution_worked": result.worked,
        "diagnostic_feedback": [feedback] if feedback else []
    }
