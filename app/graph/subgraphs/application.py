"""
Application troubleshooting subgraph.

Diagnoses application-related problems, asks diagnostic questions
when needed, proposes one troubleshooting action at a time, records
attempts, evaluates feedback, and resolves or escalates the issue.
"""

from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt

from app.config import llm, checkpointer
from app.graph.state import SupportState
from app.nodes.shared import record_attempt, process_feedback
from app.schemas.outputs import ApplicationDiagnosisOutput, ApplicationTroubleshootingOutput

# ---------------------------------------------------------------------------
# 1. Application Diagnosis Node
# ---------------------------------------------------------------------------
application_diagnosis_prompt = """
You are an Application technical support diagnostic agent.

Your job is to understand the current application problem and determine
what information is still needed before suggesting a solution.

Problem:
{problem_summary}

Previously attempted solutions:
{attempted_solutions}

Previous diagnostic feedback:
{diagnostic_feedback}

Previously verified conditions:
{verified_conditions}

Latest solution feedback:
{solution_feedback}

Rules:

1. Determine the most likely current application diagnosis.
2. If information is genuinely required to choose a useful
   troubleshooting action, ask exactly ONE diagnostic question.
3. Do not suggest a troubleshooting solution yet.
4. Do not ask for information that has already been provided
   or verified.
5. If the available information is sufficient to identify a
   reasonable troubleshooting direction, set needs_more_information
   to false and leave diagnostic_question empty.
6. Record only genuinely new facts in new_verified_conditions.
7. Do not repeat previously verified conditions.
8. Do not ask diagnostic questions just to improve the diagnosis.
   Once you can recommend a reasonable first troubleshooting test,
   STOP diagnosing and set needs_more_information to false.

Return only the required structured output.
"""

application_diagnosis_llm = llm.with_structured_output(
    ApplicationDiagnosisOutput,
    method="function_calling"
).with_retry(
    stop_after_attempt=3,
    wait_exponential_jitter=True
)


def application_diagnose(state: SupportState) -> dict:
    def _fmt(items):
        return "\n".join(items) if items else "None"

    prompt = application_diagnosis_prompt.format(
        problem_summary=state.get("problem_summary", ""),
        attempted_solutions=_fmt(state.get("attempted_solutions", [])),
        diagnostic_feedback=_fmt(state.get("diagnostic_feedback", [])),
        verified_conditions=_fmt(state.get("verified_conditions", [])),
        solution_feedback=state.get("solution_feedback", "") or "None"
    )

    result = application_diagnosis_llm.invoke(prompt)

    if len(state.get("diagnostic_feedback", [])) >= 2:
        result.needs_more_information = False
        result.diagnostic_question = ""

    existing_conditions = state.get("verified_conditions", [])
    new_conditions = result.new_verified_conditions
    verified_conditions = existing_conditions.copy()

    for condition in new_conditions:
        if condition not in verified_conditions:
            verified_conditions.append(condition)

    print("\n========== APPLICATION DIAGNOSE ==========")
    print("ATTEMPTED SOLUTIONS:", state.get("attempted_solutions", []))
    print("DIAGNOSTIC FEEDBACK:", state.get("diagnostic_feedback", []))
    print("VERIFIED CONDITIONS:", verified_conditions)
    print("NEEDS MORE INFORMATION:", result.needs_more_information)
    print("DIAGNOSTIC QUESTION:", result.diagnostic_question)
    print("DIAGNOSIS:", result.diagnosis)
    print("==========================================\n")

    return {
        "application_diagnosis": result.diagnosis,
        "needs_more_information": result.needs_more_information,
        "diagnostic_question": result.diagnostic_question,
        "verified_conditions": verified_conditions,
        "current_action_type": "diagnostic"
    }


def application_ask_diagnostic(state: SupportState) -> dict:
    feedback = interrupt({
        "type": "application_diagnostic",
        "question": state["diagnostic_question"],
    })
    return {"diagnostic_feedback": [feedback]}


# ---------------------------------------------------------------------------
# 2. Application Support Node
# ---------------------------------------------------------------------------
application_prompt = """
You are a technical support troubleshooting agent specializing in Application issues.

Your job is to troubleshoot the user's Application problem ONE STEP AT A TIME.

Problem:
{problem_summary}

Current diagnosis:
{application_diagnosis}

Diagnostic evidence:
{diagnostic_feedback}

Verified conditions:
{verified_conditions}

Previously attempted solutions:
{attempted_solutions}

Latest solution feedback:
{solution_feedback}

Rules:

1. Return exactly ONE troubleshooting action.
2. The action must test a new troubleshooting direction.
3. Do not repeat or reword a previously attempted action,
   and do not use a different command to check the same
   underlying cause.
4. Do not recommend an action that is already verified,
   unnecessary, or ruled out by the available evidence.
5. Use failed attempts and latest feedback to eliminate
   previously tested causes.
6. Prefer the smallest useful test that distinguishes
   between the remaining possible causes.
7. Do not provide alternatives, multiple actions, or branches.
8. Give a practical action the user can perform now.

Respond ONLY using the required structured output.
Call it exactly once.
"""

application_structured_llm = llm.with_structured_output(
    ApplicationTroubleshootingOutput,
    method="function_calling"
).with_retry(
    stop_after_attempt=3,
    wait_exponential_jitter=True
)


def application_support(state: SupportState) -> dict:
    def _fmt(items):
        return "\n".join(items) if items else "None"

    attempted = state.get("attempted_solutions", [])

    prompt = application_prompt.format(
        problem_summary=state["problem_summary"],
        application_diagnosis=state.get("application_diagnosis", ""),
        diagnostic_feedback=_fmt(state.get("diagnostic_feedback", [])),
        verified_conditions=_fmt(state.get("verified_conditions", [])),
        attempted_solutions=_fmt(attempted),
        solution_feedback=state.get("solution_feedback", "") or "None"
    )

    result = application_structured_llm.invoke(prompt)

    max_retries = 3
    attempts = 0

    while result.solution in attempted and attempts < max_retries:

        retry_prompt = f"""
You are a technical support troubleshooting agent specializing in Application issues.

Return exactly ONE new troubleshooting action.

Original problem:
{state["problem_summary"]}

Current diagnosis:
{state.get("application_diagnosis", "")}

Diagnostic evidence:
{_fmt(state.get("diagnostic_feedback", []))}

Verified conditions:
{_fmt(state.get("verified_conditions", []))}

Previously attempted solutions:
{_fmt(state.get("attempted_solutions", []))}

Latest feedback:
{state.get("solution_feedback", "") or "None"}

Previous generated action:
{result.solution}

Generate ONE genuinely different troubleshooting action.

Do not repeat or reword any previously attempted action.
Do not test the same underlying cause again.
Do not recommend something already confirmed by the evidence.
Do not provide multiple actions or alternatives.

Respond ONLY using the required structured output.
Call it exactly once.
"""

        result = application_structured_llm.invoke(retry_prompt)
        attempts += 1

        print("\n========== APPLICATION SUPPORT RETRY ==========")
        print("RETRY NUMBER:", attempts)
        print("GENERATED SOLUTION:", result.solution)
        print("================================================\n")

    # Still duplicate after all retries → escalate
    if result.solution in attempted:

        print("\n========== APPLICATION SUPPORT ==========")
        print("FAILED TO GENERATE NEW SOLUTION")
        print("ESCALATING")
        print("==========================================\n")

        return {
            "current_action_type": "escalate"
        }

    print("\n========== APPLICATION SUPPORT ==========")
    print("ATTEMPTED SOLUTIONS:", attempted)
    print("GENERATED SOLUTION:", result.solution)
    print("==========================================\n")

    return {
        "current_solution": result.solution,
        "current_action_type": "solution"
    }


def application_ask_solution(state: SupportState) -> dict:
    feedback = interrupt({
        "type": "troubleshooting_solution",
        "solution": state["current_solution"]
    })
    return {"solution_feedback": feedback}


# ---------------------------------------------------------------------------
# 3. Application-specific routers
# ---------------------------------------------------------------------------
def route_after_application_diagnosis(state: SupportState):
    if state["needs_more_information"]:
        return "application_ask_diagnostic"
    return "application_support"


def route_after_application_support(state: SupportState):
    if state.get("current_action_type") == "escalate":
        return "escalate"

    return "application_ask_solution"


def route_after_application_feedback(state: SupportState):
    if state["solution_worked"]:
        return "resolved"
    return "application_support"


def application_resolved(state: SupportState) -> dict:
    return {
        "current_solution": "Glad that fixed the problem! If you have any other questions or issues, feel free to ask.",
        "current_action_type": "resolved"
    }


def application_escalate(state: SupportState) -> dict:
    return {
        "current_solution": (
            "I couldn't generate a new troubleshooting step "
            "without repeating an action that has already been tried. "
            "This issue should be escalated for further investigation."
        ),
        "current_action_type": "escalated"
    }


# ---------------------------------------------------------------------------
# 4. Application Subgraph
# ---------------------------------------------------------------------------
application_builder = StateGraph(SupportState)

application_builder.add_node("application_diagnose", application_diagnose)
application_builder.add_node("application_ask_diagnostic", application_ask_diagnostic)
application_builder.add_node("application_support", application_support)
application_builder.add_node("application_ask_solution", application_ask_solution)
application_builder.add_node("record_attempt", record_attempt)
application_builder.add_node("process_feedback", process_feedback)
application_builder.add_node("application_resolved", application_resolved)
application_builder.add_node("application_escalate", application_escalate)

application_builder.add_edge(START, "application_diagnose")
application_builder.add_edge("application_ask_diagnostic", "application_diagnose")
application_builder.add_edge("application_ask_solution", "record_attempt")
application_builder.add_edge("record_attempt", "process_feedback")
application_builder.add_edge("application_resolved", END)
application_builder.add_edge("application_escalate", END)

application_builder.add_conditional_edges(
    "application_diagnose",
    route_after_application_diagnosis,
    {
        "application_ask_diagnostic": "application_ask_diagnostic",
        "application_support": "application_support"
    }
)

application_builder.add_conditional_edges(
    "application_support",
    route_after_application_support,
    {
        "application_ask_solution": "application_ask_solution",
        "escalate": "application_escalate"
    }
)

application_builder.add_conditional_edges(
    "process_feedback",
    route_after_application_feedback,
    {
        "resolved": "application_resolved",
        "application_support": "application_diagnose"
    }
)

application_graph = application_builder.compile(
    checkpointer=checkpointer
)
