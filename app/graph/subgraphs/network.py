"""
Network troubleshooting subgraph.

Diagnoses network-related problems, asks diagnostic questions when
needed, proposes one troubleshooting action at a time, records
attempts, evaluates feedback, and resolves or escalates the issue.
"""

from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt

from app.config import llm, checkpointer
from app.graph.state import SupportState
from app.nodes.shared import record_attempt, process_feedback
from app.schemas.outputs import NetworkDiagnosisOutput, NetworkTroubleshootingOutput

# ---------------------------------------------------------------------------
# 1. Network Diagnosis Node
# ---------------------------------------------------------------------------
network_diagnosis_prompt = """
You are a Network technical support diagnostic agent.

Your job is to understand the current network problem and determine
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

1. Determine the most likely current network diagnosis.
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

network_diagnosis_llm = llm.with_structured_output(
    NetworkDiagnosisOutput,
    method="function_calling"
).with_retry(
    stop_after_attempt=3,
    wait_exponential_jitter=True
)


def network_diagnose(state: SupportState) -> dict:
    def _fmt(items):
        return "\n".join(items) if items else "None"

    prompt = network_diagnosis_prompt.format(
        problem_summary=state.get("problem_summary", ""),
        attempted_solutions=_fmt(state.get("attempted_solutions", [])),
        diagnostic_feedback=_fmt(state.get("diagnostic_feedback", [])),
        verified_conditions=_fmt(state.get("verified_conditions", [])),
        solution_feedback=state.get("solution_feedback", "") or "None"
    )

    result = network_diagnosis_llm.invoke(prompt)

    if len(state.get("diagnostic_feedback", [])) >= 2:
        result.needs_more_information = False
        result.diagnostic_question = ""

    existing_conditions = state.get("verified_conditions", [])
    new_conditions = result.new_verified_conditions
    verified_conditions = existing_conditions.copy()

    for condition in new_conditions:
        if condition not in verified_conditions:
            verified_conditions.append(condition)

    print("\n========== NETWORK DIAGNOSE ==========")
    print("ATTEMPTED SOLUTIONS:")
    print(state.get("attempted_solutions", []))
    print("DIAGNOSTIC FEEDBACK:")
    print(state.get("diagnostic_feedback", []))
    print("VERIFIED CONDITIONS:")
    print(verified_conditions)
    print("NEEDS MORE INFORMATION:")
    print(result.needs_more_information)
    print("DIAGNOSTIC QUESTION:")
    print(result.diagnostic_question)
    print("DIAGNOSIS:")
    print(result.diagnosis)
    print("======================================\n")

    return {
        "network_diagnosis": result.diagnosis,
        "needs_more_information": result.needs_more_information,
        "diagnostic_question": result.diagnostic_question,
        "verified_conditions": verified_conditions,
        "current_action_type": "diagnostic"
    }


def network_ask_diagnostic(state: SupportState) -> dict:
    feedback = interrupt({
        "type": "network_diagnostic",
        "question": state["diagnostic_question"],
    })
    return {
        "diagnostic_feedback": [feedback]
    }


# ---------------------------------------------------------------------------
# 2. Network Support Node
# ---------------------------------------------------------------------------
network_prompt = """
You are a technical support troubleshooting agent specializing in Network issues.

Your job is to troubleshoot the user's Network problem ONE STEP AT A TIME.

Problem:
{problem_summary}

Current diagnosis:
{network_diagnosis}

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

network_structured_llm = llm.with_structured_output(
    NetworkTroubleshootingOutput,
    method="function_calling"
).with_retry(
    stop_after_attempt=3,
    wait_exponential_jitter=True
)


def network_support(state: SupportState) -> dict:
    def _fmt(items):
        return "\n".join(items) if items else "None"

    prompt = network_prompt.format(
        problem_summary=state["problem_summary"],
        network_diagnosis=state.get("network_diagnosis", ""),
        diagnostic_feedback=_fmt(state.get("diagnostic_feedback", [])),
        verified_conditions=_fmt(state.get("verified_conditions", [])),
        attempted_solutions=_fmt(state.get("attempted_solutions", [])),
        solution_feedback=state.get("solution_feedback", "") or "None"
    )

    result = network_structured_llm.invoke(prompt)

    max_retries = 3
    attempts = 0

    while (
        result.solution in state.get("attempted_solutions", [])
        and attempts < max_retries
    ):

        retry_prompt = f"""
You are a Network technical support troubleshooting agent.

Return exactly ONE new troubleshooting action.

Original problem:
{state["problem_summary"]}

Current diagnosis:
{state.get("network_diagnosis", "")}

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

        result = network_structured_llm.invoke(retry_prompt)
        attempts += 1

        print("\n========== NETWORK SUPPORT RETRY ==========")
        print("RETRY NUMBER:", attempts)
        print("GENERATED SOLUTION:", result.solution)
        print("===========================================\n")

    # Still duplicate after all retries → escalate
    if result.solution in state.get("attempted_solutions", []):

        print("\n========== NETWORK SUPPORT ==========")
        print("FAILED TO GENERATE NEW SOLUTION")
        print("ESCALATING")
        print("=====================================\n")

        return {
            "current_action_type": "escalate"
        }

    print("\n========== NETWORK SUPPORT ==========")
    print("ATTEMPTED SOLUTIONS:", state.get("attempted_solutions", []))
    print("GENERATED SOLUTION:", result.solution)
    print("=====================================\n")

    return {
        "current_solution": result.solution,
        "current_action_type": "solution"
    }


def network_ask_solution(state: SupportState) -> dict:
    feedback = interrupt({
        "type": "troubleshooting_solution",
        "solution": state["current_solution"]
    })
    return {"solution_feedback": feedback}


# ---------------------------------------------------------------------------
# 3. Network-specific routers
# ---------------------------------------------------------------------------
def route_after_diagnosis(state: SupportState):
    if state["needs_more_information"]:
        return "network_ask_diagnostic"
    return "network_support"


def route_after_network_feedback(state: SupportState):
    if state["solution_worked"]:
        return "resolved"
    return "network_support"


def route_after_network_support(state: SupportState):
    if state.get("current_action_type") == "escalate":
        return "escalate"

    return "network_ask_solution"


def network_resolved(state: SupportState) -> dict:
    return {
        "current_solution": "Glad that fixed the problem! If you have any other questions or issues, feel free to ask.",
        "current_action_type": "resolved"
    }


def network_escalate(state: SupportState) -> dict:
    return {
        "current_solution": (
            "I couldn't generate a new troubleshooting step "
            "without repeating an action that has already been tried. "
            "This issue should be escalated for further investigation."
        ),
        "current_action_type": "escalated",
    }


# ---------------------------------------------------------------------------
# 4. Network Subgraph
# ---------------------------------------------------------------------------
network_builder = StateGraph(SupportState)

network_builder.add_node("network_diagnose", network_diagnose)
network_builder.add_node("network_ask_diagnostic", network_ask_diagnostic)
network_builder.add_node("network_support", network_support)
network_builder.add_node("network_ask_solution", network_ask_solution)
network_builder.add_node("record_attempt", record_attempt)
network_builder.add_node("process_feedback", process_feedback)
network_builder.add_node("network_resolved", network_resolved)
network_builder.add_node("network_escalate", network_escalate)

network_builder.add_edge(START, "network_diagnose")
network_builder.add_edge("network_ask_diagnostic", "network_diagnose")
network_builder.add_edge("network_ask_solution", "record_attempt")
network_builder.add_edge("record_attempt", "process_feedback")
network_builder.add_edge("network_resolved", END)
network_builder.add_edge("network_escalate", END)

network_builder.add_conditional_edges(
    "network_diagnose",
    route_after_diagnosis,
    {
        "network_ask_diagnostic": "network_ask_diagnostic",
        "network_support": "network_support"
    }
)

network_builder.add_conditional_edges(
    "network_support",
    route_after_network_support,
    {
        "network_ask_solution": "network_ask_solution",
        "escalate": "network_escalate"
    }
)

network_builder.add_conditional_edges(
    "process_feedback",
    route_after_network_feedback,
    {
        "resolved": "network_resolved",
        "network_support": "network_diagnose"
    }
)

network_graph = network_builder.compile(checkpointer=checkpointer)
