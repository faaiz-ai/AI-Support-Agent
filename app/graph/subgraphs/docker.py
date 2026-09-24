"""
Docker troubleshooting subgraph.

Diagnoses Docker-related problems, asks diagnostic questions when
needed, proposes one troubleshooting action at a time, records
attempts, evaluates feedback, and resolves or escalates the issue.
"""

from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt

from app.config import llm, checkpointer
from app.graph.state import SupportState
from app.nodes.shared import record_attempt, process_feedback
from app.schemas.outputs import DockerDiagnosisOutput, DockerTroubleshootingOutput

# ---------------------------------------------------------------------------
# 1. Docker Diagnosis Node
# ---------------------------------------------------------------------------
docker_diagnosis_llm = llm.with_structured_output(
    DockerDiagnosisOutput,
    method="function_calling"
).with_retry(
    stop_after_attempt=3,
    wait_exponential_jitter=True
)

docker_diagnosis_prompt = """
You are the diagnostic component of a Docker troubleshooting agent.

Determine the current diagnosis, whether another diagnostic question
is necessary, and which facts have been verified.

Problem:
{problem_summary}

Diagnostic evidence:
{diagnostic_feedback}

Previously verified conditions:
{verified_conditions}

Previously attempted solutions:
{attempted_solutions}

Latest solution feedback:
{solution_feedback}

Rules:

1. Treat diagnostic evidence and solution feedback as cumulative.
2. Preserve previously verified conditions unless new evidence
   clearly contradicts them.
3. Add only facts directly confirmed by the evidence to
   verified_conditions.
4. Ask ONE diagnostic question only when information REQUIRED to
   choose the next troubleshooting action is missing.

   Do NOT ask a question merely because its answer would provide
   more certainty or help identify the exact root cause.
5. Do not ask for information that is already known from the
   evidence, verified conditions, or solution feedback.
6. If the available information is sufficient to choose a reasonable
   next troubleshooting action, set needs_more_information to False.
   A previous solution failing does not by itself mean more
   diagnostic information is required.
7. After a solution fails, use the failure as new evidence and
   determine whether a different action can be attempted. Do not
   repeat diagnostic questions for facts that are already established.
8. Do not continue diagnosing just to find the exact root cause.

   If you can identify ANY reasonable evidence-based troubleshooting
   action that tests a new direction, STOP diagnosing and set
   needs_more_information to False.

   Prefer taking a reasonable next action over asking another
   diagnostic question.
9. Do not generate solutions, commands, or fixes.
   The docker_support node handles solutions.
10. If the latest solution worked, set needs_more_information to False
    and diagnostic_question to "".
11. If no question is needed, set diagnostic_question to "".

Keep the diagnosis concise and verified_conditions limited to
important decision-relevant facts.

Respond ONLY by calling the required structured tool.
Do not write explanations, headers, bullets, or markdown outside
the tool call. Call the tool exactly once.
"""


def docker_diagnose(state: SupportState) -> dict:

    def _fmt(items):
        return "\n".join(items) if items else "None"

    prompt = docker_diagnosis_prompt.format(
        problem_summary=state.get("problem_summary", ""),
        attempted_solutions=_fmt(state.get("attempted_solutions", [])),
        diagnostic_feedback=_fmt(state.get("diagnostic_feedback", [])),
        verified_conditions=_fmt(state.get("verified_conditions", [])),
        solution_feedback=state.get("solution_feedback", "") or "None"
    )

    result = docker_diagnosis_llm.invoke(prompt)

    if len(state.get("diagnostic_feedback", [])) >= 2:
        result.needs_more_information = False
        result.diagnostic_question = ""

    existing_conditions = state.get("verified_conditions", [])
    new_conditions = result.new_verified_conditions

    verified_conditions = existing_conditions.copy()

    for condition in new_conditions:
        if condition not in verified_conditions:
            verified_conditions.append(condition)

    needs_more_information = result.needs_more_information
    diagnostic_question = result.diagnostic_question

    print("\n========== DOCKER DIAGNOSE ==========")
    print("ATTEMPTED SOLUTIONS:")
    print(state.get("attempted_solutions", []))
    print("DIAGNOSTIC FEEDBACK:")
    print(state.get("diagnostic_feedback", []))
    print("NEEDS MORE INFORMATION:")
    print(needs_more_information)
    print("DIAGNOSTIC QUESTION:")
    print(diagnostic_question)
    print("DIAGNOSIS:")
    print(result.diagnosis)
    print("=====================================\n")

    return {
        "docker_diagnosis": result.diagnosis,
        "needs_more_information": needs_more_information,
        "diagnostic_question": diagnostic_question,
        "verified_conditions": verified_conditions,
        "current_action_type": "diagnostic"
    }


def docker_ask_diagnostic(state: SupportState) -> dict:

    feedback = interrupt({
        "type": "docker_diagnostic",
        "question": state["diagnostic_question"],
        "diagnosis": state["docker_diagnosis"]
    })

    return {
        "diagnostic_feedback": [feedback]
    }


# ---------------------------------------------------------------------------
# 2. Docker Solution Node
# ---------------------------------------------------------------------------
docker_prompt = """
You are a Docker troubleshooting agent.

Generate exactly ONE troubleshooting action based on the information below.

Problem:
{problem_summary}

Diagnosis:
{docker_diagnosis}

Diagnostic evidence:
{diagnostic_feedback}

Verified conditions:
{verified_conditions}

Previously attempted solutions:
{attempted_solutions}

Latest feedback:
{solution_feedback}

Requirements:
- Give exactly one atomic troubleshooting action.
- The action must be one command OR one user action.
- If a fix normally requires multiple steps, give only the FIRST necessary action.
- Do not give alternatives.
- Do not combine multiple commands or steps.
- Do not include verification or follow-up steps.
- Do not repeat a previously attempted action.
- Do not reword or slightly modify a previously attempted action.
- Do not test the same underlying cause again.
- Prefer the smallest useful next action.
- Return only the required structured output.
"""

structured_llm = llm.with_structured_output(
    DockerTroubleshootingOutput,
    method="function_calling"
).with_retry(
    stop_after_attempt=3,
    wait_exponential_jitter=True
)


def docker_support(state: SupportState) -> dict:

    def _fmt(items):
        return "\n".join(items) if items else "None"

    attempted = state.get("attempted_solutions", [])

    prompt = docker_prompt.format(
        problem_summary=state["problem_summary"],
        docker_diagnosis=state.get("docker_diagnosis", ""),
        diagnostic_feedback=_fmt(state.get("diagnostic_feedback", [])),
        verified_conditions=_fmt(state.get("verified_conditions", [])),
        attempted_solutions=_fmt(attempted),
        solution_feedback=state.get("solution_feedback", "") or "None"
    )

    result = structured_llm.invoke(prompt)

    print("\n========== DOCKER SUPPORT ==========")
    print("ATTEMPTED SOLUTIONS:")
    print(attempted)
    print("GENERATED SOLUTION:")
    print(result.solution)
    print("====================================\n")

    max_retries = 3
    attempts = 0

    while result.solution in attempted and attempts < max_retries:

        retry_prompt = prompt + f"""

IMPORTANT:

The generated action below has already been attempted:

{result.solution}

Generate exactly ONE genuinely different troubleshooting action.

Do not repeat or reword any previously attempted action.
Do not test the same underlying cause again.
"""

        result = structured_llm.invoke(retry_prompt)
        attempts += 1

        print("\n========== DOCKER SUPPORT RETRY ==========")
        print("RETRY NUMBER:", attempts)
        print("GENERATED SOLUTION:")
        print(result.solution)
        print("==========================================\n")

    # Still duplicate after all retries → escalate
    if result.solution in attempted:

        print("\n========== DOCKER SUPPORT ==========")
        print("FAILED TO GENERATE NEW SOLUTION")
        print("ESCALATING")
        print("=====================================\n")

        return {
            "current_action_type": "escalate"
        }

    return {
        "current_solution": result.solution,
        "current_action_type": "solution"
    }


def docker_ask_solution(state: SupportState) -> dict:

    feedback = interrupt({
        "type": "troubleshooting_solution",
        "solution": state["current_solution"],
    })

    return {
        "solution_feedback": feedback
    }


# ---------------------------------------------------------------------------
# 3. Docker-specific routers
# ---------------------------------------------------------------------------
def route_after_docker_diagnosis(state: SupportState):
    if state["needs_more_information"]:
        return "docker_ask_diagnostic"
    return "docker_support"


def route_after_docker_support(state: SupportState):

    if state.get("current_action_type") == "escalate":
        return "escalate"

    return "docker_ask_solution"


def route_after_docker_feedback(state: SupportState):

    if state["solution_worked"]:
        return "resolved"

    return "docker_support"


# ---------------------------------------------------------------------------
# 4. Docker Resolve Node
# ---------------------------------------------------------------------------
def docker_resolved(state: SupportState) -> dict:

    return {
        "current_solution": (
            "Glad that fixed the problem! "
            "If you have any other questions or issues, feel free to ask."
        ),
        "current_action_type": "resolved",
    }


# ---------------------------------------------------------------------------
# 5. Docker Escalate Node
# ---------------------------------------------------------------------------
def docker_escalate(state: SupportState) -> dict:

    return {
        "current_solution": (
            "I couldn't generate a new troubleshooting step "
            "without repeating an action that has already been tried. "
            "This issue should be escalated for further investigation."
        ),
        "current_action_type": "escalated",
    }


# ---------------------------------------------------------------------------
# 6. Docker Subgraph
# ---------------------------------------------------------------------------
docker_builder = StateGraph(SupportState)

docker_builder.add_node("docker_diagnose", docker_diagnose)
docker_builder.add_node("docker_ask_diagnostic", docker_ask_diagnostic)
docker_builder.add_node("docker_support", docker_support)
docker_builder.add_node("docker_ask_solution", docker_ask_solution)
docker_builder.add_node("record_attempt", record_attempt)
docker_builder.add_node("process_feedback", process_feedback)
docker_builder.add_node("docker_resolved", docker_resolved)
docker_builder.add_node("docker_escalate", docker_escalate)

docker_builder.add_edge(START, "docker_diagnose")
docker_builder.add_edge("docker_ask_diagnostic", "docker_diagnose")
docker_builder.add_edge("docker_ask_solution", "record_attempt")
docker_builder.add_edge("record_attempt", "process_feedback")
docker_builder.add_edge("docker_resolved", END)
docker_builder.add_edge("docker_escalate", END)

docker_builder.add_conditional_edges(
    "docker_diagnose", route_after_docker_diagnosis,
    {"docker_ask_diagnostic": "docker_ask_diagnostic", "docker_support": "docker_support"}
)

docker_builder.add_conditional_edges(
    "docker_support", route_after_docker_support,
    {"docker_ask_solution": "docker_ask_solution", "escalate": "docker_escalate"}
)

docker_builder.add_conditional_edges(
    "process_feedback", route_after_docker_feedback,
    {"resolved": "docker_resolved", "docker_support": "docker_diagnose"}
)

docker_graph = docker_builder.compile(checkpointer=checkpointer)
