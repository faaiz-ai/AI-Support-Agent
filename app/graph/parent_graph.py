from langgraph.graph import StateGraph, START, END
from langgraph.types import Overwrite

from app.config import checkpointer
from app.graph.state import SupportState
from app.graph.subgraphs.docker import docker_graph
from app.graph.subgraphs.network import network_graph
from app.graph.subgraphs.application import application_graph
from app.graph.subgraphs.account import account_graph
from app.nodes.understand import understand_request, route_after_understand
from app.nodes.classify import classify
from app.nodes.clarification import ask_clarification
from app.nodes.general_answer import general_answer


def prepare_next_problem(state: SupportState) -> dict:
    problem_queue = list(state.get("problem_queue", []))

    if not problem_queue:
        return {}

    next_problem = problem_queue.pop(0)

    return {
        "problem_queue": problem_queue,
        "current_problem": next_problem["summary"],
        "user_message": next_problem["summary"],
        "problem_summary": next_problem["summary"],
        "category": next_problem["category"],
        "diagnostic_feedback": Overwrite([]),
        "attempted_solutions": Overwrite([]),
        "verified_conditions": [],
        "diagnostic_question": "",
        "needs_more_information": False,
        "current_solution": "",
        "solution_feedback": "",
        "solution_worked": False,
        "current_action_type": "",
    }


def route_to_current_problem(state: SupportState):
    return state["category"]


def advance_after_subgraph(state: SupportState):
    if state.get("problem_queue"):
        return "prepare_next_problem"
    return "finish"


def finish(state: SupportState) -> dict:
    return {}


builder = StateGraph(SupportState)

builder.add_node("understand", understand_request)
builder.add_node("general_answer", general_answer)
builder.add_node("ask_clarification", ask_clarification)
builder.add_node("classify", classify)
builder.add_node("prepare_next_problem", prepare_next_problem)
builder.add_node("docker_subgraph", docker_graph)
builder.add_node("network_subgraph", network_graph)
builder.add_node("application_subgraph", application_graph)
builder.add_node("account_subgraph", account_graph)
builder.add_node("finish", finish)

builder.add_edge(START, "understand")

builder.add_conditional_edges(
    "understand",
    route_after_understand,
    {
        "general_answer": "general_answer",
        "ask_clarification": "ask_clarification",
        "classify": "classify",
    },
)

builder.add_edge("general_answer", END)
builder.add_edge("ask_clarification", "understand")
builder.add_edge("classify", "prepare_next_problem")

builder.add_conditional_edges(
    "prepare_next_problem",
    route_to_current_problem,
    {
        "Docker": "docker_subgraph",
        "Network": "network_subgraph",
        "Application": "application_subgraph",
        "Account": "account_subgraph",
    },
)

builder.add_conditional_edges(
    "docker_subgraph",
    advance_after_subgraph,
    {
        "prepare_next_problem": "prepare_next_problem",
        "finish": "finish",
    },
)

builder.add_conditional_edges(
    "network_subgraph",
    advance_after_subgraph,
    {
        "prepare_next_problem": "prepare_next_problem",
        "finish": "finish",
    },
)

builder.add_conditional_edges(
    "application_subgraph",
    advance_after_subgraph,
    {
        "prepare_next_problem": "prepare_next_problem",
        "finish": "finish",
    },
)

builder.add_conditional_edges(
    "account_subgraph",
    advance_after_subgraph,
    {
        "prepare_next_problem": "prepare_next_problem",
        "finish": "finish",
    },
)

builder.add_edge("finish", END)

graph = builder.compile(checkpointer=checkpointer)
