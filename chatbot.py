"""
Command-line chatbot for the AI Support Troubleshooting Agent.

Runs the compiled parent graph (app.graph.parent_graph.graph) in a
loop, handling LangGraph interrupts (clarification questions,
diagnostic questions, and "did the solution work" prompts) by
pausing for user input and resuming the graph with
Command(resume=...).

Run with:  python chatbot.py
"""

import uuid

from langgraph.errors import GraphRecursionError
from langgraph.types import Command

from app.graph.parent_graph import graph

EXIT_COMMANDS = {"exit", "quit"}
RECURSION_LIMIT = 50


class ExitConversation(Exception):
    """Raised internally when the user asks to exit mid-conversation."""


def _check_exit(user_input: str) -> None:
    if user_input.strip().lower() in EXIT_COMMANDS:
        raise ExitConversation


def _print_interrupt(interrupt_obj) -> None:
    """
    Pretty-print an interrupt's payload to the user.
    Payloads can be a plain string (ask_clarification) or a dict
    (subgraph diagnostic / solution interrupts).
    """
    value = interrupt_obj.value

    if isinstance(value, dict):
        message = value.get("question") or value.get("solution") or str(value)
    else:
        message = value

    print(f"\nAgent: {message}")


def _final_message(result: dict) -> str:
    """
    Pull the right piece of the final state to show the user,
    covering the general-question, resolved, and escalated paths.
    """
    if result.get("request_type") == "general_question":
        return result.get("current_solution", "(no answer generated)")

    if result.get("current_action_type") in ("resolved", "escalated"):
        return result.get("current_solution", "")

    return result.get("current_solution") or str(result)


def _invoke(payload, config: dict) -> dict:
    """
    Wraps graph.invoke with a recursion limit and basic error
    handling so a transient LLM/API failure doesn't crash the CLI.
    """
    try:
        return graph.invoke(
            payload,
            config={**config, "recursion_limit": RECURSION_LIMIT}
        )
    except GraphRecursionError:
        print(
            "\nAgent: This is taking longer than expected to resolve. "
            "I'd recommend escalating this to a human agent."
        )
        raise ExitConversation
    except Exception as exc:  # noqa: BLE001 - surface any API/LLM error to the user
        print(f"\nAgent: Something went wrong while processing that ({exc}). "
              f"Let's try again.")
        raise ExitConversation


def run_conversation() -> None:
    """
    Runs a single problem/question through the graph to completion,
    handling any number of interrupts along the way.
    """
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    user_message = input("\nYou: ").strip()
    _check_exit(user_message)

    if not user_message:
        print("Please type a message.")
        return

    result = _invoke({"user_message": user_message}, config)

    # Keep resuming as long as the graph is paused on an interrupt.
    while "__interrupt__" in result:
        interrupt_obj = result["__interrupt__"][0]
        _print_interrupt(interrupt_obj)

        user_reply = input("You: ").strip()
        _check_exit(user_reply)

        result = _invoke(Command(resume=user_reply), config)

    # Graph reached an END state — print the final outcome.
    print(f"\nAgent: {_final_message(result)}")


def main() -> None:
    print("AI Support Troubleshooting Agent — type your issue or question.")
    print("Type 'exit' or 'quit' at any prompt to stop.\n")

    while True:
        try:
            run_conversation()
        except ExitConversation:
            print("\nGoodbye!")
            break
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        again = input("\nStart another conversation? (y/n): ").strip().lower()
        if again not in ("y", "yes"):
            print("Goodbye!")
            break


if __name__ == "__main__":
    main()
