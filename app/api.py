"""
FastAPI backend for the AI Support Troubleshooting Agent.

Exposes the compiled LangGraph parent graph over HTTP so a web
frontend can drive the same conversation flow the CLI (chatbot.py)
supports, including clarification / diagnostic / "did it work"
interrupts.

Run with:  uvicorn app.api:app --reload --port 8000
"""

import uuid

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from langgraph.errors import GraphRecursionError
from langgraph.types import Command

from app.graph.parent_graph import graph

RECURSION_LIMIT = 50

app = FastAPI(title="AI Support Troubleshooting Agent API")

# Allow the frontend (served separately or via StaticFiles below) to
# call this API during local development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class StartRequest(BaseModel):
    message: str


class ReplyRequest(BaseModel):
    thread_id: str
    message: str


class ChatResponse(BaseModel):
    thread_id: str
    reply: str
    waiting_for_input: bool
    done: bool


def _extract_interrupt_message(interrupt_obj) -> str:
    """Pull a printable question out of an interrupt's payload."""
    value = interrupt_obj.value
    if isinstance(value, dict):
        return value.get("question") or value.get("solution") or str(value)
    return value


def _extract_final_message(result: dict) -> str:
    """Pull the right piece of final state to show the user."""
    if result.get("request_type") == "general_question":
        return result.get("current_solution", "(no answer generated)")

    if result.get("current_action_type") in ("resolved", "escalated"):
        return result.get("current_solution", "")

    return result.get("current_solution") or "(no response generated)"


def _run_graph(payload, config: dict) -> ChatResponse:
    """
    Invokes the graph (or resumes it) and turns the resulting state
    into a ChatResponse, handling interrupts and graph errors the
    same way the CLI does.
    """
    try:
        result = graph.invoke(
            payload,
            config={**config, "recursion_limit": RECURSION_LIMIT},
        )
    except GraphRecursionError:
        return ChatResponse(
            thread_id=config["configurable"]["thread_id"],
            reply=(
                "This is taking longer than expected to resolve. "
                "I'd recommend escalating this to a human agent."
            ),
            waiting_for_input=False,
            done=True,
        )
    except Exception as exc:  # noqa: BLE001 - surface any API/LLM error to the user
        raise HTTPException(
            status_code=500,
            detail=f"Something went wrong while processing that: {exc}",
        ) from exc

    if "__interrupt__" in result:
        interrupt_obj = result["__interrupt__"][0]
        return ChatResponse(
            thread_id=config["configurable"]["thread_id"],
            reply=_extract_interrupt_message(interrupt_obj),
            waiting_for_input=True,
            done=False,
        )

    return ChatResponse(
        thread_id=config["configurable"]["thread_id"],
        reply=_extract_final_message(result),
        waiting_for_input=False,
        done=True,
    )


@app.post("/api/chat/start", response_model=ChatResponse)
def start_chat(request: StartRequest) -> ChatResponse:
    """Begins a new conversation thread with the agent."""
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    return _run_graph({"user_message": request.message}, config)


@app.post("/api/chat/reply", response_model=ChatResponse)
def reply_chat(request: ReplyRequest) -> ChatResponse:
    """
    Resumes an existing (interrupted) conversation thread with the
    user's reply to a clarification / diagnostic / solution prompt.
    """
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    config = {"configurable": {"thread_id": request.thread_id}}
    return _run_graph(Command(resume=request.message), config)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


# Serve the chatbot frontend (frontend/index.html + assets) at "/".
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
