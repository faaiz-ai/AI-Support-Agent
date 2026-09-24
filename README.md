# Agentic AI Technical Support Agent

An Agentic AI technical-support system built with **LangGraph, LangChain, Groq, and LangSmith**. The agent understands technical issues, classifies them, performs structured troubleshooting, tracks previous attempts, adapts when solutions fail, and escalates unresolved problems.

## Project Objective

The goal of this project is to build a stateful technical-support agent that can:

* Understand the user's technical problem
* Ask for missing information when required
* Classify problems into support categories
* Perform category-specific diagnosis
* Ask diagnostic questions and use the user's feedback
* Provide one troubleshooting solution at a time
* Track previously attempted solutions
* Avoid repeating failed solutions
* Generate alternative solutions when troubleshooting fails
* Support multiple problems in a single conversation
* Escalate when the issue cannot be resolved

## Example Interaction

### Docker GPU Troubleshooting

```text
User:
My Docker container cannot access the GPU.

Agent:
Was the container started with the `--gpus` flag (or equivalent)
to expose the GPU to the container?

User:
I'm not sure. How can I check?

Agent:
Please run `nvidia-smi` on the host and provide the output.

User:
nvidia-smi has failed because it couldn't communicate with
the NVIDIA driver.

Agent:
sudo apt install nvidia-utils-535

User:
nvidia-smi still fails.

Agent:
sudo apt install nvidia-driver-535
```

This demonstrates the agent's diagnostic process, solution attempts, and ability to generate a different troubleshooting approach after a failed solution.

## Screenshots

### Agent Conversation

The chatbot interface showing the Docker troubleshooting conversation, including diagnostic questions, user feedback, and alternative troubleshooting solutions.

![Agent Conversation](images/agent-conversation.png)

### LangSmith Trace

The LangSmith trace showing the corresponding LangGraph execution, including the parent graph, routing, Docker subgraph, diagnostic/troubleshooting nodes, and final response.

![LangSmith Trace](images/langsmith-trace.png)

## Core Workflow

```text
                    User Request
                         │
                         ▼
                  Understand Request
                         │
             ┌───────────┴───────────┐
             │                       │
        Needs Clarification      Can Classify
             │                       │
             ▼                       ▼
       Ask User for Info          Classify
                                     │
                 ┌───────────────────┼───────────────────┐
                 │                   │                   │
              Docker              Network          Application / Account
                 │                   │                   │
                 └───────────────────┴───────────────────┘
                                     │
                                     ▼
                           Category Subgraph
                                     │
                                     ▼
                               Diagnose Issue
                                     │
                                     ▼
                            Ask Diagnostic Question
                                     │
                                     ▼
                              Analyze Feedback
                                     │
                                     ▼
                           Generate One Solution
                                     │
                                     ▼
                              User Feedback
                                     │
                         ┌───────────┴───────────┐
                         │                       │
                      Solved                  Failed
                         │                       │
                         ▼                       ▼
                    Resolve Issue         Record Attempt
                                                 │
                                                 ▼
                                      Generate Different Solution
                                                 │
                                      ┌──────────┴──────────┐
                                      │                     │
                                   Solved                Still Failed
                                      │                     │
                                      ▼                     ▼
                                   Resolve              Escalate
```

## Agent Architecture

The system uses a **parent graph** to manage the overall support workflow and specialized subgraphs for different technical problem categories.

### Parent Graph

The parent graph is responsible for:

* Understanding the request
* Handling clarification
* Classifying the problem
* Managing the problem queue
* Routing to the appropriate support subgraph
* Maintaining shared state
* Handling multiple problems
* Returning the final response

### Specialized Subgraphs

The project currently supports four technical categories:

* **Docker**
* **Network**
* **Application**
* **Account**

Each category has its own troubleshooting workflow while sharing the overall support-state architecture.

## Key Agentic Behaviors

### Stateful Troubleshooting

The agent maintains state throughout the troubleshooting process instead of treating every message as an independent request.

### Diagnostic Questions

The agent can pause its workflow and ask the user for additional information before deciding on a solution.

### One Solution at a Time

The agent presents one troubleshooting action at a time and waits for the user's feedback before continuing.

### Failed Solution Tracking

Previously attempted solutions are stored in `attempted_solutions`.

This allows the agent to recognize failed attempts and avoid simply repeating the same solution.

### Feedback-Driven Troubleshooting

The agent uses user feedback to determine whether the current solution worked and whether additional troubleshooting is required.

### Alternative Solutions

When a solution fails, the agent can generate a different troubleshooting approach based on the updated state.

### Multi-Problem Support

The agent can identify and process multiple technical problems within a single conversation using a problem queue.

### Escalation

When troubleshooting cannot resolve an issue after the configured retry process, the agent can escalate the problem instead of continuing indefinitely.

## LangSmith Observability

**LangSmith** is used to trace and inspect the agent's execution.

The traces provide visibility into:

* Parent graph execution
* Request understanding
* Classification
* Problem routing
* Subgraph execution
* Diagnostic nodes
* Solution generation
* Feedback processing
* Retry behavior
* State transitions

This makes it possible to inspect how a user request moves through the LangGraph workflow rather than only observing the final chatbot response.

## Technology Stack

| Technology        | Purpose                                         |
| ----------------- | ----------------------------------------------- |
| Python            | Core development                                |
| LangGraph         | Stateful agent workflow and graph orchestration |
| LangChain         | LLM and agent components                        |
| Groq              | LLM inference                                   |
| LangSmith         | Tracing and observability                       |
| FastAPI           | Backend API                                     |
| HTML / JavaScript | Frontend chat interface                         |
| InMemorySaver     | LangGraph checkpointing during development      |

## Project Structure

```text
langgraph-project-updated/
│
├── app/
│   ├── api.py
│   ├── config.py
│   │
│   ├── graph/
│   │   ├── __init__.py
│   │   ├── parent_graph.py
│   │   ├── state.py
│   │   │
│   │   └── subgraphs/
│   │       ├── __init__.py
│   │       ├── account.py
│   │       ├── application.py
│   │       ├── docker.py
│   │       └── network.py
│   │
│   ├── nodes/
│   │   ├── __init__.py
│   │   ├── clarification.py
│   │   ├── classify.py
│   │   ├── general_answer.py
│   │   ├── shared.py
│   │   └── understand.py
│   │
│   └── schemas/
│       ├── __init__.py
│       └── outputs.py
│
├── chatbot.py
├── frontend/
│   └── index.html
├── images/
│   ├── agent-conversation.png
│   └── langsmith-trace.png
├── notebooks/
│   └── LangGraph_Project_Updated.ipynb
├── tests/
│   └── .gitkeep
├── .gitignore
├── README.md
├── requirements.txt
└── test.py
```

## Main Components

### `app/graph/state.py`

Defines the shared `SupportState` used throughout the parent graph and specialized subgraphs.

### `app/graph/parent_graph.py`

Controls the main workflow, including understanding, classification, problem queuing, routing, and subgraph execution.

### `app/graph/subgraphs/`

Contains the specialized troubleshooting workflows:

* `docker.py`
* `network.py`
* `application.py`
* `account.py`

### `app/nodes/`

Contains shared nodes responsible for understanding requests, classification, clarification, and general responses.

### `app/schemas/`

Contains structured output schemas used by the agent.

### `app/api.py`

Provides the FastAPI backend for communicating with the frontend and maintaining conversation threads.

### `frontend/index.html`

Provides the web-based chatbot interface.

## Setup

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd langgraph-project-updated
```

### 2. Create a Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=langgraph-support-agent
```

Do not commit `.env` to the repository.

### 5. Start the Backend

```bash
uvicorn app.api:app --reload --port 8000
```

Then open the frontend in your browser.

## Development Focus

This project focuses on building practical **Agentic AI workflows** rather than a simple question-answering chatbot.

The main areas explored include:

* LangGraph state management
* Conditional graph routing
* Specialized subgraphs
* Interrupt and resume workflows
* Stateful troubleshooting
* Feedback-driven decision making
* Retry and escalation logic
* Multi-problem handling
* LangSmith tracing and observability
* API integration
* Frontend integration

## Future Improvements

Potential future improvements include:

* Persistent production-grade checkpoint storage
* More technical support categories
* Improved diagnostic knowledge sources
* Retrieval-Augmented Generation (RAG)
* Authentication and user management
* Production deployment
* Expanded automated testing
* More advanced observability and evaluation

## Author

**Faaiz Nawaz**

Agentic AI & Machine Learning Engineer

Focused on **Agentic AI, LangGraph, LangChain, LangSmith, Computer Vision, and Machine Learning**.

