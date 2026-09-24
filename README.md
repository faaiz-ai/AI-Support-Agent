# Agentic AI Technical Support Agent

An **Agentic AI technical-support system** built with **LangGraph** that diagnoses technical issues, maintains conversation state, adapts troubleshooting based on user feedback, avoids repeating failed solutions, handles multiple problems, and escalates unresolved issues.

The project is designed to demonstrate how **stateful Agentic AI workflows** can be built using graph-based orchestration rather than a simple question-answering chatbot.

## 🚀 What This Agent Does

The agent follows a structured troubleshooting workflow:

* Understands the user's technical problem
* Requests missing information when necessary
* Classifies the problem into a support category
* Routes the issue to a specialized troubleshooting subgraph
* Asks diagnostic questions before suggesting solutions
* Provides **one troubleshooting action at a time**
* Waits for user feedback before continuing
* Tracks previously attempted solutions
* Avoids repeating failed solutions
* Generates alternative troubleshooting approaches
* Handles multiple technical problems in one conversation
* Escalates when an issue remains unresolved

## 🧠 Why This Is Agentic AI

This project goes beyond a basic chatbot that generates an answer and stops.

The agent maintains state throughout the troubleshooting process and uses the outcome of previous actions to determine what to do next.

For example:

```text
User Problem
     ↓
Understand Request
     ↓
Classify Problem
     ↓
Route to Specialized Subgraph
     ↓
Diagnose
     ↓
Ask Diagnostic Question
     ↓
Generate One Solution
     ↓
User Feedback
     ↓
┌───────────────┴───────────────┐
│                               │
Solved                         Failed
│                               │
↓                               ↓
Resolve                  Record Attempt
                                ↓
                         Generate Different
                            Solution
                                ↓
                         ┌──────┴──────┐
                         │             │
                      Solved        Still Failed
                         │             │
                         ↓             ↓
                      Resolve       Escalate
```

This workflow demonstrates:

* Stateful decision making
* Conditional routing
* Feedback-driven execution
* Retry logic
* Memory of previous attempts
* Human-in-the-loop interaction
* Specialized subgraphs
* Escalation logic

## 🏗️ Architecture

The system uses a **parent graph** to manage the overall support workflow and specialized subgraphs for different technical problem categories.

### Parent Graph

The parent graph handles:

* Request understanding
* Clarification
* Problem classification
* Problem queue management
* Routing
* Shared state
* Multi-problem conversations
* Final response handling

### Specialized Troubleshooting Subgraphs

The current system contains four specialized workflows:

| Category    | Purpose                                      |
| ----------- | -------------------------------------------- |
| Docker      | Container and Docker-related troubleshooting |
| Network     | Connectivity and networking troubleshooting  |
| Application | Application/backend troubleshooting          |
| Account     | Account-related troubleshooting              |

Each subgraph has its own diagnosis and solution workflow while operating on the shared support state.

## 🔄 Example: Docker GPU Troubleshooting

A user might report:

> My Docker container cannot access the GPU.

Instead of immediately providing a large list of commands, the agent investigates the problem step by step.

Example workflow:

```text
User:
My Docker container cannot access the GPU.

Agent:
Was the container started with GPU access enabled?

User:
I'm not sure.

Agent:
Please check the host with nvidia-smi and provide the result.

User:
nvidia-smi cannot communicate with the NVIDIA driver.

Agent:
Try the following troubleshooting action:
sudo apt install nvidia-utils-535

User:
It still doesn't work.

Agent:
The previous solution failed.
Let's try a different troubleshooting approach.
```

The important behavior is that the agent **records the failed attempt and changes its troubleshooting strategy** rather than simply repeating the same recommendation.

## 🔍 LangSmith Observability

The project uses **LangSmith** to inspect the execution of the LangGraph workflow.

The traces provide visibility into:

* Request understanding
* Classification
* Conditional routing
* Problem queue handling
* Subgraph execution
* Diagnostic nodes
* Solution generation
* Feedback processing
* Retry behavior
* State transitions

This makes it possible to inspect **how the agent reached a response**, rather than only looking at the final chatbot message.

### Agent Conversation

![Agent Conversation](app/images/agent_conversation%281%29.png)

### LangSmith Trace

![LangSmith Trace](app/images/langsmith_trace%281%29.png)

Additional screenshots are available in [`app/images/`](app/images/).

## 🛠️ Technology Stack

| Technology        | Role                                      |
| ----------------- | ----------------------------------------- |
| Python            | Core development                          |
| LangGraph         | Stateful workflow orchestration           |
| LangChain         | LLM integration and structured components |
| Groq              | LLM inference                             |
| LangSmith         | Tracing and observability                 |
| FastAPI           | Backend API                               |
| HTML / JavaScript | Web-based frontend                        |
| InMemorySaver     | Development-time LangGraph checkpointing  |

## 📁 Project Structure

```text
AI-Support-Agent/
│
├── app/
│   ├── api.py
│   ├── config.py
│   │
│   ├── graph/
│   │   ├── parent_graph.py
│   │   ├── state.py
│   │   │
│   │   └── subgraphs/
│   │       ├── account.py
│   │       ├── application.py
│   │       ├── docker.py
│   │       └── network.py
│   │
│   ├── nodes/
│   │   ├── clarification.py
│   │   ├── classify.py
│   │   ├── general_answer.py
│   │   ├── shared.py
│   │   └── understand.py
│   │
│   └── schemas/
│       └── outputs.py
│
├── frontend/
│   └── index.html
│
├── app/images/
│   ├── agent_conversation(1).png
│   ├── agent_conversation(2).png
│   ├── langsmith_trace(1).png
│   ├── langsmith_trace(2).png
│   └── langsmith_trace(3).png
│
├── chatbot.py
├── requirements.txt
├── README.md
└── .gitignore
```

## ⚙️ Setup

### 1. Clone the repository

```bash
git clone https://github.com/faaiz-ai/AI-Support-Agent.git
cd AI-Support-Agent
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=langgraph-support-agent
```

Do **not** commit `.env` to the repository.

### 5. Start the application

```bash
uvicorn app.api:app --reload --port 8000
```

Then open the frontend interface in your browser.

## 🎯 Key Concepts Demonstrated

This project focuses on practical Agentic AI development and explores:

* LangGraph state management
* Graph-based workflow orchestration
* Conditional routing
* Specialized subgraphs
* Interrupt and resume workflows
* Human-in-the-loop interaction
* Stateful troubleshooting
* Feedback-driven decisions
* Failed-solution tracking
* Retry and alternative-solution logic
* Multi-problem handling
* Escalation
* LangSmith tracing
* FastAPI integration
* Frontend integration

## 🔮 Future Improvements

Potential next steps include:

* Persistent production-grade checkpoint storage
* Additional technical support categories
* Retrieval-Augmented Generation (RAG)
* Authentication and user management
* Production deployment
* Expanded automated testing
* More advanced agent evaluation
* Improved observability

## 👨‍💻 Author

**Faaiz Nawaz**

Machine Learning & Agentic AI Engineer

Focused on:

**Agentic AI · LangGraph · LangChain · LangSmith · Computer Vision · Machine Learning**

---

⭐ If you find the project useful, feel free to explore the implementation and LangGraph workflow.


