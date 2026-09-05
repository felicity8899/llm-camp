# 🎓 CFA 2 Study Assistant: Multi-Subject RAG & LangGraph Agent

Welcome to the **CFA 2 Prep Study Assistant** repository! This project is a complete, production-grade educational software system designed to help CFA Level II candidates study complex concepts, practice tricky quantitative questions, and receive detailed, step-by-step mathematical and ethical explanations grounded in official curriculum materials.

We have designed and evaluated two distinct AI architectures side-by-side: a **Standard Retrieval-Augmented Generation (RAG) Chain** and a **multi-step agentic workflow powered by LangGraph**.

-----

## 📌 1. Problem Description & Solution

### The Challenge
The Chartered Financial Analyst (CFA) Level II exam is globally notorious for its extreme difficulty. It spans 9 dense subject areas (Ethics, Quantitative Methods, Economics, Financial Statement Analysis, Corporate Issuers, Equity, Fixed Income, Derivatives, and Portfolio Management) and tests candidates using two major formats:
1.  **Multi-Step Financial Calculations:** Candidates are frequently tripped up by highly specific formulas (e.g., autoregressive AR(1) models, random walks, forward rate agreements, binomial trees) where knowing the final answer is useless without a clear, step-by-step mathematical derivation.
2.  **Highly Nuanced Ethical Rules:** Standard I-VII of the CFA Institute Code of Ethics require candidates to evaluate complex professional scenarios where a tiny contextual detail completely changes whether an ethics violation has occurred.

### The Solution
Most student assistants or vanilla chatbots fail because they hallucinate formulas, give generic "textbook definition summaries" rather than answering the student's exact problem, or get confused by professional nuances. 

Our system solves this by implementing a **Configuration-Driven, Production RAG & Multi-Agent Assistant**. The system:
*   Extracts verified curriculum grounding from local databases.
*   Enforces strict mathematical derivations.
*   Features an active online AI Judge to guarantee answer quality.
*   Allows administrators to fine-tune system behaviors via a centralized `config.yaml` without changing a single line of Python code.

-----

## 📊 2. Dataset & Data Ingestion Pipeline

The system is grounded on a high-density dataset representing actual CFA curriculum practice questions, answers, and detailed rationales.

### Data Anatomy
Questions are stored in a structured JSON schema:
*   `module`: Topic area (e.g., *Fixed Income - Practice Problems 3*)
*   `question_text`: Complete problem statement along with multiple-choice options (A, B, C).
*   `correct_answer`: The verified correct option choice.
*   `explanation`: The official step-by-step mathematical derivation and standard reference.

### Ingestion Flow
1.  **Semantic Chunking:** Text documents are split into optimal semantic chunks preserving math formulas and contextual integrity.
2.  **Vector Embedding:** We utilize the high-performance OpenAI `text-embedding-3-small` model to generate semantic representations.
3.  **Vector Database:** Chunks are indexed and persisted in a **Chroma vector database** (`./chroma_db`) for fast, low-latency semantic retrieval.

-----

## ⚙️ 3. Architecture & System Flow

Our codebase supports two distinct execution frontends and two distinct backends, allowing developers to switch fluidly between development, testing, and production states:

### 🎮 The Frontends
1.  **Streamlit Web Interface & Analytics Dashboard (`app.py`)**: A fully realized, interactive web application featuring a clean conversational interface for candidates and a real-time, interactive performance tracking dashboard for system administrators.
2.  **Interactive Terminal CLI Chat (`chat_parameterized_v3.py`)**: A lightweight, developer-focused console program displaying real-time metrics and color-coded evaluation statuses directly in the terminal.

### 🧠 The Backends
*   **Pipeline A: Standard RAG (LangChain Chain)**:
    ```
    [User Query] ──► [Chroma DB (k Chunks)] ──► [RAG System Prompt] ──► [GPT-4o / gpt-4o-mini] ──► [Structured Answer]
    ```
    *Mechanism*: Single-shot retrieval. It fetches the top $k$ chunks from the database, builds the prompt context, and executes the LLM in one pass. It is fast, cost-efficient, and ideal for straightforward queries.

*   **Pipeline B: Tool-Calling Agentic Workflow (LangGraph)**:
    ```
                         ┌─────────────── Repeat if needed ──────────────┐
                     ▼                                               │
    [User Query] ──► [LangGraph Agent] ──► [Decides Action] ──► [cfa_knowledge_search tool] ──► [Final Synthesis]
    ```
    *Mechanism*: Multi-step reasoning. Built with **LangGraph**, the agent is given an active search tool. It can search the vector store multiple times with different keywords, self-correct if the first search returns insufficient math data, and aggregate multiple sources before outputting the final answer.

-----

## 🧬 4. Configuration-Driven Design (`config.yaml`)

Both pipelines are fully parameterized. Developers can control the system dynamically by editing a single configuration file.

```yaml
# CFA Prep Assistant - Production Parameters Configuration

retriever:
  search_type: "mmr"             # Retrieval strategy: "similarity" (standard search) or "mmr" (Maximal Marginal Relevance to avoid redundant context)
  k: 4                           # Number of top document chunks to retrieve and feed into the LLM context
  embedding_model: "text-embedding-3-small" # Embedding model used for indexing and vector search
  persist_directory: "./chroma_db" # Path to the persistent Chroma vector database

llm:
  model_name: "gpt-4o"           # The main language model with strong reasoning and calculation capabilities
  temperature: 0.0               # Kept at 0.0 to ensure deterministic, strict, and precise answers for financial questions

system:
  engine_type: "Standard RAG Chain"  # Active execution engine: "Standard RAG Chain" or "LangGraph Tool Agent"
  sample_size: 20                # Default test dataset sample size used during pipeline evaluation runs
```

-----

## 🧪 5. Rigorous Evaluation Framework

A primary objective of this repository is quantitative validation. We do not guess which pipeline works better; we measure it using a parameters-based grid sweep pipeline (`evaluate.py`).

### 📐 Evaluation Criteria
Reviewers can find our evaluation metrics and logic inside `evaluate.py`:
1.  **Retrieval Hit Rate (HR):** Measures if the correct reference document containing the ground-truth explanation is within the retrieved top-$k$ results.
2.  **Mean Reciprocal Rank (MRR):** Measures the precision of retrieval, rewarding the system when the most relevant chunk is retrieved at Rank 1.
3.  **Accuracy / Verification Rate (LLM-as-a-judge):** A strict evaluator model grades both outputs against the gold standard `Expected Knowledge`. Our code avoids fragile text matching by implementing a robust parsing structure that extracts clean grading verdicts.

### 📊 Automation Sweep Script (`run_grid_search.sh`)
You can run automated grid sweeps over different $K$ values and retrieval algorithms:
```bash
./run_grid_search.sh
```
This automatically tests multiple combinations (e.g., $K=3$ similarity vs $K=5$ MMR), writes stdout logs to `logs/`, and saves final quantitative JSON experiment reports directly into the `report/` directory.

### 📝 Single Evaluation Run & JSON Report Generation
You can also run a single evaluation experiment manually. For example, executing:
```bash
uv run python evaluate.py --sample-size 10 --k 3 --search-type mmr
```
This command will dynamically run the evaluation on a random sample of 10 questions using the **MMR (Maximal Marginal Relevance)** retrieval strategy with $K=3$. 

Upon completion, the pipeline automatically compiles the metrics and **generates a structured JSON report file directly inside the `report/` folder** (e.g., `report/eval_report_gpt-4o-mini_mmr_k3.json`).

This JSON report preserves the exact parameters and quantitative metrics of the run:
```json
{
    "parameters": {
        "model_name": "gpt-4o-mini",
        "embedding_model": "text-embedding-3-small",
        "search_type": "mmr",
        "k": 3,
        "temperature": 0.0,
        "sample_size": 10
    },
    "metrics": {
        "hit_rate": 90.0,
        "mrr": 0.783,
        "rag_accuracy": 70.0,
        "agent_accuracy": 90.0
    }
}
```

-----

## 🖥️ 6. App Walkthrough & Outputs

### 📊 Streamlit Web Application & Performance Dashboard (`app.py`)
For production deployments, the project provides a comprehensive **Streamlit Web Application** featuring a two-part layout:

1.  **CFA Study Assistant Chat Panel**:
    *   Candidates can submit free-form questions or paste practice problems.
    *   Explanations are displayed in high-fidelity markdown (complete with LaTeX mathematical formatting).
    *   An interactive **Thumbs Up (👍) / Thumbs Down (👎)** component allows candidates to rate answer quality, which directly records human feedback.
2.  **Real-Time Analytics & Cost Dashboard**:
    *   Built on top of **Pandas and SQLite (`metrics.db`)**, this panel queries real-time telemetry metrics.
    *   **Cost Tracking Graphs**: Visualizes cumulative API spend in USD over time, warning administrators of resource depletion.
    *   **Latency Trends**: Line charts plotting response latency in seconds, helping developers spot slow database queries or LLM bottlenecks.
    *   **Token Breakdown Metrics**: Stacked charts showing Prompt vs. Completion token ratios.
    *   **AI Judge vs. Human Alignment**: Bar charts comparing the automated Online AI Judge pass rates against actual candidate feedback scores, exposing RAG regressions immediately.

```text
+---------------------------------------------------------------------------------------------------+
| 🎓 CFA Prep AI Study Assistant Web UI                                            [Streamlit App]  |
+---------------------------------------------------------------------------------------------------+
|  [ Chat Interface ]                                     |  [ Live Metrics & Cost Dashboard ]      |
|                                                         |                                         |
|   Candidate: Why is leverage threshold 3(k+1)/n?        |   💸 Cumulative Spend: $14.28 (OpenAI)  |
|                                                         |   ⏱ Average Latency: 2.15s (Chroma DB) |
|   Assistant:                                            |                                         |
|   According to the curriculum, the leverage...          |   📈 TOKEN USAGE TRENDS                 |
|   1. Average leverage = (k + 1) / n                     |      ██████ Prompt Tokens (82%)         |
|   2. High leverage threshold = 3 * (k + 1) / n          |      ███ Completion Tokens (18%)        |
|                                                         |                                         |
|   Was this helpful? [ 👍 Thumbs Up ] [ 👎 Thumbs Down ] |   🤖 AI Judge Pass Rate: 94.2%          |
|                                                         |   👤 User Satisfaction Rate: 91.5%     |
+---------------------------------------------------------------------------------------------------+
```

---

### 💻 Live ANSI Console Dashboard
When launching the CLI assistant (`chat.py`), the console renders a real-time operational dashboard with color-coded status lights:
*   🟩 `✅ AI-Pass` (Bright Green): The automated judge determined the answer is perfectly grounded in the retrieved curriculum.
*   🟨 `⚠ AI-Review` (Bright Yellow): The response did not fully answer the prompt or relied on ungrounded knowledge, triggering a review flag.

```text
============================================================
🤖 Initializing CFA Level II CLI Assistant...

📊 CURRENT ACTIVE PARAMETERS (from config.yaml):
 - LLM Engine       : gpt-4o
 - Active Pipeline  : Standard RAG Chain
 - Search Strategy  : mmr (k=4)
============================================================
✅ Assistant is ready! Type your question below (type 'exit' or 'quit' to stop).

You: explain why leverage threshold is 3(k+1)/n in regression?

Assistant:
According to the retrieved curriculum on Quantitative Methods, leverage measures the 
distance of an independent variable value from its mean. 
For a multiple linear regression with k independent variables and n observations:
1. The average leverage value is (k + 1) / n.
2. An outlier in the independent variables (high leverage point) is identified 
   if its leverage exceeds three times the average leverage.
3. Therefore, the threshold is calculated as: 3 * (k + 1) / n.

[Metrics & AI Judge: ✅ AI-Pass | ⏱ 2.14s | 🪙 Tokens: 1045 | 💰 $0.005225]

Rate this answer [Enter to skip | type '1' for 👍 | type '-1' for 👎]: 1
✅ Human feedback recorded (👍 Thumbs Up)!
------------------------------------------------------------
```

### SQLite Metrics & Feedback Logging
Every question, answer, latency metric, token cost, automated AI verdict, and user thumbs-up/down is logged directly into a local SQLite database (`metrics.db`) across two tables:
*   `llm_metrics`: Tracks operational performance (tokens, latency, cost).
*   `feedback`: Collects user satisfaction ratings and detailed judge critiques.

-----

## 🚀 7. Installation & Quick Start

This project is built for portability, utilizing **Docker** and the lightning-fast Python package manager **`uv`**.

### Environment Variables (.env)
Create a `.env` file in the root directory:
```env
OPENAI_API_KEY=your-actual-openai-api-key-here
```

### Option A: Local Execution with `uv` (Recommended)
`uv` is extremely fast and handles environment synchronization automatically.
1.  **Install `uv` (if needed):**
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```
2.  **Sync Dependencies & Launch Environment:**
    ```bash
    uv sync
    ```
3.  **Run the Streamlit Chat & Analytics Dashboard (Web App):**
    ```bash
    uv run streamlit run app.py
    ```
4.  **Run the Interactive CLI Chat (with live SQLite Logging and color codes):**
    ```bash
    uv run python chat.py
    ```
5.  **Run the Quantitative Evaluation Pipeline:**
    ```bash
    uv run python evaluate.py --sample-size 10 --k 3 --search-type mmr
    ```

### Option B: Quick Deployment with Docker
For fully isolated, cross-platform execution:
1.  **Build and launch the container:**
    ```bash
    docker-compose up --build -d
    ```
2.  **Access the Streamlit Dashboard Web App:**
    Open your browser and navigate to `http://localhost:8501`.
3.  **Execute Interactive Chat within Container:**
    ```bash
    docker compose exec cfa-assistant python chat.py
    ```

-----

## 🧹 8. Codebase Standards & Reproducibility

*   **Zero Notebook Trash:** No messy draft `.ipynb` or `.ipynb_checkpoints` files are left in the repository.
*   **Safe Dynamic Invocation:** Prompts are designed defensively to prevent state corruption across multi-turn queries.
*   **Idempotent Directory Creation:** Automatic creation of `report/` and `logs/` directories ensures that the software never crashes due to missing folders on a fresh clone.
