#  Finance RAG Chatbot

An advanced **RAG (Retrieval-Augmented Generation)** chatbot designed to analyze complex financial documents (PDFs). 

Unlike basic RAG pipelines, this agent implements **Memory-Aware Query Rewriting**, **Cross-Encoder Reranking**, and **Intent Guardrails** to ensure high-precision answers and a natural conversational experience.

---

##  Key Features

* **Smart Context Memory:**
    * Understands pronouns ("What is *its* revenue?" → "What is *JPMorgan's* revenue?").
    * Handles topic switching naturally ("What about BlackRock?" → Switches context instantly).
* **Guardrails:**
    * A dedicated Router classifies queries *before* searching.
    * Blocks off-topic questions (e.g., cooking, movies) and handles greetings instantly without wasting compute on retrieval.
* **High-Precision Retrieval:**
    * **Hybrid Search:** Retrieves the top 20 candidates using Vector Similarity.
    * **Reranking:** Uses a **Cross-Encoder (MS-Marco)** to grade those 20 chunks and select only the Top 3 "Gold Standard" documents for the final answer.
* **Transparent Sources:**
    * Every response cites the specific PDF filename and page number.
    * Maintains source history in the chat UI for verification.

---

##  Architecture

The system follows a **Clean Architecture** pipeline:

1.  **Ingestion:** PDFs are loaded via **`PyPDFLoader`**, split into chunks (1200 characters with 300 overlap), and embedded into **ChromaDB**.
2.  **Router (Guardrail):** The LLM checks if the user input is *Greeting*, *Off-Topic*, or *Finance*.
3.  **Rewriter (Memory):** If *Finance*, the LLM rewrites the query to resolve pronouns based on chat history.
4.  **Retrieval & Reranking:**
    * *Step A:* Fetch top 20 chunks via `all-MiniLM-L6-v2`.
    * *Step B:* Re-order chunks via `cross-encoder/ms-marco-MiniLM-L-6-v2`.
5.  **Generation:** The **Mistral** model generates an answer using only the reranked evidence.

---

## Tech Stack

* **LLM Engine:** [Ollama](https://ollama.com/) (Running `mistral` locally)
* **Orchestration:** LangChain
* **Vector Database:** ChromaDB
* **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`
* **Reranker:** `cross-encoder/ms-marco-MiniLM-L-6-v2`
* **UI:** Streamlit

---

## Installation & Setup

### 1. Prerequisites
* Python 3.10+
* [Ollama](https://ollama.com/) installed and running.

### 2. Clone & Install
```bash
git clone https://github.com/jetmirterziu/rag-chatbot.git
cd rag-chatbot
pip install -r requirements.txt
```

### 3. Setup Local LLM
* Pull the Mistral model (required for reasoning):
```bash
ollama pull mistral
```

### 4. Prepare Data
* Drop your financial PDFs (e.g., Annual Reports) into the `data/` folder. (The folder is included in the repo, or you can add your own files).

## Usage

### 1. Ingest Documents
* Process the PDFs and build the vector index:
```bash
python rag/ingest.py
```

### 2. Run the Chatbot
* Launch the Streamlit interface:
```bash
streamlit run app/main.py
```
<br>

***Author***<br>
*Jetmir Terziu*