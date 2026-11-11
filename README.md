# CloudFlow AI Customer Support

CloudFlow is a fictional SaaS company used to demonstrate an end-to-end customer-support RAG application. The project combines a responsive company website, a local knowledge-base ingestion and retrieval pipeline, a Gemini-powered answer service, and a FastAPI endpoint consumed by the website chat widget.

## Features

- Six responsive company pages: Home, Products, Pricing, About Us, FAQ, and Contact.
- Company facts are written in the HTML pages and collected into page-specific text documents for indexing.
- Word-based chunking with configurable chunk size and overlap, plus source metadata for every chunk.
- Persistent ChromaDB vector store using Sentence Transformers embeddings.
- Semantic retrieval followed by hybrid reranking using semantic similarity and lexical query-term coverage.
- Grounded answers with source citations, a clear insufficient-information response, and short in-memory conversational context.
- A website chat panel that sends questions to FastAPI and keeps the latest six messages in the browser tab's `sessionStorage`.
- A mock API mode for checking the frontend-to-backend connection without initializing Gemini or the embedding model.

## Architecture

```text
Browser (static HTML/CSS/JavaScript)
  └── POST /api/chat ──> FastAPI
                           ├── validates question and conversation history
                           ├── retrieves 9 ChromaDB candidates
                           ├── hybrid-reranks candidates and selects the top 3
                           └── sends grounded context to Gemini 3.8 Flash
```

The website is served as static files. `script.js` builds the chat panel and sends `question` and `conversation_history` to `POST /api/chat`; the API returns an answer and the updated history. In normal mode, FastAPI loads the Sentence Transformers model, ChromaDB collection, and Gemini client once at startup. Each question performs fresh retrieval. Conversation history helps resolve follow-up references but is not a source of company facts.

The ingestion flow is:

1. `scraper/web_scraper.py` collects page text into `scraper/scraped_data/`.
2. `app/chunking.py` creates `data/chunks.json` with source file, page, and chunk index metadata.
3. `app/vector_store.py` embeds the chunks and synchronizes the persistent collection at `data/chroma_db/`.
4. `app/semantic_search.py` performs read-only local search; `app/rag_pipeline.py` reranks and generates cited answers.

## Technology

- **Frontend:** HTML, CSS, vanilla JavaScript
- **API:** Python, FastAPI, Uvicorn, Pydantic
- **Generation:** Google Gen AI SDK (`google-genai`) with `gemini-3.8-flash`
- **Embeddings:** Sentence Transformers, `all-MiniLM-L6-v2`, and NumPy
- **Vector storage:** persistent ChromaDB with cosine distance
- **Ingestion and scraping:** Python, Requests, Beautiful Soup
- **Evaluation:** Python `unittest` and a labeled local retrieval dataset

Python dependencies are listed in [`requirements.txt`](requirements.txt). The embedding model must be available from the Hugging Face Hub the first time it is loaded; later runs can use the local model cache.

## Setup

Use Python and a virtual environment from the project root:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The API expects the persistent vector database at `data/chroma_db/`. To rebuild the chunk file and vector store from the existing scraped text files, run:

```sh
python -m app.chunking
python -m app.vector_store
```

Building the vector store downloads or loads `sentence-transformers/all-MiniLM-L6-v2` and writes the persistent ChromaDB collection. The semantic search and retrieval evaluation require that collection and the model to be available locally.

## Configure the Gemini API key

The backend reads `GEMINI_API_KEY` from its process environment. Do not put the key in source code, `requirements.txt`, or committed configuration files. For a local shell session, enter it without echoing it and export it before starting the backend:

```sh
read -s GEMINI_API_KEY
export GEMINI_API_KEY
```

For deployment, provide the key through the hosting platform's secret manager. The API client is configured with a 60-second request timeout and automatic retries disabled. The RAG request uses `gemini-3.8-flash`, low thinking level, and a 512-token output limit.

## Run locally

Start the backend and website in separate terminals from the project root.

### 1. FastAPI backend on port 8001

Activate `.venv`, configure `GEMINI_API_KEY` in this terminal, then run:

```sh
.venv/bin/uvicorn app.api:app --host 127.0.0.1 --port 8001
```

The API endpoint is `http://127.0.0.1:8001/api/chat`. The backend's local CORS allowlist accepts the site at `http://localhost:8000` and `http://127.0.0.1:8000`.

### 2. Static website on port 8000

In a second terminal, run:

```sh
python3 -m http.server 8000 --bind 127.0.0.1
```

Open [http://localhost:8000](http://localhost:8000). The chat widget uses port 8001 for local API requests. On another host, set `window.CLOUDFLOW_API_URL` to the deployed API URL or route `/api/chat` through the same origin.

## Simulated API mode

Set `CLOUDFLOW_MOCK_LLM=1` when starting the backend to return a fixed simulated answer with an example FAQ source. This mode skips ChromaDB, the embedding model, and Gemini initialization; it verifies the website-to-API connection but does not generate AI answers.

```sh
CLOUDFLOW_MOCK_LLM=1 .venv/bin/uvicorn app.api:app --host 127.0.0.1 --port 8001
```

The mock mode does not require `GEMINI_API_KEY`. Stop the server and start it without that variable to use the real RAG pipeline.

## Tests

Run the automatically discovered test suite from the project root:

```sh
.venv/bin/python -m unittest discover -s tests -v
```

The API tests cover mock mode and the normal RAG path with a simulated Gemini client. The retrieval test uses the local embedding model and ChromaDB; it does not call Gemini. It requires `data/chroma_db/` and the embedding model to be present. To keep model loading offline when the model is already cached, set:

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m unittest discover -s tests -v
```

`test_gemini_latency.py` is a separate diagnostic script that sends real Gemini requests. It is not part of the safe test command above.

## Retrieval evaluation

Run the local labeled retrieval evaluation with:

```sh
.venv/bin/python -m evaluation.evaluate_retrieval
```

The current dataset contains five questions covering pricing, payment methods, the free trial, cancellation and refunds, and integrations. Evaluation retrieves 9 candidates, applies hybrid reranking, and checks the top 3 against expected source chunks.

| Metric | Current result |
|---|---:|
| Hit@3 | 1.00 |
| Recall@3 | 1.00 |
| MRR | 0.67 |

These are macro averages for the current five-case dataset, as documented in [`evaluation/RETRIEVAL_EVALUATION.md`](evaluation/RETRIEVAL_EVALUATION.md). The dataset is small, and the metrics measure retrieval only—not answer correctness, citation quality, abstention behavior, or generated response quality.

## Current limitations

- Gemini can return temporary HTTP 503 or 504 errors. Retries are disabled; the API reports a generic 502 response to the visitor and records safe diagnostic metadata in server logs.
- Mock mode returns a fixed test response and an example source. It does not invoke an AI model or validate answer quality.
- Retrieval quality is evaluated on only five labeled questions. Broader coverage and a separate evaluation of generated answers are still needed.
- The chatbot conversation history is temporary and limited to the current browser tab; it is not stored in a server-side account or database.

## Main project files

- `index.html`, `products.html`, `pricing.html`, `about.html`, `faq.html`, `contact.html` — company website pages and crawlable source content
- `styles.css`, `script.js` — shared visual system, responsive layout, and chat widget
- `company_knowledge.md` — approved reference for fictional company facts and policies
- `scraper/` — web scraper and extracted page text
- `app/chunking.py` — chunk creation
- `app/vector_store.py` — embedding generation and ChromaDB synchronization
- `app/semantic_search.py` — local semantic search
- `app/rag_pipeline.py` — hybrid reranking and Gemini answer generation
- `app/api.py` — FastAPI endpoint and optional mock mode
- `data/chunks.json`, `data/chroma_db/` — chunk records and persistent vector store
- `evaluation/` — retrieval labels, evaluator, and results report
- `tests/` — automatic API and retrieval tests
