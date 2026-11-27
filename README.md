# doc-chat-ai — Chat with your documents (RAG)

A small web app that lets you upload documents (PDF, TXT, Markdown, DOCX)
and ask questions about them in a chat UI. The app finds the most relevant
passages from your documents and asks a large language model to answer
**using only those passages** — if the answer is not in your documents,
it says so instead of inventing one.

This is a portfolio project: the whole RAG pipeline is written in plain,
readable Python with no heavy frameworks or vector-database dependencies.

## How it works (the RAG pipeline)

1. **Upload** — the text is extracted from the file (PDF, TXT, Markdown
   or DOCX) and saved on disk.
2. **Chunk** — the text is split into overlapping chunks of ~800 characters
   (150 characters of overlap, so context survives chunk borders).
3. **Embed** — each chunk is turned into a vector with OpenAI's
   `text-embedding-3-small` model.
4. **Retrieve** — when you ask a question, the question is embedded too,
   and the most similar chunks are found with cosine similarity.
5. **Answer** — the top chunks are sent to `gpt-4o-mini` together with a
   system prompt that says: answer **only** from this context, and say
   "I don't know" when the answer is not there. Sources are shown under
   every answer.

The vector store is a plain JSON file (`data/store.json`) with numpy
cosine similarity — no ChromaDB, no FAISS, no external service. That is
deliberate: the retrieval logic is ~40 lines and easy to explain.

## Stack

- Python 3.10+ — FastAPI + Uvicorn (backend)
- `openai` package — embeddings (`text-embedding-3-small`) and chat
  (`gpt-4o-mini` by default, configurable)
- `pypdf` — PDF text extraction, `python-docx` — DOCX extraction
- `numpy` — cosine similarity
- Plain HTML + JavaScript single-page frontend (no build step)

## API key — please read this first

**ChatGPT Plus does NOT include API access.** A Plus subscription only
works at chat.openai.com. This app talks to OpenAI's API, which is billed
separately on a pay-as-you-go basis — even though it uses the same OpenAI
account, API billing is separate from Plus.

To use this app you need an API key:

1. Go to https://platform.openai.com/api-keys (log in with your OpenAI
   account — the same one you use for ChatGPT Plus).
2. Create a new key and add a few dollars of credit under Billing.
   A few dollars is more than enough for testing: embedding a typical
   document costs a fraction of a cent, and a chat answer costs a few
   cents at most with `gpt-4o-mini`.
3. Put the key in your `.env` file as `OPENAI_API_KEY` (see setup below).

Without a key the server still starts, and uploads and the document list
still work — but documents are stored without embeddings, and chat
returns a clear error saying an API key is required. The app never fakes
answers.

## Setup

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure your API key
cp .env.example .env
# Edit .env and paste your key after OPENAI_API_KEY=

# 4. Start the server
uvicorn app.main:app --reload --port 8000
```

Open http://localhost:8000/ in your browser.

To run the tests:

```bash
pip install -r requirements-dev.txt
pytest -v
```

## API endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | Server status and whether an API key is configured (the key itself is never returned). |
| `POST` | `/api/documents/upload` | Upload one file (multipart `file` field). Returns the document id, filename and chunk count. |
| `GET` | `/api/documents` | List uploaded documents (id, filename, chunk count, upload time). |
| `DELETE` | `/api/documents/{id}` | Delete a document and all its chunks. |
| `POST` | `/api/chat` | Body: `{"question": "...", "top_k": 4}`. Returns `{"answer": "...", "sources": [{"filename": "...", "excerpt": "..."}]}`. |
| `GET` | `/` | The web UI. |

Example:

```bash
curl -F "file=@notes.pdf" http://localhost:8000/api/documents/upload
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the refund policy?"}'
```

## Project structure

```
doc-chat-ai/
├── app/
│   ├── main.py        # FastAPI app: endpoints, upload/chat flow
│   ├── config.py      # Settings from environment variables
│   ├── chunking.py    # Text splitting into overlapping chunks
│   ├── extract.py     # Text extraction from PDF/TXT/MD/DOCX
│   ├── store.py       # JSON vector store + cosine similarity search
│   └── llm.py         # OpenAI client wrapper + RAG prompt builder
├── static/
│   ├── index.html     # Single-page UI
│   └── app.js         # Upload, document list, chat
├── tests/             # pytest: chunking, retrieval ranking, prompt
├── requirements.txt
├── .env.example
└── README.md
```

## Limitations (honest notes)

- Embeddings are computed with the OpenAI API, so the app needs
  internet access and a funded API key for full functionality.
- The JSON store keeps all embeddings in memory on each request, which
  is fine for personal use (tens of documents) but not for large corpora.
- The model answers only from the retrieved chunks, so a question about
  a topic that is in the documents but was not retrieved may still get
  an "I don't know" — increasing `top_k` helps.
