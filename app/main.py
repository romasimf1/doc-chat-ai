"""doc-chat-ai — FastAPI backend.

Endpoints:
  GET  /api/health             - server status + whether an API key is set
  POST /api/documents/upload   - upload a document, chunk + embed it
  GET  /api/documents          - list uploaded documents
  DELETE /api/documents/{id}   - delete a document and its chunks
  POST /api/chat               - ask a question about the documents
  GET  /                        - the single-page frontend
"""

import os
import shutil

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app import config
from app.chunking import chunk_text
from app.extract import extract_text
from app.llm import LLMClient, build_chat_messages
from app.store import VectorStore

app = FastAPI(title="doc-chat-ai",
              description="Chat with your documents (RAG)")

# One shared store and one shared LLM client for the whole app.
# Tests replace app.state.llm with a fake client.
store = VectorStore()
app.state.llm = LLMClient()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(BASE_DIR, "static")


# -- request/response models ------------------------------------------
class ChatRequest(BaseModel):
    question: str
    top_k: int = config.DEFAULT_TOP_K


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict]


# -- health ------------------------------------------------------------
@app.get("/api/health")
def health() -> dict:
    """Status check. Never returns the API key itself, only a boolean."""
    return {
        "status": "ok",
        "api_key_configured": app.state.llm.is_configured,
        "chat_model": config.CHAT_MODEL,
        "embedding_model": config.EMBEDDING_MODEL,
        "documents": len(store.list_documents()),
    }


# -- documents ---------------------------------------------------------
@app.post("/api/documents/upload")
async def upload_document(file: UploadFile = File(...)) -> dict:
    """Upload one file, extract its text, chunk it, embed it, store it."""
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        text = extract_text(file.filename, content)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    chunks = chunk_text(text)
    if not chunks:
        raise HTTPException(
            status_code=400,
            detail="No readable text found in the uploaded file.")

    # Embeddings need an API key. Without one the document is still
    # stored (so upload + listing keep working), but its chunks have
    # no embeddings and cannot be searched until the file is
    # re-uploaded with a key configured. Chat refuses to answer
    # without a key, so no unsearchable chunks are ever used.
    if app.state.llm.is_configured:
        embeddings = app.state.llm.embed(chunks)
        warning = None
    else:
        embeddings = [[] for _ in chunks]
        warning = ("No OPENAI_API_KEY configured: document stored but "
                   "not embedded. Add the key to .env and re-upload "
                   "to make it searchable.")

    # Keep a copy of the original file on disk for reference.
    os.makedirs(config.UPLOAD_DIR, exist_ok=True)
    doc_id = store.add_document(file.filename, chunks, embeddings)
    saved_path = os.path.join(config.UPLOAD_DIR, f"{doc_id}_{file.filename}")
    with open(saved_path, "wb") as f:
        f.write(content)

    result = {"id": doc_id, "filename": file.filename,
              "chunks": len(chunks)}
    if warning:
        result["warning"] = warning
    return result


@app.get("/api/documents")
def list_documents() -> dict:
    return {"documents": store.list_documents()}


@app.delete("/api/documents/{doc_id}")
def delete_document(doc_id: str) -> dict:
    if not store.delete_document(doc_id):
        raise HTTPException(status_code=404, detail="Document not found.")
    # Remove the saved original file as well, if it exists.
    for name in os.listdir(config.UPLOAD_DIR) if os.path.exists(config.UPLOAD_DIR) else []:
        if name.startswith(doc_id + "_"):
            os.remove(os.path.join(config.UPLOAD_DIR, name))
    return {"deleted": doc_id}


# -- chat --------------------------------------------------------------
@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """Answer a question using only the uploaded documents (RAG)."""
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question is empty.")

    # No key, no answers: say so clearly instead of inventing a reply.
    if not app.state.llm.is_configured:
        raise HTTPException(
            status_code=503,
            detail="OPENAI_API_KEY is not configured. "
                   "Add your API key to .env to enable chat.")

    if not store.list_documents():
        raise HTTPException(
            status_code=400,
            detail="No documents uploaded yet. Upload a document first.")

    query_embedding = app.state.llm.embed([question])[0]
    top_chunks = store.search(query_embedding, top_k=request.top_k)
    messages = build_chat_messages(question, top_chunks)
    answer = app.state.llm.chat(messages)

    sources = [
        {"filename": chunk["filename"],
         "excerpt": chunk["text"][:300]}
        for chunk in top_chunks
    ]
    return ChatResponse(answer=answer, sources=sources)


# -- frontend ----------------------------------------------------------
@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
