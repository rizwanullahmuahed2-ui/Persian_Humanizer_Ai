from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
import uvicorn

from humanizer_graph import run_humanizer, stream_humanizer
from model_provider import DEFAULT_MODEL, ModelId, available_models, validate_model
from rag_engine import get_or_build_vectorstore, rebuild_index, CHROMA_DIR
import os

app = FastAPI(title="Persian Humanizer API v2", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)


class HumanizeRequest(BaseModel):
    text: str
    style: str = "casual"
    model: ModelId = DEFAULT_MODEL


def validate_request(req):
    text = req.text.strip()
    if not text:
        raise HTTPException(400, "متن خالی است")
    if len(text) > 2000:
        raise HTTPException(400, "متن خیلی طولانی است (حداکثر ۲۰۰۰ کاراکتر)")
    try:
        validate_model(req.model)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from None
    return text


@app.on_event("startup")
async def startup():
    print("[API] Initializing vectorstore ...")
    try:
        get_or_build_vectorstore()
        print("[API] Vectorstore ready ✓")
    except Exception as e:
        print(f"[API] Vectorstore error: {e}")


@app.get("/health")
def health():
    vs_ready = os.path.exists(CHROMA_DIR) and bool(os.listdir(CHROMA_DIR))
    return {
        "status": "ok",
        "vectorstore_ready": vs_ready,
        "version": "2.0",
        "available_models": available_models(),
        "models": {
            "humanizer": "llama3.1:8b",
            "evaluator": "rule-based (no LLM)",
            "embedding": "paraphrase-multilingual-MiniLM-L12-v2",
            "reranker": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        },
    }


@app.post("/humanize")
def humanize(req: HumanizeRequest):
    text = validate_request(req)
    result = run_humanizer(text, req.style, req.model)
    return {
        "original": text,
        "humanized": result.get("humanized_text", ""),
        "style": req.style,
        "model": req.model,
        "evaluation": result.get("evaluation", {}),
        "examples_used": len(result.get("retrieved_examples", [])),
        "error": result.get("error"),
    }


@app.post("/humanize/stream")
def humanize_stream(req: HumanizeRequest):
    """SSE endpoint — frontend receives tokens live."""
    text = validate_request(req)

    def generator():
        try:
            for chunk in stream_humanizer(text, req.style, req.model):
                yield chunk
        except Exception as e:
            yield f"data: __ERROR__{str(e)}\n\n"

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/rebuild-index")
def rebuild():
    try:
        stats = rebuild_index()
        return {"status": "ok", **stats}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/")
def root():
    return {"message": "Persian Humanizer API v2", "docs": "/docs"}


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False, log_level="info")
