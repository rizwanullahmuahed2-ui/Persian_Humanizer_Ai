"""HTTP boundary for local rewriting; model dependencies load on first use."""

import logging
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict

from model_provider import DEFAULT_MODEL, ModelId, available_models, validate_model

ROOT = Path(__file__).resolve().parents[1]
logger = logging.getLogger(__name__)
app = FastAPI(title="Persian Humanizer", version="2.1.0")
origins = os.getenv(
    "HUMANIZER_CORS_ORIGINS", "http://127.0.0.1:5500,http://localhost:5500"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins.split(","),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class HumanizeRequest(BaseModel):
    # Older clients may still send style; it cannot change the single policy.
    model_config = ConfigDict(extra="ignore")

    text: str
    model: ModelId = DEFAULT_MODEL


def validate_request(req):
    text = req.text.strip()
    if not text:
        raise HTTPException(400, "متن خالی است")
    if len(text) > 2000:
        raise HTTPException(400, "حداکثر متن ۲۰۰۰ کاراکتر است")
    try:
        validate_model(req.model)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from None
    return text


def run_humanizer(*args):
    from humanizer_graph import run_humanizer as run

    return run(*args)


def stream_humanizer(*args):
    from humanizer_graph import stream_humanizer as stream

    yield from stream(*args)


@app.get("/health")
def health():
    name = "chroma_db_train" if os.getenv("USE_TRAIN_SET") == "1" else "chroma_db_v2"
    index = ROOT / "data" / name
    return {
        "status": "ok",
        "version": "2.1.0",
        "vectorstore_ready": index.is_dir() and any(index.iterdir()),
        "available_models": available_models(),
        "readiness_note": "Configured does not verify provider connectivity or model availability.",
        "models": {
            "humanizer": DEFAULT_MODEL,
            "evaluator": "rule-based (no LLM)",
            "embedding": os.getenv(
                "EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2"
            ),
            "reranker": os.getenv(
                "RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
            ),
        },
    }


@app.post("/humanize")
def humanize(req: HumanizeRequest):
    text = validate_request(req)
    try:
        result = run_humanizer(text, req.model)
        if result.get("error") or not result.get("humanized_text", "").strip():
            raise RuntimeError("Generation did not complete")
    except Exception as exc:
        logger.warning("Generation failed: %s", type(exc).__name__)
        raise HTTPException(
            502, "بازنویسی تکمیل نشد. اتصال مدل و تنظیمات سرور را بررسی کنید."
        ) from None
    return {
        "original": text,
        "humanized": result["humanized_text"],
        "model": req.model,
        "evaluation": result.get("evaluation", {}),
        "examples_used": len(result.get("retrieved_examples", [])),
        "error": None,
    }


@app.post("/humanize/stream")
def humanize_stream(req: HumanizeRequest):
    text = validate_request(req)

    def generator():
        try:
            yield from stream_humanizer(text, req.model)
        except Exception as exc:
            logger.warning("Streaming failed: %s", type(exc).__name__)
            yield "data: __ERROR__بازنویسی تکمیل نشد. اتصال مدل و تنظیمات سرور را بررسی کنید.\n\n"

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )


@app.post("/rebuild-index")
def rebuild():
    if os.getenv("HUMANIZER_ENABLE_REBUILD") != "1":
        raise HTTPException(
            403, "Index rebuilding is disabled. Enable it only for local maintenance."
        )
    try:
        from rag_engine import rebuild_index

        return {"status": "ok", **rebuild_index()}
    except Exception as exc:
        logger.warning("Index rebuild failed: %s", type(exc).__name__)
        raise HTTPException(
            500, "Index rebuilding failed. Check the dataset and embedding setup."
        ) from None


@app.get("/")
def root():
    return {"message": "Persian Humanizer API", "docs": "/docs", "interface": "/ui/"}


app.mount("/ui", StaticFiles(directory=ROOT / "frontrend", html=True), name="interface")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="127.0.0.1", port=8000)
