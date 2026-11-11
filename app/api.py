"""FastAPI entry point for the CloudFlow RAG customer-support API."""

from __future__ import annotations

import logging
import os
import re
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

from app.rag_pipeline import (
    answer_question,
    create_gemini_client,
    get_gemini_model_name,
    update_conversation_history,
)
from app.semantic_search import load_search_context


logger = logging.getLogger(__name__)
MAX_CONVERSATION_MESSAGES = 6
MAX_QUESTION_LENGTH = 4_000
MAX_MESSAGE_LENGTH = 4_000
MOCK_MODE_ENV_VAR = "CLOUDFLOW_MOCK_LLM"
MOCK_ANSWER = (
    "Risposta simulata: il collegamento del chatbot è attivo.\n\n"
    "Fonte di esempio: FAQ — faq.txt"
)
_SAFE_PROVIDER_STATUS = re.compile(r"[A-Za-z0-9_.-]{1,64}\Z")


def _provider_error_fields(exception: Exception) -> tuple[int | None, str | None]:
    """Extract only bounded HTTP/status metadata from a wrapped provider error."""
    current: BaseException | None = exception
    seen: set[int] = set()
    for _ in range(5):
        if current is None or id(current) in seen:
            break
        seen.add(id(current))
        code = getattr(current, "code", None)
        status = getattr(current, "status", None)
        safe_code = code if isinstance(code, int) and not isinstance(code, bool) else None
        safe_status = (
            status
            if isinstance(status, str) and _SAFE_PROVIDER_STATUS.fullmatch(status)
            else None
        )
        if safe_code is not None or safe_status is not None:
            return safe_code, safe_status
        current = current.__cause__ or current.__context__
    return None, None


class ChatMessage(BaseModel):
    """One validated message supplied as conversational context."""

    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=MAX_MESSAGE_LENGTH)


class ChatRequest(BaseModel):
    """Payload accepted by ``POST /api/chat``."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=MAX_QUESTION_LENGTH)
    conversation_history: list[ChatMessage] = Field(default_factory=list)


class ChatResponse(BaseModel):
    """RAG answer and the updated, bounded conversation history."""

    answer: str
    conversation_history: list[ChatMessage]


def _validated_history(messages: list[ChatMessage]) -> list[dict[str, str]]:
    """Validate role alternation, then retain only the most recent six messages."""
    history = [message.model_dump() for message in messages]
    if history and history[0]["role"] != "user":
        raise HTTPException(
            status_code=422,
            detail="La cronologia deve iniziare con un messaggio dell'utente.",
        )
    previous_role: str | None = None
    for message in history:
        if not message["content"].strip():
            raise HTTPException(
                status_code=422,
                detail="I messaggi della cronologia non possono essere vuoti.",
            )
        if message["role"] == previous_role:
            raise HTTPException(
                status_code=422,
                detail="I ruoli nella cronologia devono alternarsi.",
            )
        previous_role = message["role"]
    return history[-MAX_CONVERSATION_MESSAGES:]


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Load the embedding model, Chroma collection, and Gemini client once."""
    application.state.gemini_model_name = get_gemini_model_name()
    if os.environ.get(MOCK_MODE_ENV_VAR) == "1":
        application.state.mock_mode = True
        application.state.search_context = None
        application.state.gemini_client = None
        logger.info("CloudFlow API mock LLM mode enabled")
        yield
        return

    application.state.mock_mode = False
    try:
        application.state.search_context = load_search_context()
        application.state.gemini_client = create_gemini_client()
    except Exception as exc:
        # Log only the exception type: configuration values and provider details
        # must not be returned to callers or accidentally included in logs.
        logger.error("CloudFlow API startup failed (%s)", type(exc).__name__)
        raise RuntimeError(
            "Impossibile inizializzare il servizio CloudFlow. "
            "Verifica la configurazione del server."
        ) from None
    yield


app = FastAPI(
    title="CloudFlow Support API",
    version="1.0.0",
    lifespan=lifespan,
)

# The static site is served locally on port 8000 according to README.md.
# Keep the allowlist narrow; add the deployed frontend origin when publishing.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://127.0.0.1:8000"],
    allow_credentials=False,
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)


@app.post("/api/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    """Answer one user question and return the updated in-memory client history."""
    question = payload.question.strip()
    if not question:
        raise HTTPException(
            status_code=422,
            detail="La domanda non può essere vuota.",
        )

    history = _validated_history(payload.conversation_history)
    if history and history[-1]["role"] != "assistant":
        raise HTTPException(
            status_code=422,
            detail="La cronologia deve terminare con una risposta dell'assistente.",
        )

    try:
        if getattr(app.state, "mock_mode", False):
            answer = MOCK_ANSWER
        else:
            # Each request performs fresh retrieval; prior turns are only passed
            # as dialog context and are not used as the source of company facts.
            answer = answer_question(
                question,
                client=app.state.gemini_client,
                search_context=app.state.search_context,
                model_name=app.state.gemini_model_name,
                conversation_history=history,
            )
        update_conversation_history(history, "user", question)
        update_conversation_history(history, "assistant", answer)
    except HTTPException:
        raise
    except Exception as exc:
        # Log the stage and safe exception metadata only. Never log exception
        # text, request data, API credentials, prompts, or retrieved content.
        provider_code, provider_status = _provider_error_fields(exc)
        logger.error(
            "CloudFlow chat request failed phase=answer_question "
            "exception_type=%s provider_http_code=%s provider_status=%s",
            type(exc).__name__,
            provider_code if provider_code is not None else "unavailable",
            provider_status if provider_status is not None else "unavailable",
        )
        raise HTTPException(
            status_code=502,
            detail="Il servizio di risposta non è al momento disponibile. Riprova più tardi.",
        ) from None

    return ChatResponse(
        answer=answer,
        conversation_history=[ChatMessage(**message) for message in history],
    )
