"""
Chat translation and chat save routes.

This router exposes the chat page's translation and save endpoints while
delegating the heavy lifting to ``app.services.chat_translation``.
"""

from __future__ import annotations

from fastapi import APIRouter

from app.config import DATA_DIR as _DATA_DIR
from app.config import HERE as _HERE
from app.schema import (
    ChatSaveRequest,
    ChatSaveResponse,
    ChatTranslationRequest,
    ChatTranslationResponse,
)
from app.services.chat_translation import save_chat as persist_chat_save
from app.services.chat_translation import translate_chat as run_chat_translation

router = APIRouter(tags=["chat"])


@router.post(
    "/api/translate_chat",
    response_model=ChatTranslationResponse,
    summary="OOC→IC translation for one or two characters",
)
def translate_chat(req: ChatTranslationRequest) -> ChatTranslationResponse:
    """Translate one or two OOC chat messages into in-character output."""
    return run_chat_translation(req, _HERE)


@router.post("/api/save_chat", response_model=ChatSaveResponse)
def save_chat(req: ChatSaveRequest) -> ChatSaveResponse:
    """Persist a chat-session save package under the configured data root."""
    return persist_chat_save(req, _DATA_DIR, _HERE)
