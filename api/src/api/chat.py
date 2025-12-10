from fastapi import APIRouter, HTTPException, Depends
from typing import Optional
import logging

from ..services.rag_service import rag_service
from ..models.user_preferences import UserPreferences

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/chat")
async def chat_endpoint(
    query: str,
    user_preferences: Optional[UserPreferences] = None,
    context: Optional[str] = None
):
    """
    Main chat endpoint that processes user queries using RAG.

    Args:
        query: The user's question
        user_preferences: Optional user preferences for personalization
        context: Optional specific context to use instead of searching

    Returns:
        Response from the RAG service
    """
    try:
        if not query or len(query.strip()) == 0:
            raise HTTPException(status_code=400, detail="Query cannot be empty")

        # Generate response using RAG service
        response = await rag_service.generate_response(query, user_preferences, context)

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/chat-selected-text")
async def chat_selected_text_endpoint(
    query: str,
    selected_text: str,
    user_preferences: Optional[UserPreferences] = None
):
    """
    Chat endpoint that processes queries based only on selected text.

    Args:
        query: The user's question about the selected text
        selected_text: The text that was selected by the user
        user_preferences: Optional user preferences for personalization

    Returns:
        Response from the RAG service based only on selected text
    """
    try:
        if not query or len(query.strip()) == 0:
            raise HTTPException(status_code=400, detail="Query cannot be empty")

        if not selected_text or len(selected_text.strip()) == 0:
            raise HTTPException(status_code=400, detail="Selected text cannot be empty")

        # Process query based only on selected text
        response = await rag_service.process_selected_text_query(query, selected_text)

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in selected text chat endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/health")
async def chat_health():
    """
    Health check for the chat service.
    """
    return {"status": "healthy", "service": "chat API"}