import logging
from typing import List, Dict, Any
from openai import AsyncOpenAI
import os
from dotenv import load_dotenv
from ..database.vector_db import search_relevant_content
from ..models.user_preferences import UserPreferences

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# Initialize OpenAI client
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class RAGService:
    """
    Service class for handling RAG (Retrieval-Augmented Generation) operations.
    """

    def __init__(self):
        self.model_name = os.getenv("OPENAI_MODEL_NAME", "gpt-3.5-turbo")
        self.max_tokens = int(os.getenv("OPENAI_MAX_TOKENS", "1000"))
        self.temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.7"))

    async def generate_response(self, query: str, user_preferences: UserPreferences = None, context: str = None) -> Dict[str, Any]:
        """
        Generate a response to the user query using RAG approach.

        Args:
            query: The user's query
            user_preferences: User preferences for personalization
            context: Optional specific context to use instead of searching

        Returns:
            Dictionary containing the response and metadata
        """
        try:
            # Get relevant context if not provided
            if not context:
                # Search for relevant content in the vector database
                relevant_docs = await search_relevant_content(query, limit=5)

                # Combine the content from relevant documents
                context_parts = []
                sources = []

                for doc in relevant_docs:
                    context_parts.append(doc["content"])
                    sources.append({
                        "title": doc["title"],
                        "chapter": doc["chapter"],
                        "section": doc["section"],
                        "url": doc["url"],
                        "relevance_score": doc["score"]
                    })

                context = "\n\n".join(context_parts)
            else:
                sources = []  # No sources when context is provided directly

            # Prepare the prompt for the LLM
            system_prompt = self._get_system_prompt(user_preferences)
            user_prompt = self._get_user_prompt(query, context)

            # Call the OpenAI API
            response = await openai_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )

            # Extract the response
            generated_text = response.choices[0].message.content.strip()

            # Return the response with metadata
            result = {
                "query": query,
                "response": generated_text,
                "sources": sources,
                "model": self.model_name,
                "timestamp": response.created
            }

            logger.info(f"Generated response for query: {query[:50]}...")
            return result

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise

    def _get_system_prompt(self, user_preferences: UserPreferences = None) -> str:
        """
        Get the system prompt for the LLM based on user preferences.

        Args:
            user_preferences: User preferences for personalization

        Returns:
            System prompt string
        """
        base_prompt = (
            "You are an AI assistant for the Physical AI & Humanoid Robotics Textbook. "
            "Your purpose is to answer questions based on the textbook content provided in the context. "
            "Always cite the specific chapters, sections, or pages when possible. "
            "If the answer is not available in the provided context, clearly state that the information is not in the textbook. "
            "Provide accurate, helpful, and concise answers."
        )

        # Add personalization based on user preferences
        if user_preferences:
            if user_preferences.adaptive_difficulty == "beginner":
                base_prompt += " Explain concepts in simple terms suitable for beginners. "
            elif user_preferences.adaptive_difficulty == "advanced":
                base_prompt += " Provide detailed explanations with advanced terminology. "

            if user_preferences.adaptive_code_samples:
                base_prompt += " Include relevant code examples when applicable. "

        return base_prompt

    def _get_user_prompt(self, query: str, context: str) -> str:
        """
        Get the user prompt for the LLM.

        Args:
            query: The user's query
            context: The retrieved context

        Returns:
            User prompt string
        """
        return (
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            f"Please provide a comprehensive answer based on the context provided, "
            f"and cite specific chapters or sections when possible."
        )

    async def process_selected_text_query(self, query: str, selected_text: str) -> Dict[str, Any]:
        """
        Process a query that should be answered based only on the selected text.

        Args:
            query: The user's query
            selected_text: The text that was selected by the user

        Returns:
            Dictionary containing the response and metadata
        """
        try:
            # Prepare the prompt for the LLM with only the selected text as context
            system_prompt = (
                "You are an AI assistant for the Physical AI & Humanoid Robotics Textbook. "
                "Your purpose is to answer questions based ONLY on the selected text provided in the context. "
                "Do not use any external knowledge. If the answer cannot be derived from the selected text, "
                "clearly state that the information is not available in the selected text. "
                "Provide accurate, helpful, and concise answers."
            )

            user_prompt = (
                f"Selected Text:\n{selected_text}\n\n"
                f"Question: {query}\n\n"
                f"Please provide an answer based ONLY on the selected text provided, "
                f"and do not use any external knowledge."
            )

            # Call the OpenAI API
            response = await openai_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )

            # Extract the response
            generated_text = response.choices[0].message.content.strip()

            # Return the response with metadata
            result = {
                "query": query,
                "response": generated_text,
                "sources": [{"type": "selected_text", "content": selected_text[:200] + "..." if len(selected_text) > 200 else selected_text}],
                "model": self.model_name,
                "timestamp": response.created
            }

            logger.info(f"Generated selected text response for query: {query[:50]}...")
            return result

        except Exception as e:
            logger.error(f"Error processing selected text query: {e}")
            raise

# Global instance
rag_service = RAGService()