import os
from typing import List, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams
from sentence_transformers import SentenceTransformer
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# Initialize Qdrant client
if os.getenv("QDRANT_API_KEY"):
    # Use cloud instance
    qdrant_client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
        prefer_grpc=True
    )
else:
    # Use local instance
    qdrant_client = QdrantClient(host="localhost", port=6333)

# Initialize sentence transformer model for embeddings
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Collection name for textbook content
COLLECTION_NAME = "textbook_content"

async def init_vector_db():
    """
    Initialize the vector database with the required collection.
    """
    try:
        # Check if collection exists
        collections = qdrant_client.get_collections()
        collection_exists = any(col.name == COLLECTION_NAME for col in collections.collections)

        if not collection_exists:
            # Create collection with vector configuration
            qdrant_client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=embedding_model.get_sentence_embedding_dimension(), distance=Distance.COSINE),
            )

            # Create payload index for content type
            qdrant_client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name="chapter",
                field_schema=models.TextIndexParams(
                    type="text",
                    tokenizer=models.TokenizerType.WORD,
                    min_token_len=2,
                    max_token_len=10,
                    lowercase=True,
                )
            )

            logger.info(f"Created collection '{COLLECTION_NAME}' with vector configuration")
        else:
            logger.info(f"Collection '{COLLECTION_NAME}' already exists")

        # Test connection
        qdrant_client.get_collection(COLLECTION_NAME)
        logger.info("Vector database initialized successfully")

    except Exception as e:
        logger.error(f"Error initializing vector database: {e}")
        raise

async def add_textbook_content(documents: List[dict]):
    """
    Add textbook content to the vector database.

    Args:
        documents: List of documents with keys: id, content, chapter, section
    """
    try:
        # Prepare points for insertion
        points = []
        for i, doc in enumerate(documents):
            # Generate embedding for the content
            embedding = embedding_model.encode(doc["content"]).tolist()

            point = models.PointStruct(
                id=doc["id"],
                vector=embedding,
                payload={
                    "content": doc["content"],
                    "chapter": doc["chapter"],
                    "section": doc.get("section", ""),
                    "title": doc.get("title", ""),
                    "url": doc.get("url", "")
                }
            )
            points.append(point)

        # Upload points to the collection
        qdrant_client.upsert(
            collection_name=COLLECTION_NAME,
            points=points
        )

        logger.info(f"Added {len(points)} documents to vector database")

    except Exception as e:
        logger.error(f"Error adding textbook content: {e}")
        raise

async def search_relevant_content(query: str, limit: int = 5) -> List[dict]:
    """
    Search for relevant content based on the query.

    Args:
        query: The search query
        limit: Number of results to return

    Returns:
        List of relevant documents with content and metadata
    """
    try:
        # Generate embedding for the query
        query_embedding = embedding_model.encode(query).tolist()

        # Search in the collection
        search_results = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_embedding,
            limit=limit,
            with_payload=True
        )

        # Format results
        results = []
        for hit in search_results:
            results.append({
                "id": hit.id,
                "content": hit.payload["content"],
                "chapter": hit.payload["chapter"],
                "section": hit.payload["section"],
                "title": hit.payload["title"],
                "url": hit.payload["url"],
                "score": hit.score
            })

        logger.info(f"Found {len(results)} relevant results for query: {query[:50]}...")
        return results

    except Exception as e:
        logger.error(f"Error searching for content: {e}")
        raise

async def get_all_chapters() -> List[str]:
    """
    Get all unique chapters in the database.

    Returns:
        List of unique chapter names
    """
    try:
        # Get all points and extract unique chapters
        all_points = qdrant_client.scroll(
            collection_name=COLLECTION_NAME,
            limit=10000,  # Adjust as needed
            with_payload=True
        )

        chapters = set()
        for point, _ in all_points:
            if "chapter" in point.payload:
                chapters.add(point.payload["chapter"])

        return list(chapters)

    except Exception as e:
        logger.error(f"Error getting all chapters: {e}")
        raise