import asyncio
import os
from dotenv import load_dotenv
from pathlib import Path

# Add the api/src directory to the Python path so we can import our modules
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.services.document_parser import document_parser
from src.database.vector_db import add_textbook_content, init_vector_db

# Load environment variables
load_dotenv()

async def load_textbook_to_vector_db():
    """
    Load all textbook content from the website/docs directory into the vector database.
    """
    print("Initializing vector database...")
    await init_vector_db()
    print("Vector database initialized successfully")

    # Path to the textbook docs directory
    docs_dir = os.path.join(os.path.dirname(__file__), '..', 'website', 'docs')
    docs_dir = os.path.abspath(docs_dir)

    if not os.path.exists(docs_dir):
        print(f"Error: Docs directory not found at {docs_dir}")
        return

    print(f"Loading textbook content from {docs_dir}...")

    # Parse all textbook documents
    documents = document_parser.parse_all_textbook_docs(docs_dir)

    if not documents:
        print("No documents found to load")
        return

    print(f"Found {len(documents)} document sections to load")

    # Add documents to vector database
    try:
        await add_textbook_content(documents)
        print(f"Successfully loaded {len(documents)} document sections into vector database")
    except Exception as e:
        print(f"Error loading documents to vector database: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(load_textbook_to_vector_db())