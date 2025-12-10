from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging
import os
from dotenv import load_dotenv

from .api.chat import router as chat_router
from .database.vector_db import init_vector_db

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan event handler for FastAPI application.
    This will run startup and shutdown events.
    """
    logger.info("Initializing vector database...")
    await init_vector_db()
    logger.info("Vector database initialized successfully")
    yield
    logger.info("Shutting down application...")

# Create FastAPI application
app = FastAPI(
    title="Physical AI & Humanoid Robotics Textbook RAG API",
    description="API for Retrieval-Augmented Generation chatbot for the Physical AI & Humanoid Robotics textbook",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(chat_router, prefix="/api/v1", tags=["chat"])

@app.get("/")
async def root():
    """
    Root endpoint for health check.
    """
    return {"message": "Physical AI & Humanoid Robotics Textbook RAG API is running!"}

@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    """
    return {"status": "healthy", "service": "RAG API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.src.main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=True
    )