"""
Configuration file for the Healthcare Intelligent Assistant
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
VECTOR_STORE_DIR = BASE_DIR / "vector_store"
TESTS_DIR = BASE_DIR / "tests"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
VECTOR_STORE_DIR.mkdir(exist_ok=True)
TESTS_DIR.mkdir(exist_ok=True)

# API Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "")

# LLM Configuration
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")  # "openai", "google", "ollama", "huggingface"
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4-turbo-preview")  # or "gemini-pro", "llama2", "mistral", etc.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# Ollama Configuration (for local models)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# HuggingFace Configuration
HUGGINGFACE_MODEL = os.getenv("HUGGINGFACE_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")

# Temperature for generation (0-1, lower = more deterministic)
TEMPERATURE = 0.2  # Lower temperature for clinical accuracy

# RAG Configuration
CHUNK_SIZE = 1000  # Size of text chunks for embedding
CHUNK_OVERLAP = 200  # Overlap between chunks
TOP_K_RETRIEVAL = 5  # Number of documents to retrieve
SIMILARITY_THRESHOLD = 0.7  # Minimum similarity score for retrieval

# FAISS Configuration
VECTOR_DIMENSION = 1536  # For OpenAI text-embedding-3-small
FAISS_INDEX_TYPE = "IndexFlatL2"  # Exact search

# System prompts
SYSTEM_ROLE = """You are an advanced clinical decision support assistant designed to help healthcare 
professionals make informed decisions. You provide accurate, evidence-based information by leveraging 
a comprehensive medical knowledge base and patient records.

IMPORTANT GUIDELINES:
1. Always base your responses on the retrieved context
2. If information is not available in the context, clearly state this
3. Use clinical reasoning and cite relevant guidelines when applicable
4. Maintain patient privacy and confidentiality
5. Provide clear, actionable recommendations
6. Highlight any potential risks or contraindications
7. Never make definitive diagnoses - suggest considerations for healthcare providers

Remember: You are a support tool, not a replacement for clinical judgment."""

# Synthetic data configuration
NUM_PATIENTS = 50
NUM_MEDICAL_DOCS = 100
NUM_CLINICAL_GUIDELINES = 20

# Logging
LOG_LEVEL = "INFO"
