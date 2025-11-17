"""
Vector Store Management using FAISS

Handles document embedding, indexing, and retrieval.
"""
import json
import pickle
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np

try:
    import faiss
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.docstore.in_memory import InMemoryDocstore
    from langchain_community.vectorstores import FAISS
    from langchain.schema import Document
    IMPORTS_AVAILABLE = True
except ImportError:
    print("Warning: LangChain or FAISS not installed. Install with: pip install -r requirements.txt")
    IMPORTS_AVAILABLE = False
    # Define placeholder for type hints
    Document = None

import config


class VectorStoreManager:
    """Manages FAISS vector store for medical documents"""
    
    def __init__(self, embeddings_model):
        """
        Initialize vector store manager.
        
        Args:
            embeddings_model: LangChain embeddings model (OpenAI or Google)
        """
        self.embeddings = embeddings_model
        self.vector_store = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def load_documents(self) -> List[Document]:
        """
        Load all medical documents from JSON files.
        
        Returns:
            List of LangChain Document objects
        """
        documents = []
        
        # Load patient records
        print("Loading patient records...")
        patient_file = config.DATA_DIR / "patient_records.json"
        if patient_file.exists():
            with open(patient_file, 'r') as f:
                patients = json.load(f)
            
            for patient in patients:
                # Create a readable text representation
                text = self._format_patient_record(patient)
                doc = Document(
                    page_content=text,
                    metadata={
                        "source": "patient_record",
                        "patient_id": patient.get("patient_id"),
                        "patient_name": patient.get("name"),
                        "type": "patient_data"
                    }
                )
                documents.append(doc)
        
        # Load medical knowledge
        print("Loading medical knowledge base...")
        knowledge_file = config.DATA_DIR / "medical_knowledge.json"
        if knowledge_file.exists():
            with open(knowledge_file, 'r') as f:
                knowledge_docs = json.load(f)
            
            for doc_data in knowledge_docs:
                doc = Document(
                    page_content=f"Title: {doc_data['title']}\nCategory: {doc_data['category']}\n\n{doc_data['content']}",
                    metadata={
                        "source": "medical_knowledge",
                        "title": doc_data['title'],
                        "category": doc_data.get('category', 'General'),
                        "type": "knowledge"
                    }
                )
                documents.append(doc)
        
        # Load clinical guidelines
        print("Loading clinical guidelines...")
        guidelines_file = config.DATA_DIR / "clinical_guidelines.json"
        if guidelines_file.exists():
            with open(guidelines_file, 'r') as f:
                guidelines = json.load(f)
            
            for guideline in guidelines:
                doc = Document(
                    page_content=f"Guideline: {guideline['title']}\nOrganization: {guideline['organization']}\nYear: {guideline['year']}\n\n{guideline['content']}",
                    metadata={
                        "source": "clinical_guideline",
                        "title": guideline['title'],
                        "organization": guideline.get('organization'),
                        "type": "guideline"
                    }
                )
                documents.append(doc)
        
        print(f"Loaded {len(documents)} documents")
        return documents
    
    def _format_patient_record(self, patient: Dict) -> str:
        """Format patient record as readable text"""
        parts = [
            f"Patient Record - {patient.get('name')}",
            f"Patient ID: {patient.get('patient_id')}",
            f"Age: {patient.get('age')} years",
            f"Gender: {patient.get('gender')}",
            f"MRN: {patient.get('medical_record_number')}",
            "",
            "Active Medical Conditions:",
            ", ".join(patient.get('conditions', [])),
            "",
            "Current Medications:",
        ]
        
        for med in patient.get('medications', []):
            parts.append(f"  - {med}")
        
        parts.append("")
        parts.append(f"Known Allergies: {', '.join(patient.get('allergies', []))}")
        
        # Add vital signs
        vitals = patient.get('vital_signs', {})
        parts.append("")
        parts.append("Recent Vital Signs:")
        parts.append(f"  Blood Pressure: {vitals.get('blood_pressure')} mmHg")
        parts.append(f"  Heart Rate: {vitals.get('heart_rate')} bpm")
        parts.append(f"  Temperature: {vitals.get('temperature')}°F")
        parts.append(f"  SpO2: {vitals.get('oxygen_saturation')}%")
        
        # Add lab results
        labs = patient.get('lab_results', {})
        if labs:
            parts.append("")
            parts.append("Recent Laboratory Results:")
            for lab_name, lab_data in labs.items():
                parts.append(f"  {lab_name}: {lab_data['value']} {lab_data['unit']}")
        
        parts.append("")
        parts.append(f"Last Visit: {patient.get('last_visit')}")
        parts.append(f"Primary Care Physician: {patient.get('primary_care_physician')}")
        
        return "\n".join(parts)
    
    def create_vector_store(self, documents: List[Document] = None):
        """
        Create FAISS vector store from documents.
        
        Args:
            documents: List of documents to index. If None, loads from data files.
        """
        if documents is None:
            documents = self.load_documents()
        
        # Split documents into chunks
        print("Splitting documents into chunks...")
        chunks = self.text_splitter.split_documents(documents)
        print(f"Created {len(chunks)} chunks")
        
        # Create vector store
        print("Creating embeddings and building FAISS index...")
        print("(This may take a few minutes depending on the number of documents)")
        
        self.vector_store = FAISS.from_documents(
            documents=chunks,
            embedding=self.embeddings
        )
        
        print("✓ Vector store created successfully!")
    
    def save_vector_store(self, path: Path = None):
        """
        Save vector store to disk.
        
        Args:
            path: Directory path to save the vector store
        """
        if self.vector_store is None:
            raise ValueError("No vector store to save. Create one first.")
        
        if path is None:
            path = config.VECTOR_STORE_DIR
        
        print(f"Saving vector store to {path}...")
        self.vector_store.save_local(str(path))
        print("✓ Vector store saved!")
    
    def load_vector_store(self, path: Path = None):
        """
        Load vector store from disk.
        
        Args:
            path: Directory path to load the vector store from
        """
        if path is None:
            path = config.VECTOR_STORE_DIR
        
        if not path.exists():
            raise FileNotFoundError(f"Vector store not found at {path}")
        
        print(f"Loading vector store from {path}...")
        self.vector_store = FAISS.load_local(
            str(path),
            self.embeddings,
            allow_dangerous_deserialization=True  # Our data is safe
        )
        print("✓ Vector store loaded!")
    
    def search(self, query: str, k: int = None, filter_dict: Dict = None) -> List[Tuple[Document, float]]:
        """
        Search for relevant documents.
        
        Args:
            query: Search query
            k: Number of results to return
            filter_dict: Metadata filters (e.g., {"type": "patient_data"})
        
        Returns:
            List of (Document, similarity_score) tuples
        """
        if self.vector_store is None:
            raise ValueError("Vector store not initialized. Create or load one first.")
        
        if k is None:
            k = config.TOP_K_RETRIEVAL
        
        # Perform similarity search with scores
        results = self.vector_store.similarity_search_with_score(query, k=k)
        
        # Filter by metadata if specified
        if filter_dict:
            results = [
                (doc, score) for doc, score in results
                if all(doc.metadata.get(key) == value for key, value in filter_dict.items())
            ]
        
        return results
    
    def get_relevant_context(self, query: str, k: int = None) -> str:
        """
        Get formatted context from relevant documents.
        
        Args:
            query: Search query
            k: Number of documents to retrieve
        
        Returns:
            Formatted context string
        """
        results = self.search(query, k=k)
        
        context_parts = []
        for i, (doc, score) in enumerate(results, 1):
            # Only include documents above similarity threshold
            if score > config.SIMILARITY_THRESHOLD:
                continue
                
            context_parts.append(f"--- Source {i} (Relevance: {1-score:.2f}) ---")
            context_parts.append(f"Type: {doc.metadata.get('type', 'unknown')}")
            if 'title' in doc.metadata:
                context_parts.append(f"Title: {doc.metadata['title']}")
            context_parts.append(f"\n{doc.page_content}\n")
        
        return "\n".join(context_parts) if context_parts else "No relevant context found."
    
    def get_patient_context(self, patient_identifier: str) -> str:
        """
        Get context for a specific patient.
        
        Args:
            patient_identifier: Patient name, ID, or MRN
        
        Returns:
            Formatted patient context
        """
        # Search for patient with broader query
        query = f"Patient {patient_identifier}"
        results = self.search(query, k=10, filter_dict={"type": "patient_data"})
        
        # Find best match
        for doc, score in results:
            patient_name = doc.metadata.get('patient_name', '')
            patient_id = doc.metadata.get('patient_id', '')
            
            if (patient_identifier.lower() in patient_name.lower() or 
                patient_identifier.upper() in patient_id.upper()):
                return doc.page_content
        
        return f"Patient '{patient_identifier}' not found in records."


def initialize_vector_store(embeddings_model, force_rebuild: bool = False):
    """
    Initialize or load vector store.
    
    Args:
        embeddings_model: LangChain embeddings model
        force_rebuild: If True, rebuild vector store even if it exists
    
    Returns:
        VectorStoreManager instance
    """
    manager = VectorStoreManager(embeddings_model)
    
    vector_store_path = config.VECTOR_STORE_DIR
    
    if vector_store_path.exists() and not force_rebuild:
        print("Found existing vector store, loading...")
        try:
            manager.load_vector_store()
            return manager
        except Exception as e:
            print(f"Error loading vector store: {e}")
            print("Rebuilding vector store...")
    
    # Create new vector store
    print("Creating new vector store...")
    manager.create_vector_store()
    manager.save_vector_store()
    
    return manager
