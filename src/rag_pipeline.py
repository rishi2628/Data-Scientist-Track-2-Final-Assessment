"""
RAG Pipeline Implementation

Orchestrates the retrieval-augmented generation process for clinical queries.
"""
from typing import Dict, List, Optional, Tuple
import config
from src.vectorstore import VectorStoreManager
from src.prompt_templates import (
    create_rag_prompt,
    create_patient_summary_prompt,
    create_retrieval_query
)


class ClinicalRAGPipeline:
    """RAG pipeline for clinical decision support"""
    
    def __init__(self, llm, vector_store_manager: VectorStoreManager):
        """
        Initialize RAG pipeline.
        
        Args:
            llm: LangChain LLM instance
            vector_store_manager: Initialized vector store manager
        """
        self.llm = llm
        self.vector_store = vector_store_manager
        self.conversation_history = []
    
    def query(
        self,
        user_query: str,
        patient_identifier: Optional[str] = None,
        include_history: bool = False,
        k: int = None
    ) -> Dict:
        """
        Process a clinical query using RAG.
        
        Args:
            user_query: User's clinical question
            patient_identifier: Optional patient name/ID for patient-specific queries
            include_history: Whether to include conversation history
            k: Number of documents to retrieve
        
        Returns:
            Dictionary with response and metadata
        """
        # Optimize query for retrieval
        retrieval_query = create_retrieval_query(user_query)
        
        # Retrieve relevant context
        print("\n[RETRIEVAL] Retrieving relevant medical knowledge...")
        relevant_docs = self.vector_store.search(retrieval_query, k=k)
        
        # Format retrieved context
        context_parts = []
        sources = []
        
        for i, (doc, score) in enumerate(relevant_docs, 1):
            # Lower score means higher similarity in FAISS L2 distance
            similarity = 1 / (1 + score)  # Convert distance to similarity
            
            if similarity < config.SIMILARITY_THRESHOLD:
                continue
            
            context_parts.append(f"\n--- Source {i} (Relevance: {similarity:.2%}) ---")
            context_parts.append(f"Type: {doc.metadata.get('type', 'unknown')}")
            
            if 'title' in doc.metadata:
                context_parts.append(f"Title: {doc.metadata['title']}")
            
            context_parts.append(f"\n{doc.page_content}\n")
            
            sources.append({
                "type": doc.metadata.get('type'),
                "title": doc.metadata.get('title', 'Unknown'),
                "relevance": similarity
            })
        
        medical_context = "\n".join(context_parts) if context_parts else "No highly relevant context found."
        
        # Get patient-specific context if requested
        patient_context = None
        if patient_identifier:
            print(f"[PATIENT] Retrieving patient information for: {patient_identifier}")
            patient_context = self.vector_store.get_patient_context(patient_identifier)
        
        # Create comprehensive prompt
        prompt = create_rag_prompt(
            query=user_query,
            context=medical_context,
            patient_context=patient_context
        )
        
        # Add conversation history if requested
        if include_history and self.conversation_history:
            history_text = "\n\n=== CONVERSATION HISTORY ===\n"
            for entry in self.conversation_history[-3:]:  # Last 3 exchanges
                history_text += f"Q: {entry['query']}\nA: {entry['response']}\n\n"
            prompt = history_text + prompt
        
        # Generate response
        print("[GENERATION] Generating clinical response...")
        response = self.llm.invoke(prompt)
        
        # Extract text from response (handles different LLM response formats)
        if hasattr(response, 'content'):
            response_text = response.content
        else:
            response_text = str(response)
        
        # Store in conversation history
        self.conversation_history.append({
            "query": user_query,
            "response": response_text,
            "patient": patient_identifier
        })
        
        return {
            "query": user_query,
            "response": response_text,
            "sources": sources,
            "patient_context": patient_context is not None,
            "num_sources": len(sources)
        }
    
    def query_with_patient(self, user_query: str, patient_name: str, **kwargs) -> Dict:
        """
        Convenience method for patient-specific queries.
        
        Args:
            user_query: Clinical question
            patient_name: Patient name or ID
            **kwargs: Additional arguments for query()
        
        Returns:
            Query result dictionary
        """
        return self.query(user_query, patient_identifier=patient_name, **kwargs)
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
    
    def get_history(self) -> List[Dict]:
        """Get conversation history"""
        return self.conversation_history


class EnhancedClinicalRAG(ClinicalRAGPipeline):
    """Enhanced RAG with additional features"""
    
    def multi_step_reasoning(self, query: str, patient_identifier: Optional[str] = None) -> Dict:
        """
        Perform multi-step reasoning for complex queries.
        
        Args:
            query: Complex clinical question
            patient_identifier: Optional patient identifier
        
        Returns:
            Response with reasoning steps
        """
        print("\n[REASONING] Performing multi-step clinical reasoning...")
        
        # Step 1: Decompose the query
        decomposition_prompt = f"""Break down this complex clinical query into simpler sub-questions:

Query: {query}

Provide 2-4 specific sub-questions that would help answer this comprehensively."""
        
        decomposition = self.llm.invoke(decomposition_prompt)
        sub_questions = str(decomposition.content if hasattr(decomposition, 'content') else decomposition)
        
        print(f"\n[SUB-QUESTIONS] Identified:\n{sub_questions}\n")
        
        # Step 2: Answer each sub-question
        # For simplicity, we'll use the main query with enhanced retrieval
        result = self.query(query, patient_identifier=patient_identifier, k=10)
        
        # Add reasoning steps to result
        result["reasoning_steps"] = sub_questions
        result["approach"] = "multi-step"
        
        return result
    
    def explain_recommendations(self, response_dict: Dict) -> str:
        """
        Generate explanation for clinical recommendations.
        
        Args:
            response_dict: Response from query()
        
        Returns:
            Detailed explanation
        """
        explanation_prompt = f"""Provide a detailed explanation of the clinical reasoning behind this response:

Original Query: {response_dict['query']}
Response: {response_dict['response']}

Explain:
1. What clinical guidelines support this recommendation?
2. What patient factors were considered?
3. What are the key decision points?
4. What alternatives might exist?

Provide a structured explanation."""
        
        explanation = self.llm.invoke(explanation_prompt)
        return explanation.content if hasattr(explanation, 'content') else str(explanation)
    
    def identify_drug_interactions(self, medications: List[str]) -> Dict:
        """
        Check for drug interactions.
        
        Args:
            medications: List of medication names
        
        Returns:
            Dictionary with interaction information
        """
        query = f"What are the potential drug interactions between these medications: {', '.join(medications)}?"
        
        # Search specifically for drug interaction information
        result = self.query(query, k=8)
        
        return {
            "medications": medications,
            "interaction_analysis": result["response"],
            "sources": result["sources"]
        }
    
    def calculate_clinical_score(self, score_name: str, patient_data: Dict) -> Dict:
        """
        Calculate clinical risk scores (e.g., CHA2DS2-VASc, TIMI, etc.)
        
        Args:
            score_name: Name of the clinical score
            patient_data: Patient information
        
        Returns:
            Score calculation and interpretation
        """
        query = f"""Calculate the {score_name} score for this patient and provide interpretation:

Patient Information:
{create_patient_summary_prompt(patient_data)}

Provide:
1. The calculated score and each component
2. Clinical interpretation
3. Recommended actions based on the score"""
        
        result = self.query(query, k=5)
        
        return {
            "score_type": score_name,
            "calculation": result["response"],
            "sources": result["sources"]
        }
