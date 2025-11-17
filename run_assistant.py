"""
Healthcare Intelligent Assistant - Main Application

This is the main entry point for the clinical decision support system.
Run this script to interact with the assistant.
"""
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

import config


def check_api_keys():
    """Check if API keys are configured"""
    # Local and Ollama don't need API keys
    if config.LLM_PROVIDER in ["ollama", "local"]:
        print(f"✓ Using {config.LLM_PROVIDER} (no API key needed)")
        return True
    
    # HuggingFace with local embeddings doesn't strictly need a key for some models
    if config.LLM_PROVIDER == "huggingface" and not config.HUGGINGFACE_API_KEY:
        print("⚠️  Warning: No HuggingFace API key found. Some models may not work.")
        print("   You can get a free token at: https://huggingface.co/settings/tokens")
        return True  # Allow to continue, some models work without key
    
    if not config.OPENAI_API_KEY and not config.GOOGLE_API_KEY and not config.HUGGINGFACE_API_KEY:
        print("❌ Error: No API key found!")
        print("\n💡 You have several options:")
        print("\n1. 🆓 Use Ollama (FREE, Local, No API Key):")
        print("   - Install from https://ollama.ai")
        print("   - Run: ollama pull llama2")
        print("   - Set in .env: LLM_PROVIDER=ollama")
        print("\n2. 🆓 Use Google Gemini (FREE API):")
        print("   - Get key from https://makersuite.google.com/app/apikey")
        print("   - Set in .env: LLM_PROVIDER=google, GOOGLE_API_KEY=your_key")
        print("\n3. 💰 Use OpenAI (Paid, Best Quality):")
        print("   - Get key from https://platform.openai.com/")
        print("   - Set in .env: LLM_PROVIDER=openai, OPENAI_API_KEY=your_key")
        print("\n📖 See OLLAMA_SETUP.md for detailed free setup guide")
        return False
    return True


def initialize_system():
    """Initialize the RAG system"""
    print("\n" + "="*60)
    print("HEALTHCARE INTELLIGENT ASSISTANT")
    print("Clinical Decision Support System with RAG")
    print("="*60 + "\n")
    
    if not check_api_keys():
        sys.exit(1)
    
    # Check if data exists
    data_exists = (
        (config.DATA_DIR / "patient_records.json").exists() and
        (config.DATA_DIR / "medical_knowledge.json").exists()
    )
    
    if not data_exists:
        print("📊 Synthetic data not found. Generating...")
        from src.data_generator import save_data
        save_data()
        print()
    
    # Initialize LLM and embeddings
    print("🔧 Initializing AI models...")
    
    try:
        if config.LLM_PROVIDER == "openai":
            from langchain_openai import ChatOpenAI, OpenAIEmbeddings
            
            llm = ChatOpenAI(
                model=config.LLM_MODEL,
                temperature=config.TEMPERATURE,
                api_key=config.OPENAI_API_KEY
            )
            embeddings = OpenAIEmbeddings(
                model=config.EMBEDDING_MODEL,
                api_key=config.OPENAI_API_KEY
            )
            print(f"  ✓ Using OpenAI: {config.LLM_MODEL}")
            
        elif config.LLM_PROVIDER == "google":
            from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
            
            llm = ChatGoogleGenerativeAI(
                model=config.LLM_MODEL,
                temperature=config.TEMPERATURE,
                google_api_key=config.GOOGLE_API_KEY
            )
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=config.GOOGLE_API_KEY
            )
            print(f"  ✓ Using Google: {config.LLM_MODEL}")
            
        elif config.LLM_PROVIDER == "ollama":
            from langchain_community.llms import Ollama
            from langchain_community.embeddings import HuggingFaceEmbeddings
            
            llm = Ollama(
                model=config.LLM_MODEL,
                base_url=config.OLLAMA_BASE_URL,
                temperature=config.TEMPERATURE
            )
            # Use local sentence transformers for embeddings
            embeddings = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2"  # Fast, free local embeddings
            )
            print(f"  ✓ Using Ollama (Local): {config.LLM_MODEL}")
            print("  ✓ Using Local Embeddings: all-MiniLM-L6-v2")
            
        elif config.LLM_PROVIDER == "huggingface":
            from langchain_community.llms import HuggingFaceHub
            from langchain_community.embeddings import HuggingFaceEmbeddings
            
            llm = HuggingFaceHub(
                repo_id=config.HUGGINGFACE_MODEL,
                huggingfacehub_api_token=config.HUGGINGFACE_API_KEY,
                model_kwargs={"temperature": config.TEMPERATURE, "max_length": 512}
            )
            embeddings = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2"
            )
            print(f"  ✓ Using HuggingFace: {config.HUGGINGFACE_MODEL}")
            
        elif config.LLM_PROVIDER == "local":
            from transformers import pipeline
            from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
            from langchain_community.embeddings import HuggingFaceEmbeddings
            
            print(f"  Loading local model: {config.LLM_MODEL}...")
            print("  (First run will download model, please wait...)")
            
            # Create text generation pipeline
            hf_pipeline = pipeline(
                "text-generation",
                model=config.LLM_MODEL,
                max_new_tokens=150,  # Generate up to 150 new tokens
                temperature=config.TEMPERATURE,
                device=-1  # CPU
            )
            
            llm = HuggingFacePipeline(pipeline=hf_pipeline)
            
            # Use local sentence transformers for embeddings
            embeddings = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2"
            )
            print(f"  ✓ Using Local Model: {config.LLM_MODEL}")
            print("  ✓ Using Local Embeddings: all-MiniLM-L6-v2")
            
        else:
            print(f"❌ Unknown LLM provider: {config.LLM_PROVIDER}")
            print("\nSupported providers: openai, google, ollama, huggingface, local")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Error initializing LLM: {e}")
        print("\nMake sure you have installed the required packages:")
        print("  pip install -r requirements.txt")
        sys.exit(1)
    
    # Initialize vector store
    print("\n🗄️  Setting up vector store...")
    from src.vectorstore import initialize_vector_store
    
    try:
        vector_store = initialize_vector_store(embeddings)
        print("  ✓ Vector store ready")
    except Exception as e:
        print(f"❌ Error initializing vector store: {e}")
        sys.exit(1)
    
    # Initialize RAG pipeline
    print("\n🔗 Initializing RAG pipeline...")
    from src.rag_pipeline import EnhancedClinicalRAG
    
    rag_pipeline = EnhancedClinicalRAG(llm, vector_store)
    print("  ✓ RAG pipeline ready")
    
    return llm, rag_pipeline


def print_response(result: dict):
    """Pretty print the response"""
    print("\n" + "="*60)
    print("RESPONSE")
    print("="*60)
    print(result['response'])
    print("\n" + "-"*60)
    print(f"📚 Sources used: {result['num_sources']}")
    if result['sources']:
        print("\nTop sources:")
        for i, source in enumerate(result['sources'][:3], 1):
            print(f"  {i}. {source['title']} (Relevance: {source['relevance']:.1%})")
    print("="*60 + "\n")


def interactive_mode(rag_pipeline):
    """Run interactive question-answering mode"""
    print("\n🩺 Interactive Clinical Assistant Mode")
    print("-" * 60)
    print("Ask clinical questions, or use special commands:")
    print("  /patient <name> - Set patient context")
    print("  /clear - Clear conversation history")
    print("  /history - Show conversation history")
    print("  /eval - Run evaluation suite")
    print("  /quit or /exit - Exit the program")
    print("-" * 60 + "\n")
    
    current_patient = None
    
    while True:
        try:
            # Get user input
            user_input = input("🩺 You: ").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.startswith('/'):
                cmd_parts = user_input.split(maxsplit=1)
                cmd = cmd_parts[0].lower()
                
                if cmd in ['/quit', '/exit']:
                    print("\n👋 Thank you for using the Clinical Assistant. Goodbye!")
                    break
                
                elif cmd == '/clear':
                    rag_pipeline.clear_history()
                    current_patient = None
                    print("✓ History cleared")
                    continue
                
                elif cmd == '/history':
                    history = rag_pipeline.get_history()
                    if not history:
                        print("No conversation history yet.")
                    else:
                        print(f"\n📜 Conversation History ({len(history)} exchanges):")
                        for i, entry in enumerate(history, 1):
                            print(f"\n{i}. Q: {entry['query']}")
                            print(f"   A: {entry['response'][:100]}...")
                    continue
                
                elif cmd == '/patient':
                    if len(cmd_parts) > 1:
                        current_patient = cmd_parts[1]
                        print(f"✓ Patient context set to: {current_patient}")
                    else:
                        print("Usage: /patient <name>")
                    continue
                
                elif cmd == '/eval':
                    print("\n🧪 Running evaluation suite...")
                    run_evaluation(rag_pipeline)
                    continue
                
                else:
                    print(f"Unknown command: {cmd}")
                    continue
            
            # Process query
            result = rag_pipeline.query(
                user_input,
                patient_identifier=current_patient,
                include_history=True
            )
            
            print_response(result)
            
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()


def demo_mode(rag_pipeline):
    """Run demo with predefined queries"""
    print("\n🎬 Demo Mode - Showcasing Clinical Assistant Capabilities\n")
    
    demo_queries = [
        {
            "query": "What are the first-line treatments for Type 2 Diabetes Mellitus?",
            "description": "General medical knowledge query"
        },
        {
            "query": "Explain the CHA2DS2-VASc score for atrial fibrillation",
            "description": "Clinical guideline query"
        },
        {
            "query": "What are important drug interactions with warfarin?",
            "description": "Drug interaction query"
        }
    ]
    
    for i, demo in enumerate(demo_queries, 1):
        print(f"\n{'='*60}")
        print(f"DEMO QUERY {i}: {demo['description']}")
        print(f"{'='*60}")
        print(f"Query: {demo['query']}\n")
        
        result = rag_pipeline.query(demo['query'])
        print_response(result)
        
        if i < len(demo_queries):
            input("\nPress Enter to continue to next demo...")


def run_evaluation(rag_pipeline):
    """Run system evaluation"""
    from src.evaluator import RAGEvaluator, create_test_cases
    
    # Get LLM from pipeline
    evaluator = RAGEvaluator(rag_pipeline, rag_pipeline.llm)
    
    # Create test cases
    test_cases = create_test_cases()
    
    # Run evaluation
    print(f"\nRunning {len(test_cases)} test cases...")
    summary = evaluator.run_test_suite(test_cases)
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Total tests: {summary['total_tests']}")
    print(f"Average quality score: {summary['average_quality_score']:.2f}/5.0")
    print(f"High quality responses: {summary['high_quality_responses']} ({summary['high_quality_percentage']:.1f}%)")
    print(f"Low hallucination risk: {summary['low_hallucination_count']} ({summary['low_hallucination_count']/summary['total_tests']*100:.1f}%)")
    print("="*60)
    
    # Save results
    evaluator.save_results()
    
    # Generate and print detailed report
    report = evaluator.generate_report()
    print("\n" + report)


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Healthcare Intelligent Assistant - Clinical Decision Support System"
    )
    parser.add_argument(
        '--mode',
        choices=['interactive', 'demo', 'eval', 'generate-data'],
        default='interactive',
        help='Operation mode (default: interactive)'
    )
    parser.add_argument(
        '--rebuild-vector-store',
        action='store_true',
        help='Rebuild the vector store from scratch'
    )
    
    args = parser.parse_args()
    
    # Handle data generation mode
    if args.mode == 'generate-data':
        print("Generating synthetic healthcare data...")
        from src.data_generator import save_data
        save_data()
        return
    
    # Initialize system
    _, rag_pipeline = initialize_system()
    
    print("\n✅ System initialized successfully!\n")
    
    # Run selected mode
    if args.mode == 'interactive':
        interactive_mode(rag_pipeline)
    elif args.mode == 'demo':
        demo_mode(rag_pipeline)
    elif args.mode == 'eval':
        run_evaluation(rag_pipeline)


if __name__ == "__main__":
    main()
