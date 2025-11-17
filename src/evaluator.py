"""
Evaluation Module for Clinical RAG System

Provides metrics and evaluation functions to assess system performance.
"""
import json
from typing import List, Dict, Tuple
from pathlib import Path
import config


class RAGEvaluator:
    """Evaluate RAG system performance"""
    
    def __init__(self, rag_pipeline, llm):
        """
        Initialize evaluator.
        
        Args:
            rag_pipeline: ClinicalRAGPipeline instance
            llm: LLM for evaluation tasks
        """
        self.rag_pipeline = rag_pipeline
        self.llm = llm
        self.test_results = []
    
    def evaluate_retrieval_quality(self, query: str, expected_topics: List[str]) -> Dict:
        """
        Evaluate if retrieval captures expected topics.
        
        Args:
            query: Test query
            expected_topics: List of topics that should be retrieved
        
        Returns:
            Evaluation metrics
        """
        # Get retrieved documents
        results = self.rag_pipeline.vector_store.search(query, k=10)
        
        retrieved_content = " ".join([doc.page_content.lower() for doc, _ in results])
        
        # Check topic coverage
        topics_found = []
        topics_missed = []
        
        for topic in expected_topics:
            if topic.lower() in retrieved_content:
                topics_found.append(topic)
            else:
                topics_missed.append(topic)
        
        recall = len(topics_found) / len(expected_topics) if expected_topics else 0
        
        return {
            "query": query,
            "topics_expected": expected_topics,
            "topics_found": topics_found,
            "topics_missed": topics_missed,
            "recall": recall,
            "num_documents_retrieved": len(results)
        }
    
    def evaluate_response_quality(self, query: str, response: str, ground_truth: str = None) -> Dict:
        """
        Evaluate response quality using LLM-as-judge.
        
        Args:
            query: Original query
            response: System response
            ground_truth: Optional ground truth answer
        
        Returns:
            Quality scores
        """
        evaluation_prompt = f"""Evaluate this clinical AI response on the following criteria (score 1-5 for each):

Query: {query}
Response: {response}
{f"Expected Answer: {ground_truth}" if ground_truth else ""}

Rate on these dimensions:
1. **Clinical Accuracy**: Is the medical information correct?
2. **Completeness**: Does it fully address the query?
3. **Clarity**: Is it well-structured and understandable?
4. **Safety**: Does it appropriately prioritize patient safety?
5. **Evidence-Based**: Is it grounded in medical evidence?

Provide scores in JSON format:
{{
  "clinical_accuracy": <score>,
  "completeness": <score>,
  "clarity": <score>,
  "safety": <score>,
  "evidence_based": <score>,
  "justification": "<brief explanation>"
}}"""
        
        try:
            eval_response = self.llm.invoke(evaluation_prompt)
            eval_text = eval_response.content if hasattr(eval_response, 'content') else str(eval_response)
            
            # Try to parse JSON from response
            # Look for JSON block
            import re
            json_match = re.search(r'\{[^}]+\}', eval_text, re.DOTALL)
            if json_match:
                scores = json.loads(json_match.group())
            else:
                # Fallback to manual scoring
                scores = {
                    "clinical_accuracy": 3,
                    "completeness": 3,
                    "clarity": 3,
                    "safety": 3,
                    "evidence_based": 3,
                    "justification": "Could not parse structured scores"
                }
        except Exception as e:
            print(f"Error in evaluation: {e}")
            scores = {
                "clinical_accuracy": 0,
                "completeness": 0,
                "clarity": 0,
                "safety": 0,
                "evidence_based": 0,
                "justification": f"Evaluation error: {str(e)}"
            }
        
        # Calculate average score
        score_values = [v for k, v in scores.items() if k != "justification" and isinstance(v, (int, float))]
        scores["average_score"] = sum(score_values) / len(score_values) if score_values else 0
        
        return scores
    
    def check_hallucination(self, query: str, response: str, retrieved_context: str) -> Dict:
        """
        Check if response contains hallucinations.
        
        Args:
            query: Original query
            response: System response
            retrieved_context: Context that was retrieved
        
        Returns:
            Hallucination assessment
        """
        hallucination_prompt = f"""Analyze this clinical AI response for potential hallucinations or unsupported claims.

Retrieved Context:
{retrieved_context[:2000]}  # Limit context length

AI Response:
{response}

Answer these questions:
1. Are all factual claims supported by the retrieved context?
2. Are there specific numbers, dosages, or guidelines that aren't in the context?
3. Does the response add information beyond what was retrieved?

Respond in JSON format:
{{
  "is_grounded": <true/false>,
  "unsupported_claims": ["claim1", "claim2"],
  "hallucination_risk": "<low/medium/high>",
  "explanation": "<brief explanation>"
}}"""
        
        try:
            hal_response = self.llm.invoke(hallucination_prompt)
            hal_text = hal_response.content if hasattr(hal_response, 'content') else str(hal_response)
            
            # Try to parse JSON
            import re
            json_match = re.search(r'\{[^}]+\}', hal_text, re.DOTALL)
            if json_match:
                analysis = json.loads(json_match.group())
            else:
                analysis = {
                    "is_grounded": True,
                    "unsupported_claims": [],
                    "hallucination_risk": "unknown",
                    "explanation": "Could not parse analysis"
                }
        except Exception as e:
            analysis = {
                "is_grounded": True,
                "unsupported_claims": [],
                "hallucination_risk": "unknown",
                "explanation": f"Analysis error: {str(e)}"
            }
        
        return analysis
    
    def run_test_suite(self, test_cases: List[Dict]) -> Dict:
        """
        Run a suite of test cases.
        
        Args:
            test_cases: List of test case dictionaries with 'query' and optional 'expected_topics'
        
        Returns:
            Aggregated test results
        """
        results = []
        
        print(f"\n🧪 Running {len(test_cases)} test cases...\n")
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"Test {i}/{len(test_cases)}: {test_case['query'][:60]}...")
            
            # Run query
            response_dict = self.rag_pipeline.query(test_case['query'])
            
            # Evaluate retrieval
            if 'expected_topics' in test_case:
                retrieval_eval = self.evaluate_retrieval_quality(
                    test_case['query'],
                    test_case['expected_topics']
                )
            else:
                retrieval_eval = {}
            
            # Evaluate response quality
            quality_eval = self.evaluate_response_quality(
                test_case['query'],
                response_dict['response'],
                test_case.get('ground_truth')
            )
            
            # Check hallucination
            context = self.rag_pipeline.vector_store.get_relevant_context(test_case['query'])
            hallucination_check = self.check_hallucination(
                test_case['query'],
                response_dict['response'],
                context
            )
            
            # Compile results
            test_result = {
                "test_id": i,
                "query": test_case['query'],
                "response": response_dict['response'],
                "retrieval_metrics": retrieval_eval,
                "quality_scores": quality_eval,
                "hallucination_check": hallucination_check,
                "sources_used": response_dict['sources']
            }
            
            results.append(test_result)
            self.test_results.append(test_result)
            
            print(f"  ✓ Average quality score: {quality_eval.get('average_score', 0):.2f}/5")
            print(f"  ✓ Hallucination risk: {hallucination_check.get('hallucination_risk', 'unknown')}\n")
        
        # Calculate aggregate metrics
        avg_quality = sum(r['quality_scores']['average_score'] for r in results) / len(results)
        high_quality_count = sum(1 for r in results if r['quality_scores']['average_score'] >= 4.0)
        
        low_hallucination = sum(1 for r in results if r['hallucination_check'].get('hallucination_risk') == 'low')
        
        summary = {
            "total_tests": len(results),
            "average_quality_score": avg_quality,
            "high_quality_responses": high_quality_count,
            "high_quality_percentage": (high_quality_count / len(results)) * 100,
            "low_hallucination_count": low_hallucination,
            "detailed_results": results
        }
        
        return summary
    
    def save_results(self, filepath: Path = None):
        """Save test results to file"""
        if filepath is None:
            filepath = config.TESTS_DIR / "test_results.json"
        
        with open(filepath, 'w') as f:
            json.dump(self.test_results, f, indent=2)
        
        print(f"✓ Results saved to {filepath}")
    
    def generate_report(self) -> str:
        """Generate a text report of evaluation results"""
        if not self.test_results:
            return "No test results available."
        
        report_lines = [
            "=" * 60,
            "CLINICAL RAG SYSTEM - EVALUATION REPORT",
            "=" * 60,
            "",
            f"Total Tests Run: {len(self.test_results)}",
            ""
        ]
        
        # Calculate statistics
        avg_scores = {
            "clinical_accuracy": 0,
            "completeness": 0,
            "clarity": 0,
            "safety": 0,
            "evidence_based": 0
        }
        
        for result in self.test_results:
            for key in avg_scores.keys():
                avg_scores[key] += result['quality_scores'].get(key, 0)
        
        for key in avg_scores.keys():
            avg_scores[key] /= len(self.test_results)
        
        report_lines.append("AVERAGE QUALITY SCORES (out of 5):")
        for metric, score in avg_scores.items():
            report_lines.append(f"  {metric.replace('_', ' ').title()}: {score:.2f}")
        
        report_lines.append("")
        report_lines.append("HALLUCINATION ANALYSIS:")
        
        hallucination_counts = {"low": 0, "medium": 0, "high": 0, "unknown": 0}
        for result in self.test_results:
            risk = result['hallucination_check'].get('hallucination_risk', 'unknown')
            hallucination_counts[risk] = hallucination_counts.get(risk, 0) + 1
        
        for risk_level, count in hallucination_counts.items():
            percentage = (count / len(self.test_results)) * 100
            report_lines.append(f"  {risk_level.title()}: {count} ({percentage:.1f}%)")
        
        report_lines.append("")
        report_lines.append("=" * 60)
        
        return "\n".join(report_lines)


def create_test_cases() -> List[Dict]:
    """Create standard test cases for evaluation"""
    test_cases = [
        {
            "query": "What are the first-line treatments for Type 2 Diabetes?",
            "expected_topics": ["metformin", "lifestyle", "diet", "exercise"],
            "ground_truth": "Metformin is the first-line medication unless contraindicated. Lifestyle modifications including diet and exercise are essential."
        },
        {
            "query": "How should we manage a patient with newly diagnosed atrial fibrillation?",
            "expected_topics": ["anticoagulation", "CHA2DS2-VASc", "rate control", "rhythm control"],
            "ground_truth": "Assess stroke risk with CHA2DS2-VASc score. Consider anticoagulation if score >= 2. Implement rate or rhythm control strategy."
        },
        {
            "query": "What are the contraindications for metformin?",
            "expected_topics": ["renal function", "eGFR", "lactic acidosis", "liver disease"],
            "ground_truth": "Major contraindications include eGFR <30, severe liver disease, and conditions predisposing to lactic acidosis."
        },
        {
            "query": "What medications interact with warfarin?",
            "expected_topics": ["NSAIDs", "antibiotics", "INR", "bleeding risk"],
            "ground_truth": "Many medications interact with warfarin including NSAIDs, certain antibiotics, and amiodarone. Requires INR monitoring."
        },
        {
            "query": "What are the stages of chronic kidney disease?",
            "expected_topics": ["eGFR", "staging", "albuminuria", "nephrology referral"],
            "ground_truth": "CKD is staged 1-5 based on eGFR. Stage 5 is eGFR <15. Nephrology referral recommended for stage 4 or higher."
        }
    ]
    
    return test_cases
