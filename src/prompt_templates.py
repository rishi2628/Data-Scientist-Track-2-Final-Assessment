"""
Prompt Engineering Templates for Clinical Decision Support

This module contains carefully crafted prompts optimized for healthcare applications.
"""

# System prompt for clinical assistant
CLINICAL_ASSISTANT_SYSTEM_PROMPT = """You are an advanced clinical decision support AI assistant designed to help healthcare professionals 
make informed, evidence-based decisions. You have access to a comprehensive medical knowledge base and patient records.

CORE PRINCIPLES:
1. **Evidence-Based**: Base all responses on retrieved medical knowledge and patient data
2. **Patient Safety**: Prioritize patient safety in all recommendations
3. **Clinical Accuracy**: Provide precise, medically accurate information
4. **Transparency**: Clearly indicate when information is unavailable
5. **No Diagnosis**: Support clinical reasoning without making definitive diagnoses

RESPONSE GUIDELINES:
- Use clinical reasoning and cite relevant guidelines
- Highlight potential risks, contraindications, and drug interactions
- Suggest monitoring parameters and follow-up recommendations
- If information is insufficient, clearly state limitations
- Maintain professional medical terminology while being clear
- Always consider patient-specific factors (age, comorbidities, allergies)

Remember: You are a decision support tool to augment, not replace, clinical judgment."""


# Few-shot examples for clinical reasoning
FEW_SHOT_EXAMPLES = """
Example 1:
Query: "Should we start anticoagulation for this patient with new AFib?"
Reasoning: First, I need to assess stroke risk using CHA2DS2-VASc score and bleeding risk using HAS-BLED score. 
Based on the patient's age of 72, hypertension, and diabetes, the CHA2DS2-VASc score is 4 (high risk). 
The bleeding risk appears moderate. Therefore, anticoagulation is strongly indicated. DOACs are preferred over warfarin.
Response: "Anticoagulation is recommended. The patient's CHA2DS2-VASc score of 4 indicates high stroke risk. 
Consider apixaban 5mg twice daily or rivaroxaban 20mg daily. Monitor renal function and bleeding signs."

Example 2:
Query: "What labs should we order for this patient on metformin?"
Reasoning: Metformin requires monitoring of renal function (eGFR) because it's contraindicated in severe renal impairment 
and can cause lactic acidosis. Also monitor for vitamin B12 deficiency with long-term use. Check HbA1c for glycemic control.
Response: "For metformin monitoring: (1) Baseline and annual eGFR/creatinine, (2) HbA1c every 3-6 months, 
(3) Consider vitamin B12 annually, especially if on metformin >4 years or symptoms of neuropathy."

Example 3:
Query: "Patient has penicillin allergy. Can they take cephalosporins?"
Reasoning: Need to assess the type of penicillin reaction. IgE-mediated reactions (anaphylaxis, urticaria) have ~2% 
cross-reactivity with cephalosporins. Non-IgE reactions (rash) have minimal cross-reactivity. Later-generation 
cephalosporins have lower cross-reactivity.
Response: "Cross-reactivity between penicillin and cephalosporins is ~2%. Risk is higher with first-generation 
cephalosporins. If the penicillin allergy was anaphylaxis, avoid cephalosporins or use with caution under supervision. 
If it was a mild rash, third-generation cephalosporins are generally safe. Document the specific reaction."
"""


def create_rag_prompt(query: str, context: str, patient_context: str = None) -> str:
    """
    Create a comprehensive RAG prompt combining query, retrieved context, and patient information.
    
    Args:
        query: User's clinical question
        context: Retrieved medical knowledge and guidelines
        patient_context: Relevant patient information (optional)
    
    Returns:
        Formatted prompt for the LLM
    """
    prompt_parts = [CLINICAL_ASSISTANT_SYSTEM_PROMPT]
    
    # Add few-shot examples for better reasoning
    prompt_parts.append("\nEXAMPLES OF CLINICAL REASONING:")
    prompt_parts.append(FEW_SHOT_EXAMPLES)
    
    # Add retrieved context
    prompt_parts.append("\n=== RELEVANT MEDICAL KNOWLEDGE ===")
    prompt_parts.append(context)
    
    # Add patient-specific context if available
    if patient_context:
        prompt_parts.append("\n=== PATIENT INFORMATION ===")
        prompt_parts.append(patient_context)
    
    # Add the actual query with chain-of-thought instruction
    prompt_parts.append("\n=== CLINICAL QUERY ===")
    prompt_parts.append(query)
    
    prompt_parts.append("\n=== YOUR RESPONSE ===")
    prompt_parts.append("Let's approach this systematically:")
    prompt_parts.append("1. First, consider the relevant clinical context from the knowledge base")
    prompt_parts.append("2. Apply patient-specific factors if available")
    prompt_parts.append("3. Provide evidence-based recommendations")
    prompt_parts.append("4. Highlight any important considerations or precautions")
    prompt_parts.append("\nResponse:")
    
    return "\n".join(prompt_parts)


def create_patient_summary_prompt(patient_data: dict) -> str:
    """
    Create a concise patient summary from structured data.
    
    Args:
        patient_data: Dictionary containing patient information
    
    Returns:
        Formatted patient summary
    """
    summary_parts = []
    
    # Demographics
    summary_parts.append(f"Patient: {patient_data.get('name', 'Unknown')}")
    summary_parts.append(f"Age: {patient_data.get('age')} years, Gender: {patient_data.get('gender')}")
    summary_parts.append(f"MRN: {patient_data.get('medical_record_number')}")
    
    # Medical conditions
    if patient_data.get('conditions'):
        summary_parts.append(f"\nActive Conditions: {', '.join(patient_data['conditions'])}")
    
    # Current medications
    if patient_data.get('medications'):
        summary_parts.append("\nCurrent Medications:")
        for med in patient_data['medications']:
            summary_parts.append(f"  - {med}")
    
    # Allergies
    if patient_data.get('allergies'):
        summary_parts.append(f"\nAllergies: {', '.join(patient_data['allergies'])}")
    
    # Recent vitals
    if patient_data.get('vital_signs'):
        vitals = patient_data['vital_signs']
        summary_parts.append("\nRecent Vital Signs:")
        summary_parts.append(f"  - BP: {vitals.get('blood_pressure')} mmHg")
        summary_parts.append(f"  - HR: {vitals.get('heart_rate')} bpm")
        summary_parts.append(f"  - Temp: {vitals.get('temperature')}°F")
        summary_parts.append(f"  - SpO2: {vitals.get('oxygen_saturation')}%")
    
    # Recent labs
    if patient_data.get('lab_results'):
        summary_parts.append("\nRecent Lab Results:")
        for lab_name, lab_data in list(patient_data['lab_results'].items())[:5]:  # Show first 5
            summary_parts.append(f"  - {lab_name}: {lab_data['value']} {lab_data['unit']}")
    
    return "\n".join(summary_parts)


def create_retrieval_query(user_query: str) -> str:
    """
    Optimize user query for better retrieval from vector store.
    
    Args:
        user_query: Original user question
    
    Returns:
        Optimized query for retrieval
    """
    # Add medical context keywords to improve retrieval
    optimization_prefix = "Medical knowledge about: "
    return optimization_prefix + user_query


# Template for evaluating response quality
EVALUATION_PROMPT = """Evaluate the following clinical AI response on these criteria:

1. **Clinical Accuracy** (1-5): Is the medical information correct?
2. **Evidence-Based** (1-5): Is the response grounded in medical evidence?
3. **Safety** (1-5): Does it prioritize patient safety?
4. **Completeness** (1-5): Does it address all aspects of the question?
5. **Clarity** (1-5): Is it clear and well-structured?

Query: {query}
Response: {response}
Retrieved Context: {context}

Provide scores and brief justification for each criterion."""


# Template for identifying hallucinations
HALLUCINATION_CHECK_PROMPT = """Review this clinical AI response for potential hallucinations or unsupported claims.

Retrieved Context:
{context}

AI Response:
{response}

Questions to answer:
1. Are all factual claims in the response supported by the retrieved context?
2. Are there any specific medical recommendations not grounded in the provided information?
3. Are there any fabricated statistics, dosages, or guidelines?
4. Overall assessment: Does this response stay faithful to the source material?

Provide a detailed analysis."""
