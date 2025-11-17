"""
Synthetic Healthcare Data Generator

Generates realistic patient records, medical knowledge documents,
and clinical guidelines for the RAG system.
"""
import json
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict
import config

# Medical data templates
FIRST_NAMES = [
    "James", "Mary", "John", "Patricia", "Robert", "Jennifer", "Michael", "Linda",
    "William", "Elizabeth", "David", "Barbara", "Richard", "Susan", "Joseph", "Jessica",
    "Thomas", "Sarah", "Charles", "Karen", "Christopher", "Nancy", "Daniel", "Lisa",
    "Matthew", "Betty", "Anthony", "Margaret", "Mark", "Sandra", "Donald", "Ashley"
]

LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
    "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson",
    "Thomas", "Taylor", "Moore", "Jackson", "Martin", "Lee", "Thompson", "White"
]

MEDICAL_CONDITIONS = [
    "Type 2 Diabetes Mellitus",
    "Hypertension",
    "Coronary Artery Disease",
    "Chronic Obstructive Pulmonary Disease",
    "Asthma",
    "Atrial Fibrillation",
    "Congestive Heart Failure",
    "Chronic Kidney Disease",
    "Osteoarthritis",
    "Depression",
    "Anxiety Disorder",
    "Hyperlipidemia",
    "Gastroesophageal Reflux Disease",
    "Hypothyroidism",
    "Osteoporosis"
]

MEDICATIONS = {
    "Type 2 Diabetes Mellitus": ["Metformin 500mg", "Glipizide 5mg", "Insulin Glargine 10 units"],
    "Hypertension": ["Lisinopril 10mg", "Amlodipine 5mg", "Losartan 50mg", "Hydrochlorothiazide 25mg"],
    "Coronary Artery Disease": ["Aspirin 81mg", "Atorvastatin 20mg", "Metoprolol 50mg"],
    "Chronic Obstructive Pulmonary Disease": ["Albuterol inhaler", "Tiotropium inhaler", "Prednisone 5mg"],
    "Asthma": ["Albuterol inhaler", "Fluticasone inhaler", "Montelukast 10mg"],
    "Atrial Fibrillation": ["Apixaban 5mg", "Metoprolol 25mg", "Digoxin 0.125mg"],
    "Congestive Heart Failure": ["Furosemide 40mg", "Carvedilol 12.5mg", "Spironolactone 25mg"],
    "Chronic Kidney Disease": ["Erythropoietin", "Calcium carbonate", "Sodium bicarbonate"],
    "Osteoarthritis": ["Acetaminophen 500mg", "Ibuprofen 400mg", "Celecoxib 200mg"],
    "Depression": ["Sertraline 50mg", "Escitalopram 10mg", "Bupropion 150mg"],
    "Anxiety Disorder": ["Buspirone 10mg", "Alprazolam 0.5mg", "Escitalopram 10mg"],
    "Hyperlipidemia": ["Atorvastatin 20mg", "Rosuvastatin 10mg", "Ezetimibe 10mg"],
    "Gastroesophageal Reflux Disease": ["Omeprazole 20mg", "Pantoprazole 40mg", "Famotidine 20mg"],
    "Hypothyroidism": ["Levothyroxine 100mcg", "Levothyroxine 75mcg"],
    "Osteoporosis": ["Alendronate 70mg weekly", "Calcium/Vitamin D", "Denosumab injection"]
}

ALLERGIES = [
    "Penicillin", "Sulfa drugs", "Aspirin", "Iodine contrast", "Latex",
    "Codeine", "Morphine", "NSAIDs", "Shellfish", "None known"
]

VITAL_SIGNS_RANGES = {
    "blood_pressure_systolic": (110, 180),
    "blood_pressure_diastolic": (70, 110),
    "heart_rate": (60, 100),
    "temperature": (97.0, 99.5),
    "respiratory_rate": (12, 20),
    "oxygen_saturation": (92, 100)
}

LAB_VALUES = {
    "Hemoglobin A1C": (5.0, 12.0, "%"),
    "Fasting Glucose": (70, 250, "mg/dL"),
    "Total Cholesterol": (150, 300, "mg/dL"),
    "LDL Cholesterol": (70, 200, "mg/dL"),
    "HDL Cholesterol": (30, 80, "mg/dL"),
    "Triglycerides": (50, 300, "mg/dL"),
    "Creatinine": (0.6, 2.5, "mg/dL"),
    "eGFR": (30, 120, "mL/min"),
    "TSH": (0.5, 8.0, "mIU/L"),
    "Potassium": (3.5, 5.5, "mmol/L"),
    "Sodium": (135, 145, "mmol/L")
}


def generate_patient_records(num_patients: int = 50) -> List[Dict]:
    """Generate synthetic patient records"""
    patients = []
    
    for i in range(num_patients):
        patient_id = f"PT{str(i+1).zfill(5)}"
        age = random.randint(25, 85)
        gender = random.choice(["Male", "Female"])
        
        # Generate conditions (1-4 per patient)
        num_conditions = random.randint(1, 4)
        conditions = random.sample(MEDICAL_CONDITIONS, num_conditions)
        
        # Generate medications based on conditions
        medications = []
        for condition in conditions:
            if condition in MEDICATIONS:
                meds = random.sample(MEDICATIONS[condition], min(2, len(MEDICATIONS[condition])))
                medications.extend(meds)
        
        # Generate allergies
        allergies = [random.choice(ALLERGIES)]
        
        # Generate vital signs
        vitals = {
            "blood_pressure": f"{random.randint(*VITAL_SIGNS_RANGES['blood_pressure_systolic'])}/{random.randint(*VITAL_SIGNS_RANGES['blood_pressure_diastolic'])}",
            "heart_rate": random.randint(*VITAL_SIGNS_RANGES['heart_rate']),
            "temperature": round(random.uniform(*VITAL_SIGNS_RANGES['temperature']), 1),
            "respiratory_rate": random.randint(*VITAL_SIGNS_RANGES['respiratory_rate']),
            "oxygen_saturation": random.randint(*VITAL_SIGNS_RANGES['oxygen_saturation'])
        }
        
        # Generate lab results
        labs = {}
        for lab_name, (min_val, max_val, unit) in LAB_VALUES.items():
            value = round(random.uniform(min_val, max_val), 2)
            labs[lab_name] = {"value": value, "unit": unit}
        
        # Generate visit history
        last_visit = datetime.now() - timedelta(days=random.randint(1, 180))
        
        patient = {
            "patient_id": patient_id,
            "name": f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}",
            "age": age,
            "gender": gender,
            "date_of_birth": (datetime.now() - timedelta(days=age*365)).strftime("%Y-%m-%d"),
            "medical_record_number": f"MRN{random.randint(100000, 999999)}",
            "conditions": conditions,
            "medications": medications,
            "allergies": allergies,
            "vital_signs": vitals,
            "lab_results": labs,
            "last_visit": last_visit.strftime("%Y-%m-%d"),
            "primary_care_physician": f"Dr. {random.choice(LAST_NAMES)}",
            "insurance": random.choice(["Medicare", "Medicaid", "Private Insurance", "Blue Cross Blue Shield"])
        }
        
        patients.append(patient)
    
    return patients


def generate_medical_knowledge() -> List[Dict]:
    """Generate medical knowledge base documents"""
    knowledge_docs = []
    
    # Disease information
    disease_info = [
        {
            "title": "Type 2 Diabetes Mellitus - Overview",
            "category": "Endocrinology",
            "content": """Type 2 Diabetes Mellitus is a chronic metabolic disorder characterized by insulin resistance 
            and relative insulin deficiency. Key points:
            
            Diagnosis: Fasting glucose ≥126 mg/dL or HbA1c ≥6.5% or random glucose ≥200 mg/dL with symptoms.
            
            First-line treatment: Metformin 500-2000mg daily, unless contraindicated.
            
            Target HbA1c: Generally <7% for most patients, individualize based on age and comorbidities.
            
            Monitoring: HbA1c every 3 months if not at goal, every 6 months if stable. Annual screening for 
            complications including retinopathy, nephropathy, and neuropathy.
            
            Complications: Cardiovascular disease, chronic kidney disease, retinopathy, neuropathy, foot ulcers."""
        },
        {
            "title": "Hypertension Management Guidelines",
            "category": "Cardiology",
            "content": """Hypertension is defined as sustained blood pressure ≥130/80 mmHg.
            
            Classification:
            - Normal: <120/80 mmHg
            - Elevated: 120-129/<80 mmHg
            - Stage 1: 130-139/80-89 mmHg
            - Stage 2: ≥140/90 mmHg
            
            First-line medications: ACE inhibitors, ARBs, calcium channel blockers, or thiazide diuretics.
            
            Target BP: <130/80 mmHg for most adults. <140/90 mmHg for adults ≥65 years.
            
            Lifestyle modifications: DASH diet, sodium restriction (<2g/day), regular exercise, weight loss, 
            limit alcohol consumption.
            
            Monitoring: Home BP monitoring recommended. Office visits every 3-6 months once stable."""
        },
        {
            "title": "Atrial Fibrillation - Anticoagulation Strategy",
            "category": "Cardiology",
            "content": """Atrial fibrillation increases stroke risk 5-fold. Anticoagulation decision based on CHA2DS2-VASc score:
            
            CHA2DS2-VASc Score:
            - Congestive heart failure: 1 point
            - Hypertension: 1 point
            - Age ≥75: 2 points
            - Diabetes: 1 point
            - Stroke/TIA/thromboembolism: 2 points
            - Vascular disease: 1 point
            - Age 65-74: 1 point
            - Sex category (female): 1 point
            
            Score 0 (males) or 1 (females): No anticoagulation
            Score 1 (males): Consider anticoagulation
            Score ≥2: Anticoagulation recommended
            
            Preferred anticoagulants: DOACs (apixaban, rivaroxaban, edoxaban, dabigatran) over warfarin.
            
            Bleeding risk: Assess using HAS-BLED score. High bleeding risk doesn't preclude anticoagulation 
            but requires addressing modifiable risk factors."""
        },
        {
            "title": "Chronic Kidney Disease - Staging and Management",
            "category": "Nephrology",
            "content": """CKD is classified by eGFR and albuminuria status:
            
            Stages:
            - Stage 1: eGFR ≥90 with kidney damage
            - Stage 2: eGFR 60-89 with kidney damage
            - Stage 3a: eGFR 45-59
            - Stage 3b: eGFR 30-44
            - Stage 4: eGFR 15-29
            - Stage 5: eGFR <15 or on dialysis
            
            Management priorities:
            1. Control blood pressure (target <130/80, <120/80 if albuminuria)
            2. Use ACE inhibitors or ARBs if albuminuria present
            3. Glycemic control if diabetic (HbA1c <7%)
            4. Avoid nephrotoxins (NSAIDs, contrast when possible)
            5. Monitor electrolytes, especially potassium
            6. Refer to nephrology when eGFR <30 or rapidly declining
            
            Complications: Anemia (treat with ESAs/iron), mineral bone disorder, metabolic acidosis."""
        },
        {
            "title": "Heart Failure - Guideline-Directed Medical Therapy",
            "category": "Cardiology",
            "content": """Heart failure with reduced ejection fraction (HFrEF) requires comprehensive medical therapy:
            
            Four pillars of GDMT:
            1. ACE inhibitor/ARB/ARNI (e.g., sacubitril-valsartan)
            2. Beta-blocker (carvedilol, metoprolol succinate, or bisoprolol)
            3. Mineralocorticoid receptor antagonist (spironolactone or eplerenone)
            4. SGLT2 inhibitor (dapagliflozin or empagliflozin)
            
            Additional therapies:
            - Loop diuretic for volume management
            - Hydralazine-nitrate combination for African Americans
            - Ivabradine if heart rate >70 despite beta-blocker
            
            Device therapy: Consider ICD for primary prevention if EF ≤35%. CRT if EF ≤35%, 
            LBBB, and QRS ≥150ms.
            
            Monitoring: Daily weights, sodium restriction, fluid restriction if hyponatremia."""
        }
    ]
    
    knowledge_docs.extend(disease_info)
    
    # Drug interactions
    drug_interactions = [
        {
            "title": "Common Drug Interactions - Cardiovascular Medications",
            "category": "Pharmacology",
            "content": """Important drug interactions to monitor:
            
            ACE Inhibitors + Potassium-sparing diuretics: Risk of hyperkalemia. Monitor potassium closely.
            
            Beta-blockers + Calcium channel blockers (diltiazem/verapamil): Risk of bradycardia and heart block. 
            Avoid combination or use with extreme caution.
            
            Warfarin + Multiple interactions: NSAIDs (bleeding risk), antibiotics (INR changes), 
            amiodarone (INR increase). Requires frequent INR monitoring.
            
            Statins + Gemfibrozil: Increased myopathy risk. Avoid combination; use fenofibrate if needed.
            
            Digoxin + Amiodarone: Digoxin toxicity. Reduce digoxin dose by 50%.
            
            DOACs + Strong P-gp inhibitors: Increased bleeding risk. Reduce DOAC dose or avoid."""
        },
        {
            "title": "Diabetes Medication - Adverse Effects and Monitoring",
            "category": "Endocrinology",
            "content": """Key monitoring points for diabetes medications:
            
            Metformin:
            - Adverse effects: GI upset (dose-related), lactic acidosis (rare)
            - Contraindications: eGFR <30, severe liver disease
            - Monitoring: Renal function annually, B12 levels periodically
            
            Sulfonylureas:
            - Adverse effects: Hypoglycemia, weight gain
            - Caution: Elderly, renal impairment
            - Monitoring: Glucose levels
            
            SGLT2 Inhibitors:
            - Adverse effects: Genital mycotic infections, DKA (rare), Fournier's gangrene (very rare)
            - Benefits: Cardiovascular and renal protection
            - Monitoring: Volume status, foot care
            
            GLP-1 Agonists:
            - Adverse effects: Nausea, pancreatitis (rare)
            - Benefits: Weight loss, cardiovascular protection
            - Contraindications: Personal/family history of medullary thyroid cancer or MEN2"""
        }
    ]
    
    knowledge_docs.extend(drug_interactions)
    
    # Clinical guidelines
    clinical_guidelines = [
        {
            "title": "Preventive Care Guidelines - Adult Screening",
            "category": "Preventive Medicine",
            "content": """Evidence-based screening recommendations:
            
            Cardiovascular:
            - Blood pressure: All adults annually
            - Lipid panel: Men ≥35, women ≥45, or earlier if risk factors
            - Diabetes screening: Adults age 35-70 with BMI ≥25
            
            Cancer Screening:
            - Colorectal: Age 45-75 (colonoscopy every 10 years or FIT annually)
            - Breast: Mammography every 1-2 years, age 50-74
            - Cervical: Pap smear every 3 years age 21-65, or HPV testing every 5 years age 30-65
            - Lung: Low-dose CT annually for age 50-80 with 20 pack-year history
            
            Infectious Disease:
            - Hepatitis C: One-time screening for adults born 1945-1965
            - HIV: At least once for everyone age 15-65
            
            Immunizations:
            - Influenza: Annually for all adults
            - Pneumococcal: Age ≥65 (PCV15/PCV20 or PPSV23)
            - Shingles: Age ≥50 (Shingrix, 2 doses)
            - Tdap: Once, then Td booster every 10 years"""
        }
    ]
    
    knowledge_docs.extend(clinical_guidelines)
    
    return knowledge_docs


def generate_clinical_guidelines() -> List[Dict]:
    """Generate clinical practice guidelines"""
    guidelines = [
        {
            "title": "Acute Coronary Syndrome - Management Protocol",
            "organization": "American Heart Association",
            "year": 2023,
            "content": """ACS Management Protocol:
            
            IMMEDIATE (ED):
            1. ECG within 10 minutes of arrival
            2. Aspirin 324mg chewed (unless contraindicated)
            3. Nitroglycerin sublingual if chest pain
            4. Oxygen if O2 sat <90%
            5. Cardiac biomarkers (troponin)
            
            STEMI Protocol:
            - Goal: PCI within 90 minutes (door-to-balloon time)
            - If PCI not available: Fibrinolysis within 30 minutes
            - Dual antiplatelet therapy: Aspirin + P2Y12 inhibitor
            - Anticoagulation: Heparin or bivalirudin
            
            NSTEMI/Unstable Angina:
            - Risk stratification (HEART or TIMI score)
            - Early invasive strategy if high risk
            - Medical management: Aspirin, P2Y12 inhibitor, anticoagulation, beta-blocker, statin
            
            POST-ACS:
            - Cardiac rehab referral
            - Secondary prevention: High-intensity statin, ACE inhibitor, beta-blocker
            - Dual antiplatelet therapy for 12 months"""
        },
        {
            "title": "Sepsis Recognition and Management - Surviving Sepsis Campaign",
            "organization": "Society of Critical Care Medicine",
            "year": 2023,
            "content": """Sepsis Management Guidelines:
            
            Recognition (qSOFA - 2 of 3):
            - Altered mental status
            - Respiratory rate ≥22
            - Systolic BP ≤100 mmHg
            
            HOUR 1 BUNDLE:
            1. Measure lactate level (repeat if >2 mmol/L)
            2. Obtain blood cultures before antibiotics
            3. Administer broad-spectrum antibiotics
            4. Rapid administration of 30 mL/kg crystalloid for hypotension or lactate ≥4
            5. Apply vasopressors if hypotensive during or after fluid resuscitation (MAP ≥65)
            
            Antibiotic Selection:
            - Consider source, local resistance patterns
            - De-escalate based on cultures and clinical improvement
            - Typical duration: 7-10 days
            
            Source Control:
            - Identify and control source within 12 hours
            - Examples: Drain abscess, remove infected device
            
            Supportive Care:
            - Mechanical ventilation if ARDS: Low tidal volume (6 mL/kg IBW)
            - Glycemic control: Target <180 mg/dL
            - DVT prophylaxis, stress ulcer prophylaxis"""
        }
    ]
    
    return guidelines


def save_data():
    """Generate and save all synthetic data"""
    print("Generating synthetic healthcare data...")
    
    # Generate data
    patients = generate_patient_records(config.NUM_PATIENTS)
    medical_knowledge = generate_medical_knowledge()
    clinical_guidelines = generate_clinical_guidelines()
    
    # Save to JSON files
    print(f"Saving {len(patients)} patient records...")
    with open(config.DATA_DIR / "patient_records.json", "w") as f:
        json.dump(patients, f, indent=2)
    
    print(f"Saving {len(medical_knowledge)} medical knowledge documents...")
    with open(config.DATA_DIR / "medical_knowledge.json", "w") as f:
        json.dump(medical_knowledge, f, indent=2)
    
    print(f"Saving {len(clinical_guidelines)} clinical guidelines...")
    with open(config.DATA_DIR / "clinical_guidelines.json", "w") as f:
        json.dump(clinical_guidelines, f, indent=2)
    
    print("✓ Data generation complete!")
    print(f"  - Patient records: {config.DATA_DIR / 'patient_records.json'}")
    print(f"  - Medical knowledge: {config.DATA_DIR / 'medical_knowledge.json'}")
    print(f"  - Clinical guidelines: {config.DATA_DIR / 'clinical_guidelines.json'}")


if __name__ == "__main__":
    save_data()
