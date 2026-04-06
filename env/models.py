from pydantic import BaseModel
from typing import List
from enum import Enum

class Diagnosis(str, Enum):
    STROKE = "stroke"
    CARDIAC = "cardiac"
    UNCLEAR = "unclear"

class TriageLevel(str, Enum):
    EMERGENCY = "call_ambulance"
    URGENT = "visit_doctor_today"
    MONITOR = "monitor_at_home"
    SAFE = "self_care"

class TestRecommendation(str, Enum):
    ECG = "ECG"
    CT_SCAN = "CT_scan"
    BLOOD_TEST = "blood_test"
    NONE = "none"

class PatientHistory(BaseModel):
    age: int
    gender: str
    weight_kg: float
    past_conditions: List[str]
    current_medications: List[str]
    family_history: List[str]

class Symptoms(BaseModel):
    description: str
    duration_minutes: int
    severity: int

class Observation(BaseModel):
    patient_history: PatientHistory
    current_symptoms: Symptoms
    step_number: int
    done: bool

class Action(BaseModel):
    diagnosis: Diagnosis
    triage: TriageLevel
    recommended_test: TestRecommendation

class Reward(BaseModel):
    total: float
    diagnosis_score: float
    triage_score: float
    test_score: float
    penalty: float
    feedback: str
    
