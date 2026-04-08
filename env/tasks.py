from models import (
    PatientHistory, Symptoms,
    Action, Reward, Diagnosis, TriageLevel, TestRecommendation
)

# ── TASK 1: EASY ──────────────────────────────────────────────────────────────
TASK_1_PATIENT = {
    "history": PatientHistory(
        age=68,
        gender="male",
        weight_kg=82.0,
        past_conditions=["hypertension"],
        current_medications=["aspirin"],
        family_history=["father had stroke"]
    ),
    "symptoms": Symptoms(
        description="sudden facial drooping on right side, slurred speech, cannot raise left arm",
        duration_minutes=20,
        severity=9
    )
}

TASK_1_ANSWER = {
    "diagnosis": Diagnosis.STROKE,
    "triage": TriageLevel.EMERGENCY,
    "test": TestRecommendation.CT_SCAN
}

# ── TASK 2: MEDIUM ────────────────────────────────────────────────────────────
TASK_2_PATIENT = {
    "history": PatientHistory(
        age=57,
        gender="female",
        weight_kg=91.5,
        past_conditions=["diabetes", "hypertension", "high cholesterol"],
        current_medications=["metformin", "lisinopril", "statins"],
        family_history=["mother had heart attack at 60"]
    ),
    "symptoms": Symptoms(
        description="chest tightness, shortness of breath, mild nausea, fatigue since morning",
        duration_minutes=90,
        severity=6
    )
}

TASK_2_ANSWER = {
    "diagnosis": Diagnosis.CARDIAC,
    "triage": TriageLevel.EMERGENCY,
    "test": TestRecommendation.ECG
}

# ── TASK 3: HARD ──────────────────────────────────────────────────────────────
TASK_3_PATIENT = {
    "history": PatientHistory(
        age=63,
        gender="male",
        weight_kg=88.0,
        past_conditions=["hypertension", "atrial fibrillation"],
        current_medications=["warfarin", "beta blockers"],
        family_history=["father had stroke", "brother had heart attack"]
    ),
    "symptoms": Symptoms(
        description="sudden dizziness, mild chest discomfort, slight confusion, headache",
        duration_minutes=35,
        severity=7
    )
}

TASK_3_ANSWER = {
    "diagnosis": Diagnosis.UNCLEAR,
    "triage": TriageLevel.MONITOR,
    "test": TestRecommendation.BLOOD_TEST
}

# ── GRADER ────────────────────────────────────────────────────────────────────
def grade(action: Action, task_number: int) -> Reward:
    answers = {
        1: TASK_1_ANSWER,
        2: TASK_2_ANSWER,
        3: TASK_3_ANSWER
    }
    correct = answers[task_number]

    diagnosis_score = 0.4 if action.diagnosis == correct["diagnosis"] else 0.0
    triage_score    = 0.35 if action.triage == correct["triage"] else 0.0
    test_score      = 0.25 if action.recommended_test == correct["test"] else 0.0

    penalty = 0.0
    if (correct["triage"] == TriageLevel.EMERGENCY and
        action.triage == TriageLevel.SAFE):
        penalty = -1.0

    total = max(0.0, diagnosis_score + triage_score + test_score + penalty)

    feedback_parts = []
    if diagnosis_score > 0: feedback_parts.append("✅ diagnosis correct")
    else: feedback_parts.append("❌ diagnosis wrong")
    if triage_score > 0: feedback_parts.append("✅ triage correct")
    else: feedback_parts.append("❌ triage wrong")
    if test_score > 0: feedback_parts.append("✅ test correct")
    else: feedback_parts.append("❌ test wrong")
    if penalty < 0: feedback_parts.append("⚠️ penalty: told critical patient to self care")

    return Reward(
        total=round(total, 2),
        diagnosis_score=diagnosis_score,
        triage_score=triage_score,
        test_score=test_score,
        penalty=penalty,
        feedback=" | ".join(feedback_parts)
    )