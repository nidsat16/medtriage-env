import os
import json
from openai import OpenAI
from env.environment import MedTriageEnv
from env.tasks import grade, TASK_1_PATIENT, TASK_2_PATIENT, TASK_3_PATIENT
from env.models import Action, Diagnosis, TriageLevel, TestRecommendation

# ── RUNTIME CONFIG (Nidhi's Requirement) ─────────────────────────────────────
# Judges will "inject" these variables during Phase 2 evaluation.
# We provide defaults so it still works for your local testing.

API_BASE = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")

# Authentication: Checks for both HF and OpenAI naming conventions
AUTH_TOKEN = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY") or "not-set"

client = OpenAI(
    base_url=API_BASE,
    api_key=AUTH_TOKEN
)

TASKS = [
    {"number": 1, "difficulty": "easy",   "patient": TASK_1_PATIENT},
    {"number": 2, "difficulty": "medium", "patient": TASK_2_PATIENT},
    {"number": 3, "difficulty": "hard",   "patient": TASK_3_PATIENT},
]

# ── PROMPT BUILDER ─────────────────────────────────────────────────────────────
def build_prompt(patient: dict) -> str:
    h = patient["history"]
    s = patient["symptoms"]
    return f"""You are an AI medical triage assistant.
Based on the patient history and symptoms, respond with ONLY a JSON object.

PATIENT HISTORY:
- Age: {h.age}, Gender: {h.gender}, Weight: {h.weight_kg}kg
- Conditions: {', '.join(h.past_conditions)}
- Meds: {', '.join(h.current_medications)}

SYMPTOMS:
- {s.description} (Duration: {s.duration_minutes} min, Severity: {s.severity}/10)

Your response must be a JSON object in this exact format:
{{
    "diagnosis": "stroke" | "cardiac" | "unclear",
    "triage": "call_ambulance" | "visit_doctor_today" | "monitor_at_home" | "self_care",
    "recommended_test": "ECG" | "CT_scan" | "blood_test" | "none"
}}"""

# ── PARSER ─────────────────────────────────────────────────────────────────────
def parse_response(response_text: str) -> Action:
    try:
        # Clean markdown code blocks if the model includes them
        clean = response_text.strip().replace("```json", "").replace("```", "").strip()
        data = json.loads(clean)
        return Action(
            diagnosis=Diagnosis(data["diagnosis"]),
            triage=TriageLevel(data["triage"]),
            recommended_test=TestRecommendation(data["recommended_test"])
        )
    except Exception as e:
        # Fallback to "Unclear" if the model fails to follow JSON format
        return Action(
            diagnosis=Diagnosis.UNCLEAR,
            triage=TriageLevel.MONITOR_AT_HOME,
            recommended_test=TestRecommendation.NONE
        )

# ── MAIN EVALUATION LOOP ───────────────────────────────────────────────────────
def run_evaluation():
    print("=" * 50)
    print("OPENENV PHASE 2: RUNTIME EVALUATION")
    print(f"Target Endpoint: {API_BASE}")
    print(f"Model Name:      {MODEL_NAME}")
    print("=" * 50)

    # Initialize the local environment
    env = MedTriageEnv(seed=42)
    total_score = 0.0

    for task in TASKS:
        print(f"\nEvaluating Task {task['number']} ({task['difficulty'].upper()})...")
        prompt = build_prompt(task["patient"])

        try:
            response = client.chat.completions.create(
                model=MODEL_NAME, 
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )

            result_text = response.choices[0].message.content
            action = parse_response(result_text)
            
            # Grade the AI's action using the built-in grader
            reward = grade(action, task_number=task["number"])
            
            print(f"  AI Diagnosis: {action.diagnosis.value}")
            print(f"  Task Reward:  {reward.total}")
            total_score += reward.total

        except Exception as e:
            print(f"  ❌ Runtime Error: {e}")

    avg_score = round(total_score / len(TASKS), 2)
    print("\n" + "=" * 50)
    print(f"FINAL EVALUATION AVERAGE: {avg_score}")
    print("=" * 50)

if __name__ == "__main__":
    run_evaluation()