import os
import json
import time
from openai import OpenAI
from environment import MedTriageEnv
from tasks import grade, TASK_1_PATIENT, TASK_2_PATIENT, TASK_3_PATIENT
from models import Action, Diagnosis, TriageLevel, TestRecommendation

# ── REQUIRED ENVIRONMENT VARIABLES (Section 3 of Guidelines) ─────────────────
# Defaults are required for API_BASE_URL and MODEL_NAME
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

# ── INITIALIZE OPENAI CLIENT (Section 2 of Guidelines) ───────────────────────
client = OpenAI(
   
    base_url="https://router.huggingface.co/v1",
    api_key=f"{HF_TOKEN}"
)


TASKS = [
    {"id": "task_1", "number": 1, "patient": TASK_1_PATIENT},
    {"id": "task_2", "number": 2, "patient": TASK_2_PATIENT},
    {"id": "task_3", "number": 3, "patient": TASK_3_PATIENT},
]

# ── PROMPT BUILDER ──────────────────────────────────────────────────────────
def build_prompt(patient: dict) -> str:
    h = patient["history"]
    s = patient["symptoms"]
    return f"""You are an AI medical triage assistant.
Respond with ONLY a JSON object.

PATIENT: Age {h.age}, {h.gender}, {h.weight_kg}kg.
HISTORY: {', '.join(h.past_conditions)}.
SYMPTOMS: {s.description} (Duration: {s.duration_minutes} min, Severity: {s.severity}/10).

JSON Format:
{{
    "diagnosis": "stroke" | "cardiac" | "unclear",
    "triage": "call_ambulance" | "visit_doctor_today" | "monitor_at_home" | "self_care",
    "recommended_test": "ECG" | "CT_scan" | "blood_test" | "none"
}}"""

# ── PARSER ──────────────────────────────────────────────────────────────────
def parse_response(text: str) -> Action:
    try:
        clean = text.strip().replace("```json", "").replace("```", "").strip()
        data = json.loads(clean)
        return Action(
            diagnosis=Diagnosis(data["diagnosis"]),
            triage=TriageLevel(data["triage"]),
            recommended_test=TestRecommendation(data["recommended_test"])
        )
    except Exception:
        return Action(diagnosis=Diagnosis.UNCLEAR, triage=TriageLevel.MONITOR_AT_HOME, recommended_test=TestRecommendation.NONE)

# ── MAIN EVALUATION LOOP (Section 4: Mandatory Output Format) ───────────────
def run_evaluation():
    env_name = "medtriage-env"

    for task in TASKS:
        # [START] line
        print(f"[START] task={task['id']} env={env_name} model={MODEL_NAME}")
        
        prompt = build_prompt(task["patient"])
        
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME, 
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )

            result_text = response.choices[0].message.content
            action = parse_response(result_text)
            reward_obj = grade(action, task_number=task["number"])
            
            # Format outputs for strict compliance
            # Note: success/done must be lowercase 'true' or 'false'
            r_val = f"{reward_obj.total:.2f}"
            action_val = f"{action.diagnosis.value}_{action.triage.value}"
            
            # [STEP] line
            print(f"[STEP] step=1 action={action_val} reward={r_val} done=true error=null")
            
            # [END] line
            print(f"[END] success=true steps=1 rewards={r_val}")

        except Exception as e:
            err = str(e).replace(" ", "_")[:50]
            print(f"[STEP] step=1 action=error reward=0.00 done=true error={err}")
            print(f"[END] success=false steps=1 rewards=0.00")

if __name__ == "__main__":
    run_evaluation()



time.sleep(300) # This keeps the container "Running" for 5 minutes


    
