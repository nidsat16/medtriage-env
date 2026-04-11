import os
import json
import time
import traceback
import threading
import http.server
from openai import OpenAI
from environment import MedTriageEnv
from tasks import grade, TASK_1_PATIENT, TASK_2_PATIENT, TASK_3_PATIENT
from models import Action, Diagnosis, TriageLevel, TestRecommendation

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN", "").strip()

print(f"DEBUG URL: {API_BASE_URL}")
print(f"DEBUG MODEL: {MODEL_NAME}")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable is required")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

TASKS = [
    {"id": "task_1", "number": 1, "patient": TASK_1_PATIENT},
    {"id": "task_2", "number": 2, "patient": TASK_2_PATIENT},
    {"id": "task_3", "number": 3, "patient": TASK_3_PATIENT},
]

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
        return Action(diagnosis=Diagnosis.UNCLEAR, triage=TriageLevel.MONITOR, recommended_test=TestRecommendation.NONE)

def run_evaluation():
    env_name = "medtriage-env"
    for task in TASKS:
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
            r_val = f"{reward_obj.total:.2f}"
            action_val = f"{action.diagnosis.value}_{action.triage.value}"
            print(f"[STEP] step=1 action={action_val} reward={r_val} done=true error=null")
            print(f"[END] success=true steps=1 rewards={r_val}")
        except Exception as e:
            traceback.print_exc()
            err = str(e).replace(" ", "_")[:200]
            print(f"[STEP] step=1 action=error reward=0.00 done=true error={err}")
            print(f"[END] success=false steps=1 rewards=0.00")

def run_server():
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"OK")
        def log_message(self, format, *args):
            pass
    httpd = http.server.HTTPServer(("0.0.0.0", 7860), Handler)
    httpd.serve_forever()

if __name__ == "__main__":
    # Start HTTP server in background
    t = threading.Thread(target=run_server, daemon=True)
    t.start()
    print("HTTP server started on port 7860")
    
    # Run evaluation
    run_evaluation()
    print("Evaluation complete. Keeping container alive...")
    
    while True:
        time.sleep(3600)
