import os
from openai import OpenAI
from env.environment import MedTriageEnv
from env.tasks import grade, TASK_1_PATIENT, TASK_2_PATIENT, TASK_3_PATIENT
from env.models import Action, Diagnosis, TriageLevel, TestRecommendation
import json

# ── SETUP ─────────────────────────────────────────────────────────────────────
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

TASKS = [
    {"number": 1, "difficulty": "easy",   "patient": TASK_1_PATIENT},
    {"number": 2, "difficulty": "medium", "patient": TASK_2_PATIENT},
    {"number": 3, "difficulty": "hard",   "patient": TASK_3_PATIENT},
]

# ── PROMPT BUILDER ─────────────────────────────────────────────────────────────
def build_prompt(patient: dict) -> str:
    h = patient["history"]
    s = patient["symptoms"]
    return f"""You are an AI medical triage assistant for a home-based health app.
A patient has submitted their information. Based on their history and symptoms,
you must make three decisions.

PATIENT HISTORY:
- Age: {h.age}
- Gender: {h.gender}
- Weight: {h.weight_kg} kg
- Past conditions: {', '.join(h.past_conditions)}
- Current medications: {', '.join(h.current_medications)}
- Family history: {', '.join(h.family_history)}

CURRENT SYMPTOMS:
- Description: {s.description}
- Duration: {s.duration_minutes} minutes
- Severity: {s.severity}/10

You must respond with ONLY a JSON object in this exact format:
{{
    "diagnosis": "stroke" or "cardiac" or "unclear",
    "triage": "call_ambulance" or "visit_doctor_today" or "monitor_at_home" or "self_care",
    "recommended_test": "ECG" or "CT_scan" or "blood_test" or "none"
}}

No explanation. No extra text. Just the JSON object."""

# ── PARSER ─────────────────────────────────────────────────────────────────────
def parse_response(response_text: str) -> Action:
    try:
        # Clean up response
        clean = response_text.strip()
        data = json.loads(clean)

        return Action(
            diagnosis=Diagnosis(data["diagnosis"]),
            triage=TriageLevel(data["triage"]),
            recommended_test=TestRecommendation(data["recommended_test"])
        )
    except Exception as e:
        print(f"  ⚠️ Parse error: {e}")
        print(f"  Raw response: {response_text}")
        # Return a default action if parsing fails
        return Action(
            diagnosis=Diagnosis.UNCLEAR,
            triage=TriageLevel.MONITOR,
            recommended_test=TestRecommendation.NONE
        )

# ── MAIN RUNNER ────────────────────────────────────────────────────────────────
def run_baseline():
    print("=" * 50)
    print("MedTriage-Env Baseline Evaluation")
    print("=" * 50)

    env = MedTriageEnv(seed=42)
    total_score = 0.0
    results = []

    for task in TASKS:
        print(f"\nTask {task['number']} ({task['difficulty'].upper()})")
        print("-" * 30)

        # Set up patient
        patient = task["patient"]
        prompt = build_prompt(patient)

        # Call OpenAI API
        print("  Calling GPT-4...")
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        response_text = response.choices[0].message.content
        print(f"  Raw response: {response_text}")

        # Parse response into Action
        action = parse_response(response_text)
        print(f"  Diagnosis: {action.diagnosis.value}")
        print(f"  Triage: {action.triage.value}")
        print(f"  Test: {action.recommended_test.value}")

        # Grade the action
        reward = grade(action, task_number=task["number"])
        print(f"  Score: {reward.total}")
        print(f"  Feedback: {reward.feedback}")

        total_score += reward.total
        results.append({
            "task": task["number"],
            "difficulty": task["difficulty"],
            "score": reward.total,
            "feedback": reward.feedback
        })

    # Final summary
    avg_score = round(total_score / len(TASKS), 2)
    print("\n" + "=" * 50)
    print("FINAL RESULTS")
    print("=" * 50)
    for r in results:
        print(f"Task {r['task']} ({r['difficulty']}): {r['score']}")
    print(f"\nAverage Score: {avg_score}")
    print("=" * 50)

if __name__ == "__main__":
    run_baseline()