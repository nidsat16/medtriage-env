from env.environment import MedTriageEnv
from env.tasks import grade
from env.models import Action, Diagnosis, TriageLevel, TestRecommendation

# Initialize
env = MedTriageEnv(seed=42)
obs = env.reset()
print(f"\n--- Patient Case ---")
print(f"Symptoms: {obs.current_symptoms.description}")

# Define Action
action = Action(
    diagnosis=Diagnosis.STROKE,
    triage=TriageLevel.EMERGENCY,
    recommended_test=TestRecommendation.CT_SCAN
)

# Run grading
reward = grade(action, task_number=1)
print(f"\n--- Results ---")
print(f"Score: {reward.total}")
print(f"Feedback: {reward.feedback}\n")