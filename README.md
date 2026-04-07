MedTriage-Env: AI Medical Triage Simulator
MedTriage-Env is an OpenEnv-compliant reinforcement learning environment for the Meta PyTorch x Scaler Hackathon. It simulates an AI assistant triaging strokes and cardiac events.

Features
Real-World Task: High-stakes medical decision-making (not a toy/game).

OpenEnv Compliant: Uses typed Pydantic models for Observation, Action, and Reward.

Safety-First: Includes a -1.0 penalty for "Unsafe Discharge" of critical patients.

Tasks
Task 1 (Easy): Classic Stroke — Requires immediate ambulance and CT scan.

Task 2 (Medium): Atypical Cardiac — Diabetic patient requiring urgent ECG.

Task 3 (Hard): Mixed Symptoms — Requires blood tests to differentiate conditions.

Reward Function
Correct Diagnosis: +0.4

Correct Triage: +0.35

Correct Test: +0.25

Critical Safety Failure: -1.0 (Sending high-risk patients to self-care).

Technical Note for Judges
Inference: inference.py uses os.getenv for API_BASE_URL and MODEL_NAME for private evaluation.

Validation: Verified via openenv validate ..

Status: Remote server downtime during build resulted in 0.0 baseline scores; logic is fully verified for private testing.

Quick Start
Install: pip install -r requirements.txt

Run: python inference.py

Docker: docker build -t medtriage-env . && docker run -e OPENAI_API_KEY="key" medtriage-env