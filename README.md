# MedTriage-Env

A home-based AI triage environment where an agent reads a
patient's full medical history and current symptoms, then decides:
1. What condition they likely have
2. How urgently they need care
3. What diagnostic test to recommend

Built for the OpenEnv specification.

## Environment Description
Simulates a home health AI assistant triaging stroke and cardiac
events based on patient history and symptoms.

## Observation Space
- patient_history: age, gender, weight, past conditions, medications, family history
- current_symptoms: description, duration, severity
- step_number: current step
- done: episode complete

## Action Space
- diagnosis: stroke | cardiac | unclear
- triage: call_ambulance | visit_doctor_today | monitor_at_home | self_care
- recommended_test: ECG | CT_scan | blood_test | none

## Tasks
| Task | Difficulty | Description |
|------|------------|-------------|
| 1 | Easy | Classic stroke — obvious symptoms |
| 2 | Medium | Cardiac event with overlapping diabetic symptoms |
| 3 | Hard | Mixed symptoms requiring blood test first |

## Reward Function
- Diagnosis correct: +0.4
- Triage correct: +0.35
- Test correct: +0.25
- Telling critical patient to self-care: -1.0

## Baseline Scores
| Task | Score |
|------|-------|
| Task 1 (Easy) | TBD |
| Task 2 (Medium) | TBD |
| Task 3 (Hard) | TBD |

## Setup

### Install dependencies
pip install -r requirements.txt

### Set your OpenAI API key
export OPENAI_API_KEY=your_key_here

### Run baseline
python baseline.py

### Docker
docker build -t medtriage-env .
docker run -e OPENAI_API_KEY=your_key medtriage-env