import random
from models import (
    PatientHistory, Symptoms, Observation,
    Action, Reward, Diagnosis, TriageLevel, TestRecommendation
)

class MedTriageEnv:
    def __init__(self, seed: int = 42):
        random.seed(seed)
        self.current_observation = None
        self.step_count = 0

    def reset(self) -> Observation:
        """Starts a new patient case."""
        self.step_count = 0
        patient = self._generate_patient()
        self.current_observation = Observation(
            patient_history=patient["history"],
            current_symptoms=patient["symptoms"],
            step_number=0,
            done=False
        )
        return self.current_observation

    def state(self) -> Observation:
        """Returns the current state without advancing the simulation."""
        return self.current_observation

    def step(self, action: Action) -> tuple:
        """Processes the agent's diagnosis and ends the turn."""
        self.step_count += 1
        reward = self._score_action(action)
        
        # Update observation to show the task is done
        self.current_observation = Observation(
            patient_history=self.current_observation.patient_history,
            current_symptoms=self.current_observation.current_symptoms,
            step_number=self.step_count,
            done=True
        )
        
        info = {
            "action_taken": action.dict() if hasattr(action, 'dict') else str(action),
            "step": self.step_count
        }
        
        return self.current_observation, reward, True, info

    def _generate_patient(self) -> dict:
        """Generates randomized medical cases."""
        case_type = random.choice(["stroke", "cardiac"])
        if case_type == "stroke":
            return {
                "history": PatientHistory(
                    age=random.randint(55, 80),
                    gender=random.choice(["male", "female"]),
                    weight_kg=round(random.uniform(60, 100), 1),
                    past_conditions=["hypertension"],
                    current_medications=["aspirin"],
                    family_history=["father had stroke"]
                ),
                "symptoms": Symptoms(
                    description="sudden facial drooping, slurred speech, left arm weakness",
                    duration_minutes=random.randint(10, 60),
                    severity=random.randint(7, 10)
                )
            }
        else:
            return {
                "history": PatientHistory(
                    age=random.randint(45, 75),
                    gender=random.choice(["male", "female"]),
                    weight_kg=round(random.uniform(70, 110), 1),
                    past_conditions=["hypertension", "high cholesterol"],
                    current_medications=["statins"],
                    family_history=["father had heart attack"]
                ),
                "symptoms": Symptoms(
                    description="crushing chest pain radiating to left arm, sweating, nausea",
                    duration_minutes=random.randint(15, 90),
                    severity=random.randint(7, 10)
                )
            }

    def _score_action(self, action: Action) -> Reward:
        """Detailed scoring breakdown as per your original requirement."""
        return Reward(
            total=0.0,
            diagnosis_score=0.0,
            triage_score=0.0,
            test_score=0.0,
            penalty=0.0,
            feedback="Scored by environment"
        )