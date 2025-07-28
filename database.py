import json
import os
from datetime import datetime
from typing import Dict, List, Optional

class JSONDatabase:
    def __init__(self, db_file: str = "medai_data.json"):
        self.db_file = db_file
        self.data = self.load_data()
    
    def load_data(self) -> Dict:
        """Load data from JSON file or create empty structure."""
        if os.path.exists(self.db_file):
            try:
                with open(self.db_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        
        # Default empty structure
        return {
            "patients": {},
            "doctors": {},
            "appointments": [],
            "next_appointment_id": 1
        }
    
    def save_data(self):
        """Save data to JSON file."""
        with open(self.db_file, 'w') as f:
            json.dump(self.data, f, indent=2, default=str)
    
    def add_patient(self, email: str, name: str, medical_history: str = None, 
                   current_medication: str = None, current_symptoms: str = None) -> str:
        """Add a new patient or update existing one."""
        patient_data = {
            "name": name,
            "email": email,
            "medical_history": medical_history,
            "current_medication": current_medication,
            "current_symptoms": current_symptoms,
            "created_at": datetime.now().isoformat()
        }
        
        self.data["patients"][email] = patient_data
        self.save_data()
        return email
    
    def get_patient_by_email(self, email: str) -> Optional[Dict]:
        """Get patient information by email."""
        return self.data["patients"].get(email)
    
    def update_patient_symptoms(self, patient_email: str, current_symptoms: str) -> bool:
        """Update current symptoms for a patient."""
        if patient_email in self.data["patients"]:
            self.data["patients"][patient_email]["current_symptoms"] = current_symptoms
            self.save_data()
            return True
        return False
    
    def clear_patient_symptoms(self, patient_email: str) -> bool:
        """Clear current symptoms for a patient (after session ends)."""
        if patient_email in self.data["patients"]:
            self.data["patients"][patient_email]["current_symptoms"] = None
            self.save_data()
            return True
        return False
    
    def get_patient_medical_history(self, patient_email: str) -> List[Dict]:
        """Get patient's medical history."""
        patient = self.get_patient_by_email(patient_email)
        if patient and patient.get("medical_history"):
            return [{
                'id': 1,
                'patient_email': patient_email,
                'record_type': 'medical_history',
                'description': patient["medical_history"],
                'details': {'current_medication': patient.get("current_medication")},
                'date_recorded': datetime.now().date().isoformat(),
                'doctor_email': None,
                'is_active': True,
                'created_at': patient.get("created_at", datetime.now().isoformat())
            }]
        return []
    
    def add_doctor(self, email: str, name: str, specialization: str, days_available: str = None) -> str:
        """Add a doctor to the system."""
        doctor_data = {
            "name": name,
            "email": email,
            "specialization": specialization,
            "days_available": days_available,
            "created_at": datetime.now().isoformat()
        }
        
        self.data["doctors"][email] = doctor_data
        self.save_data()
        return email
    
    def get_doctors_by_specialization(self, specialization: str) -> List[Dict]:
        """Get doctors by specialization."""
        doctors = []
        for email, doctor in self.data["doctors"].items():
            if (specialization.lower() in doctor["specialization"].lower() or 
                doctor["specialization"] == "General Medicine"):
                doctors.append({
                    'id': email,
                    'email': email,
                    'name': doctor["name"],
                    'specialization': doctor["specialization"],
                    'days_available': doctor["days_available"],
                    'created_at': doctor["created_at"]
                })
        
        # Sort by exact match first, then by name
        doctors.sort(key=lambda x: (x["specialization"] != specialization, x["name"]))
        return doctors
    
    def create_appointment(self, patient_email: str, doctor_email: str, symptoms: str, 
                          appointment_date: datetime, google_event_id: str = None) -> int:
        """Create a new appointment."""
        appointment_id = self.data["next_appointment_id"]
        appointment_data = {
            "id": appointment_id,
            "patient_email": patient_email,
            "doctor_email": doctor_email,
            "symptoms": symptoms,
            "appointment_date": appointment_date.isoformat(),
            "status": "scheduled",
            "google_event_id": google_event_id,
            "created_at": datetime.now().isoformat()
        }
        
        self.data["appointments"].append(appointment_data)
        self.data["next_appointment_id"] += 1
        self.save_data()
        return appointment_id
    
    def get_all_patients(self) -> List[Dict]:
        """Get all patients."""
        return list(self.data["patients"].values())
    
    def get_all_doctors(self) -> List[Dict]:
        """Get all doctors."""
        return list(self.data["doctors"].values())
    
    def get_appointments(self) -> List[Dict]:
        """Get all appointments."""
        return self.data["appointments"]

# Create database instance
db = JSONDatabase()
