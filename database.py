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
                    data = json.load(f)
                    # Ensure all required keys exist
                    if "next_patient_id" not in data:
                        data["next_patient_id"] = 1
                    if "next_doctor_id" not in data:
                        data["next_doctor_id"] = 1
                    if "next_appointment_id" not in data:
                        data["next_appointment_id"] = 1
                    if "post_visit_records" not in data:
                        data["post_visit_records"] = []
                    if "next_visit_record_id" not in data:
                        data["next_visit_record_id"] = 1
                    
                    # Migration: Add appointment_completed flag to existing appointments
                    if "appointments" in data:
                        for appointment in data["appointments"]:
                            if "appointment_completed" not in appointment:
                                appointment["appointment_completed"] = False
                    
                    return data
            except:
                pass
        
        # Default empty structure with ID counters
        return {
            "patients": {},
            "doctors": {},
            "appointments": [],
            "post_visit_records": [],  # New table for post-visit data
            "next_patient_id": 1,
            "next_doctor_id": 1,
            "next_appointment_id": 1,
            "next_visit_record_id": 1  # New ID counter for visit records
        }
    
    def save_data(self):
        """Save data to JSON file."""
        with open(self.db_file, 'w') as f:
            json.dump(self.data, f, indent=2, default=str)
    
    def add_patient(self, email: str, name: str, medical_history: str = None, 
                   current_medication: str = None, current_symptoms: str = None) -> int:
        """Add a new patient or update existing one. Returns patient ID."""
        # Check if patient already exists by email
        existing_patient = self.get_patient_by_email(email)
        if existing_patient:
            # Update existing patient
            patient_id = existing_patient["id"]
            patient_data = {
                "id": patient_id,
                "name": name,
                "email": email,
                "role": "Patient",
                "medical_history": medical_history or existing_patient.get("medical_history"),
                "current_medication": current_medication or existing_patient.get("current_medication"),
                "current_symptoms": current_symptoms or existing_patient.get("current_symptoms"),
                "created_at": existing_patient.get("created_at", datetime.now().isoformat())
            }
            self.data["patients"][str(patient_id)] = patient_data
            self.save_data()
            return patient_id
        else:
            # Create new patient
            patient_id = self.data["next_patient_id"]
            patient_data = {
                "id": patient_id,
                "name": name,
                "email": email,
                "role": "Patient",
                "medical_history": medical_history,
                "current_medication": current_medication,
                "current_symptoms": current_symptoms,
                "created_at": datetime.now().isoformat()
            }
            
            self.data["patients"][str(patient_id)] = patient_data
            self.data["next_patient_id"] += 1
            self.save_data()
            return patient_id
    
    def add_doctor(self, email: str, name: str, specialization: str, days_available: str = None) -> int:
        """Add a doctor to the system. Returns doctor ID."""
        # Check if doctor already exists by email
        existing_doctor = self.get_doctor_by_email(email)
        if existing_doctor:
            # Update existing doctor
            doctor_id = existing_doctor["id"]
            doctor_data = {
                "id": doctor_id,
                "name": name,
                "email": email,
                "role": "Doctor",
                "specialization": specialization,
                "days_available": days_available or existing_doctor.get("days_available"),
                "created_at": existing_doctor.get("created_at", datetime.now().isoformat())
            }
            self.data["doctors"][str(doctor_id)] = doctor_data
            self.save_data()
            return doctor_id
        else:
            # Create new doctor
            doctor_id = self.data["next_doctor_id"]
            doctor_data = {
                "id": doctor_id,
                "name": name,
                "email": email,
                "role": "Doctor",
                "specialization": specialization,
                "days_available": days_available,
                "created_at": datetime.now().isoformat()
            }
            
            self.data["doctors"][str(doctor_id)] = doctor_data
            self.data["next_doctor_id"] += 1
            self.save_data()
            return doctor_id
    
    def get_patient_by_email(self, email: str) -> Optional[Dict]:
        """Get patient information by email."""
        for patient in self.data["patients"].values():
            if patient["email"] == email:
                return patient
        return None
    
    def get_patient_by_id(self, patient_id: int) -> Optional[Dict]:
        """Get patient information by ID."""
        return self.data["patients"].get(str(patient_id))

    def get_doctor_by_email(self, email: str) -> Optional[Dict]:
        """Get doctor information by email."""
        for doctor in self.data["doctors"].values():
            if doctor["email"] == email:
                return doctor
        return None
    
    def get_doctor_by_id(self, doctor_id: int) -> Optional[Dict]:
        """Get doctor information by ID."""
        return self.data["doctors"].get(str(doctor_id))
    
    def update_patient_symptoms(self, patient_email: str, current_symptoms: str) -> bool:
        """Update current symptoms for a patient."""
        patient = self.get_patient_by_email(patient_email)
        if patient:
            patient["current_symptoms"] = current_symptoms
            self.data["patients"][str(patient["id"])] = patient
            self.save_data()
            return True
        return False
    
    def clear_patient_symptoms(self, patient_email: str) -> bool:
        """Clear current symptoms for a patient (after session ends)."""
        patient = self.get_patient_by_email(patient_email)
        if patient:
            patient["current_symptoms"] = None
            self.data["patients"][str(patient["id"])] = patient
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
    
    def get_doctors_by_specialization(self, specialization: str) -> List[Dict]:
        """Get doctors by specialization."""
        doctors = []
        for doctor_id, doctor in self.data["doctors"].items():
            if (specialization.lower() in doctor["specialization"].lower() or 
                doctor["specialization"] == "General Medicine"):
                doctors.append({
                    'id': doctor_id,
                    'email': doctor["email"],
                    'name': doctor["name"], 
                    'specialization': doctor["specialization"],
                    'days_available': doctor["days_available"],
                    'created_at': doctor["created_at"]
                })
        
        doctors.sort(key=lambda x: (x["specialization"] != specialization, x["name"]))
        return doctors
    
    def create_appointment(self, patient_email: str, doctor_email: str, symptoms: str, 
                          appointment_date: datetime, google_event_id: str = None) -> int:
        """Create a new appointment."""
        # Get patient and doctor by email to get their IDs
        patient = self.get_patient_by_email(patient_email)
        doctor = self.get_doctor_by_email(doctor_email)
        
        if not patient:
            raise ValueError(f"Patient not found: {patient_email}")
        if not doctor:
            raise ValueError(f"Doctor not found: {doctor_email}")
        
        appointment_id = self.data["next_appointment_id"]
        appointment_data = {
            "id": appointment_id,
            "patient_id": patient["id"],
            "patient_email": patient_email,
            "doctor_id": doctor["id"],
            "doctor_email": doctor_email,
            "symptoms": symptoms,
            "appointment_date": appointment_date.isoformat(),
            "status": "scheduled",
            "appointment_completed": False,  # New flag to track appointment completion
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

    def user_exists(self, email: str) -> bool:
        """Check if a user exists in the database."""
        return self.get_patient_by_email(email) is not None or self.get_doctor_by_email(email) is not None

    def register_user(self, email: str, name: str, role: str, specialization: Optional[str] = None) -> int:
        """Register a new user (doctor or patient). Returns user ID."""
        if role.lower() == "doctor":
            return self.add_doctor(email, name, specialization or "General Medicine")
        elif role.lower() == "patient":
            return self.add_patient(email, name)
        else:
            raise ValueError("Invalid role. Must be 'doctor' or 'patient'.")

    def login_user(self, email: str) -> Optional[Dict]:
        """Log in a user by retrieving their data."""
        user = self.get_patient_by_email(email) or self.get_doctor_by_email(email)
        return user
    
    def get_appointment_by_id(self, appointment_id: int) -> Optional[Dict]:
        """Get appointment by ID."""
        for appointment in self.data["appointments"]:
            if appointment["id"] == appointment_id:
                return appointment
        return None
    
    def delete_appointment(self, appointment_id: int) -> bool:
        """Delete an appointment by ID."""
        for i, appointment in enumerate(self.data["appointments"]):
            if appointment["id"] == appointment_id:
                del self.data["appointments"][i]
                self.save_data()
                return True
        return False
    
    def update_appointment_completion_status(self, appointment_id: int, completed: bool = True) -> bool:
        """Update appointment completion status."""
        for appointment in self.data["appointments"]:
            if appointment["id"] == appointment_id:
                appointment["appointment_completed"] = completed
                if completed:
                    appointment["status"] = "completed"
                    appointment["completed_at"] = datetime.now().isoformat()
                self.save_data()
                return True
        return False
    
    def get_doctor_active_appointment(self, doctor_email: str) -> Optional[Dict]:
        """Get the doctor's current active (scheduled but not completed) appointment."""
        doctor_appointments = [
            apt for apt in self.data["appointments"] 
            if apt["doctor_email"] == doctor_email 
            and apt["status"] == "scheduled" 
            and not apt.get("appointment_completed", False)
        ]
        
        if doctor_appointments:
            # Sort by appointment date and return the earliest non-completed one
            doctor_appointments.sort(key=lambda x: x["appointment_date"])
            return doctor_appointments[0]
        return None
    
    def update_patient(self, patient_email: str, updated_data: Dict) -> bool:
        """Update patient data."""
        patient = self.get_patient_by_email(patient_email)
        if patient:
            patient_id = patient["id"]
            # Merge updated data with existing data
            for key, value in updated_data.items():
                if key != "id":  # Don't allow ID changes
                    patient[key] = value
            self.data["patients"][str(patient_id)] = patient
            self.save_data()

    def add_post_visit_record(self, patient_email: str, doctor_email: str, 
                             visit_summary: str, medications: List[Dict] = None, 
                             instructions: str = None, next_appointment: str = None,
                             appointment_id: int = None) -> int:
        """Add a comprehensive post-visit record to the database."""
        try:
            # Get patient and doctor info
            patient = self.get_patient_by_email(patient_email)
            doctor = self.get_doctor_by_email(doctor_email)
            
            if not patient or not doctor:
                raise ValueError(f"Patient or doctor not found")
            
            # Create visit record
            visit_record_id = self.data["next_visit_record_id"]
            visit_record = {
                "id": visit_record_id,
                "patient_id": patient["id"],
                "patient_email": patient_email,
                "patient_name": patient["name"],
                "doctor_id": doctor["id"],
                "doctor_email": doctor_email,
                "doctor_name": doctor["name"],
                "appointment_id": appointment_id,
                "visit_date": datetime.now().isoformat(),
                "visit_summary": visit_summary,
                "medications": medications or [],
                "instructions": instructions,
                "next_appointment": next_appointment,
                "created_at": datetime.now().isoformat()
            }
            
            # Add to database
            self.data["post_visit_records"].append(visit_record)
            self.data["next_visit_record_id"] += 1
            
            # Update patient's current medication if provided
            if medications:
                current_meds = []
                for med in medications:
                    if isinstance(med, dict):
                        current_meds.append(f"{med.get('name', '')} - {med.get('dosage', '')} - {med.get('frequency', '')}")
                    else:
                        current_meds.append(str(med))
                
                patient["current_medication"] = "; ".join(current_meds)
                self.data["patients"][str(patient["id"])] = patient
            
            self.save_data()
            return visit_record_id
            
        except Exception as e:
            print(f"Error adding post-visit record: {e}")
            return None

    def get_patient_visit_history(self, patient_email: str) -> List[Dict]:
        """Get all visit records for a patient."""
        visit_history = []
        for record in self.data["post_visit_records"]:
            if record["patient_email"] == patient_email:
                visit_history.append(record)
        
        # Sort by visit date (most recent first)
        visit_history.sort(key=lambda x: x["visit_date"], reverse=True)
        return visit_history

    def get_doctor_patient_visits(self, doctor_email: str, patient_email: str = None) -> List[Dict]:
        """Get visit records for a doctor, optionally filtered by patient."""
        visits = []
        for record in self.data["post_visit_records"]:
            if record["doctor_email"] == doctor_email:
                if patient_email is None or record["patient_email"] == patient_email:
                    visits.append(record)
        
        # Sort by visit date (most recent first)
        visits.sort(key=lambda x: x["visit_date"], reverse=True)
        return visits

    def get_patient_current_medications(self, patient_email: str) -> List[Dict]:
        """Get current medications from the most recent visit."""
        visit_history = self.get_patient_visit_history(patient_email)
        
        if not visit_history:
            return []
        
        # Get medications from most recent visit
        most_recent_visit = visit_history[0]
        return most_recent_visit.get("medications", [])

    def get_visit_record_by_id(self, visit_record_id: int) -> Optional[Dict]:
        """Get a specific visit record by ID."""
        for record in self.data["post_visit_records"]:
            if record["id"] == visit_record_id:
                return record
        return None

# Create database instance
db = JSONDatabase()
