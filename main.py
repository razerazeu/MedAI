import streamlit as st
import os
import requests
from typing import TypedDict, Literal, Annotated, List, Dict
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.messages import AnyMessage
from dotenv import load_dotenv
from IPython.display import Image, display
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from datetime import datetime, timedelta
import re
from database import db
import pickle
from google_auth_oauthlib.flow import InstalledAppFlow
import base64
import pytz

from google_calendar_auth import GoogleCalendarManager, authenticate_google_calendar, is_calendar_authenticated
from gmail_integration import GmailManager, authenticate_gmail, is_gmail_authenticated
from drug_safety import check_medication_safety, get_drug_interaction_history, get_patient_medications_with_safety_check

def parse_natural_language_datetime(date_input: str, time_input: str, current_date: datetime = None) -> tuple:
    """
    Parse natural language date and time inputs into a datetime object.
    Returns (datetime_object, success_boolean, error_message)
    """
    if current_date is None:
        current_date = datetime.now()
    
    try:
        # Handle date parsing
        date_input = date_input.lower().strip()
        time_input = time_input.lower().strip()
        
        # Date parsing
        target_date = None
        
        if date_input in ['today']:
            target_date = current_date.date()
        elif date_input in ['tomorrow']:
            target_date = (current_date + timedelta(days=1)).date()
        elif date_input in ['day after tomorrow']:
            target_date = (current_date + timedelta(days=2)).date()
        elif date_input.startswith('next '):
            # Handle "next monday", "next tuesday", etc.
            day_name = date_input.replace('next ', '').strip()
            days_of_week = {
                'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3,
                'friday': 4, 'saturday': 5, 'sunday': 6
            }
            if day_name in days_of_week:
                current_weekday = current_date.weekday()
                target_weekday = days_of_week[day_name]
                days_ahead = target_weekday - current_weekday
                if days_ahead <= 0:  # Target day already happened this week
                    days_ahead += 7
                target_date = (current_date + timedelta(days=days_ahead)).date()
        elif date_input.startswith('in '):
            # Handle "in 2 days", "in 1 week", etc.
            match = re.search(r'in (\d+) (day|days|week|weeks)', date_input)
            if match:
                number = int(match.group(1))
                unit = match.group(2)
                if unit.startswith('day'):
                    target_date = (current_date + timedelta(days=number)).date()
                elif unit.startswith('week'):
                    target_date = (current_date + timedelta(weeks=number)).date()
        else:
            # Try to parse as standard date format (YYYY-MM-DD, MM/DD/YYYY, etc.)
            date_formats = ['%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%m-%d-%Y']
            for fmt in date_formats:
                try:
                    target_date = datetime.strptime(date_input, fmt).date()
                    break
                except ValueError:
                    continue
        
        if target_date is None:
            return None, False, f"Could not understand date: '{date_input}'. Try: 'today', 'tomorrow', 'next Monday', 'in 3 days', or YYYY-MM-DD format."
        
        # Time parsing
        target_time = None
        
        # Handle common time formats
        time_patterns = [
            (r'(\d{1,2}):(\d{2})\s*(am|pm)', lambda m: parse_12_hour_time(int(m.group(1)), int(m.group(2)), m.group(3))),
            (r'(\d{1,2})\s*(am|pm)', lambda m: parse_12_hour_time(int(m.group(1)), 0, m.group(2))),
            (r'(\d{1,2}):(\d{2})', lambda m: (int(m.group(1)), int(m.group(2)))),  # 24-hour format
            (r'noon', lambda m: (12, 0)),
            (r'midnight', lambda m: (0, 0)),
        ]
        
        for pattern, parser in time_patterns:
            match = re.search(pattern, time_input)
            if match:
                try:
                    hour, minute = parser(match)
                    if 0 <= hour <= 23 and 0 <= minute <= 59:
                        target_time = (hour, minute)
                        break
                except:
                    continue
        
        if target_time is None:
            return None, False, f"Could not understand time: '{time_input}'. Try: '2:30 PM', '14:30', '2 PM', 'noon', 'midnight'."
        
        # Combine date and time
        hour, minute = target_time
        appointment_datetime = datetime.combine(target_date, datetime.min.time().replace(hour=hour, minute=minute))
        
        return appointment_datetime, True, ""
    
    except Exception as e:
        return None, False, f"Error parsing date/time: {str(e)}"

def parse_12_hour_time(hour: int, minute: int, period: str) -> tuple:
    """Convert 12-hour time to 24-hour format."""
    if period.lower() == 'pm' and hour != 12:
        hour += 12
    elif period.lower() == 'am' and hour == 12:
        hour = 0
    return hour, minute

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini",temperature=0,api_key=os.getenv("OPENAI_API_KEY"))

@tool
def MedlinePlus(query: str) -> str:
    """Searches MedlinePlus for health information."""
    try:
        url = "https://wsearch.nlm.nih.gov/ws/query"
        params = {'db': 'healthTopics', 'term': query, 'retmax': 3}
        
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        
        if 'list' in data and 'document' in data['list']:
            documents = data['list']['document']
            if not isinstance(documents, list):
                documents = [documents]
            
            results = []
            for doc in documents[:3]:
                title = doc.get('content', {}).get('title', 'No title')
                summary = doc.get('content', {}).get('FullSummary', 'No summary')[:200] + "..."
                results.append(f"**{title}**\n{summary}")
            
            return f"MedlinePlus results for '{query}':\n\n" + "\n\n".join(results)
        
        return f"No results found for '{query}' in MedlinePlus."
    
    except Exception as e:
        return f"Error searching MedlinePlus: {str(e)}"

@tool
def get_patient_medical_history(patient_email: str) -> str:
    """Get the complete medical history for a patient by their email. Use this when doctors need to review past records."""
    try:
        patient = db.get_patient_by_email(patient_email)
        if not patient:
            return f"No patient found with email: {patient_email}"
        
        records = db.get_patient_medical_history(patient_email)
        if not records:
            return f"No medical records found for {patient['name']} ({patient_email})"
        
        # Format records by type
        allergies = [r for r in records if r['record_type'] == 'allergy']
        medications = [r for r in records if r['record_type'] == 'medication']
        conditions = [r for r in records if r['record_type'] == 'condition']
        visits = [r for r in records if r['record_type'] == 'visit']
        medical_history = [r for r in records if r['record_type'] == 'medical_history']
        
        result = f"Medical History for {patient['name']} ({patient_email}):\n\n"
        
        # Show basic medical history first
        if medical_history:
            result += "📋 MEDICAL HISTORY:\n"
            for history in medical_history:
                result += f"- {history['description']}\n"
                # Show current medication if available
                details = history.get('details', {})
                if details and details.get('current_medication'):
                    result += f"- Current Medication: {details['current_medication']}\n"
            result += "\n"
        
        # Show current symptoms from patient data
        # Note: Symptoms are now tied to specific appointments, not stored as patient data
        # To see symptoms, check the appointment details for this patient
        
        if allergies:
            result += "🚨 ALLERGIES:\n"
            for allergy in allergies:
                result += f"- {allergy['description']} (Recorded: {allergy['date_recorded']})\n"
            result += "\n"
        
        if medications:
            result += "💊 CURRENT MEDICATIONS:\n"
            for med in medications:
                details = med['details']
                dosage = details.get('dosage', 'Not specified') if details else 'Not specified'
                result += f"- {med['description']} - {dosage} (Prescribed: {med['date_recorded']})\n"
            result += "\n"
        
        if conditions:
            result += "🏥 MEDICAL CONDITIONS:\n"
            for condition in conditions:
                result += f"- {condition['description']} (Diagnosed: {condition['date_recorded']})\n"
            result += "\n"
        
        if visits:
            result += "📋 RECENT VISITS:\n"
            for visit in visits[:3]:  # Show last 3 visits
                result += f"- {visit['date_recorded']}: {visit['description']}\n"
        
        return result
        
    except Exception as e:
        return f"Error retrieving medical history: {str(e)}"

@tool
def find_doctor_by_name(doctor_name: str) -> str:
    """Find a doctor's details by their name. Use this when a doctor introduces themselves and you need their email/info."""
    try:
        all_doctors = db.get_all_doctors()
        
        # Clean the input name (remove "Dr.", "Doctor", etc.)
        clean_name = doctor_name.lower().replace("dr.", "").replace("doctor", "").strip()
        
        found_doctors = []
        for doctor in all_doctors:
            # Check if the clean name matches (partial match)
            if clean_name in doctor['name'].lower():
                found_doctors.append(doctor)
        
        if not found_doctors:
            return f"No doctor found with name matching '{doctor_name}'. Available doctors: {', '.join([d['name'] for d in all_doctors])}"
        
        if len(found_doctors) == 1:
            doctor = found_doctors[0]
            return f"Found Dr. {doctor['name']}!\nEmail: {doctor['email']}\nSpecialization: {doctor['specialization']}\nDays Available: {doctor.get('days_available', 'Not specified')}"
        
        # Multiple matches
        result = f"Found {len(found_doctors)} doctors matching '{doctor_name}':\n"
        for i, doctor in enumerate(found_doctors, 1):
            result += f"{i}. Dr. {doctor['name']} ({doctor['email']}) - {doctor['specialization']}\n"
        return result
        
    except Exception as e:
        return f"Error finding doctor: {str(e)}"
    
@tool 
def find_patient_by_name(patient_name: str) -> str:
    """Find a patient's details by their name. Use this when a patient revisits."""
    try:
        all_patients = db.get_all_patients()
        for patient in all_patients:
            if patient['name'].lower() == patient_name.lower():
                return f"Found patient {patient['name']}!\nEmail: {patient['email']}\nMedical History: {patient['medical_history']}"
        return f"No patient found with name: {patient_name}"
    except Exception as e:
        return f"Error finding patient: {str(e)}"

@tool
def find_patient_by_name_or_email(identifier: str) -> str:
    """Find a patient's details by their name or email. Use this to identify patients in the database."""
    try:
        all_patients = db.get_all_patients()

        # Search by email or name
        matches = [
            patient for patient in all_patients
            if identifier.lower() in patient['email'].lower() or identifier.lower() in patient['name'].lower()
        ]

        if not matches:
            return f"No patient found matching '{identifier}'."

        if len(matches) == 1:
            patient = matches[0]
            return f"Found patient: {patient['name']}\nEmail: {patient['email']}\nMedical History: {patient.get('medical_history', 'None')}\n(Note: Symptoms are stored with specific appointments)"

        # Multiple matches
        result = f"Found {len(matches)} patients matching '{identifier}':\n"
        for i, patient in enumerate(matches, 1):
            result += f"{i}. {patient['name']} ({patient['email']})\n"
        return result

    except Exception as e:
        return f"Error finding patient: {str(e)}"

@tool
def get_doctor_appointments(doctor_email: str) -> str:
    """Get all appointments for a specific doctor. Use this when doctors want to see their schedule. Fetch the email from the database"""
    try:
        # Check if doctor exists
        doctor_found = False
        all_doctors = db.get_all_doctors()
        for doctor in all_doctors:
            if doctor['email'] == doctor_email:
                doctor_found = True
                doctor_name = doctor['name']
                break
        
        if not doctor_found:
            return f"No doctor found with email: {doctor_email}"
        
        # Get all appointments for this doctor
        all_appointments = db.get_appointments()
        doctor_appointments = [apt for apt in all_appointments if apt['doctor_email'] == doctor_email]
        
        # Filter to only show SCHEDULED appointments (not completed ones)
        scheduled_appointments = [apt for apt in doctor_appointments if apt['status'] == 'scheduled']
        
        if not scheduled_appointments:
            return f"No scheduled appointments found for Dr. {doctor_name} ({doctor_email})"
        
        result = f"📅 Scheduled Appointments for Dr. {doctor_name}:\n\n"
        
        # Sort appointments by date
        scheduled_appointments.sort(key=lambda x: x['appointment_date'])
        
        for apt in scheduled_appointments:
            # Parse date for better formatting
            apt_date = datetime.fromisoformat(apt['appointment_date'].replace('Z', '+00:00'))
            formatted_date = apt_date.strftime('%Y-%m-%d at %I:%M %p')
            
            result += f"🔹 **Appointment #{apt['id']}**\n"
            result += f"   Patient: {apt['patient_email']}\n"
            result += f"   Date: {formatted_date}\n"
            result += f"   Symptoms: {apt['symptoms']}\n"
            result += f"   Status: {apt['status']}\n"
            if apt.get('google_event_id'):
                result += f"   Calendar Event: {apt['google_event_id'][:20]}...\n"
            result += "\n"
        
        return result
        
    except Exception as e:
        return f"Error retrieving appointments: {str(e)}"

@tool
def book_appointment_with_doctor(patient_email: str, patient_name: str, symptoms: str, preferred_date: str, preferred_time: str, specialization: str = "General Medicine") -> str:
    """Book an appointment for a patient with an available doctor. 
    Supports natural language date/time:
    - Dates: 'today', 'tomorrow', 'next Monday', 'in 3 days', or YYYY-MM-DD
    - Times: '2:30 PM', '14:30', '2 PM', 'noon', 'midnight'
    """
    try:
        print(f"🔍 Comprehensive duplicate check for {patient_email}...")  # Debug log
        
        # Parse and validate patient's preferred datetime using natural language
        appointment_datetime, parse_success, parse_error = parse_natural_language_datetime(
            preferred_date, preferred_time, datetime.now()
        )
        
        if not parse_success:
            return f"❌ **Invalid Date/Time Format**\n\n{parse_error}\n\n**Examples of valid inputs:**\n- Date: 'today', 'tomorrow', 'next Monday', 'in 3 days', '2025-07-31'\n- Time: '2:30 PM', '14:30', '2 PM', 'noon'"
        
        # Validate that the appointment is in the future
        if appointment_datetime <= datetime.now():
            return f"❌ **Invalid Date/Time**\n\nThe selected appointment time ({appointment_datetime.strftime('%Y-%m-%d at %I:%M %p')}) is in the past.\n\nPlease choose a future date and time."
        
        # Validate that the appointment is within reasonable range (not more than 60 days ahead)
        if appointment_datetime > datetime.now() + timedelta(days=60):
            return f"❌ **Date Too Far**\n\nThe selected appointment time is more than 60 days in the future.\n\nPlease choose a date within the next 2 months."
        
        # COMPREHENSIVE DUPLICATE PREVENTION
        all_appointments = db.get_appointments()
        
        # Check 1: Exact same patient with any scheduled appointment in the next 7 days
        patient_scheduled_appointments = [
            apt for apt in all_appointments 
            if apt['patient_email'] == patient_email 
            and apt['status'] == 'scheduled'
            and (datetime.fromisoformat(apt['appointment_date'].replace('Z', '+00:00').replace('+00:00', '')) - datetime.now()).total_seconds() > 0  # Future appointments
            and (datetime.fromisoformat(apt['appointment_date'].replace('Z', '+00:00').replace('+00:00', '')) - datetime.now()).total_seconds() < 604800  # Within 7 days
        ]
        
        if patient_scheduled_appointments:
            print(f"❌ Patient already has scheduled appointments in the next 7 days")  # Debug log
            existing_apt = patient_scheduled_appointments[0]
            apt_date = datetime.fromisoformat(existing_apt['appointment_date'].replace('Z', '+00:00').replace('+00:00', ''))
            return f"❌ You already have a scheduled appointment:\n\n📅 **Existing Appointment**\nAppointment ID: {existing_apt['id']}\nDate: {apt_date.strftime('%Y-%m-%d at %I:%M %p')}\nDoctor: {existing_apt['doctor_email']}\nSymptoms: {existing_apt['symptoms']}\n\n⚠️ **Policy**: Only one appointment per patient per week is allowed. Please wait for your current appointment or cancel it first if you need to reschedule."
        
        # Check 2: Similar symptoms in the last 24 hours (prevent rapid re-booking)
        recent_similar_appointments = [
            apt for apt in all_appointments 
            if apt['patient_email'] == patient_email 
            and apt['symptoms'].strip().lower() == symptoms.strip().lower()
            and apt['status'] in ['scheduled', 'completed']  # Include completed to prevent immediate re-booking
            and (datetime.now() - datetime.fromisoformat(apt['created_at'].replace('Z', '+00:00').replace('+00:00', ''))).total_seconds() < 86400  # 24 hours
        ]
        
        if recent_similar_appointments:
            print(f"❌ Similar symptoms appointment found within 24 hours")  # Debug log
            recent_apt = recent_similar_appointments[0]
            return f"❌ Similar appointment detected within 24 hours:\n\n📋 **Recent Appointment**\nAppointment ID: {recent_apt['id']}\nSymptoms: {recent_apt['symptoms']}\nStatus: {recent_apt['status']}\n\n⚠️ **Policy**: Cannot book appointments with similar symptoms within 24 hours. Please wait or contact support if this is urgent."
        
        # Check 3: Multiple appointments being created rapidly (within 10 minutes)
        very_recent_appointments = [
            apt for apt in all_appointments 
            if apt['patient_email'] == patient_email 
            and (datetime.now() - datetime.fromisoformat(apt['created_at'].replace('Z', '+00:00').replace('+00:00', ''))).total_seconds() < 600  # 10 minutes
        ]
        
        if very_recent_appointments:
            print(f"❌ Patient trying to create multiple appointments rapidly")  # Debug log
            return f"❌ Multiple booking attempt detected!\n\n⚠️ **Policy**: Please wait at least 10 minutes between appointment booking attempts. This prevents accidental duplicate bookings.\n\nYour recent appointment: {very_recent_appointments[0]['id']}"
        
        print(f"✅ All duplicate checks passed. Proceeding with booking...")  # Debug log
        
        # Add patient to database if not exists
        patient_id = db.add_patient(patient_email, patient_name)
        
        # Find doctors by specialization
        doctors = db.get_doctors_by_specialization(specialization)
        print(f"Found {len(doctors)} doctors for specialization '{specialization}': {[d['email'] for d in doctors]}")  # Debug log
        if not doctors:
            return f"No doctors found for specialization: {specialization}"
        
        selected_doctor = doctors[0]
        print(f"Selected doctor: {selected_doctor['name']} ({selected_doctor['email']})")  # Debug log
        
        # Check if the doctor is available at the patient's preferred time
        appointment_time = appointment_datetime
        
        # Check for doctor conflicts at the requested time (1-hour appointment window)
        doctor_appointments_at_time = [
            apt for apt in all_appointments
            if apt['doctor_email'] == selected_doctor['email']
            and apt['status'] == 'scheduled'
            and abs((datetime.fromisoformat(apt['appointment_date'].replace('Z', '+00:00').replace('+00:00', '')) - appointment_time).total_seconds()) < 3600  # Within 1 hour
        ]
        
        if doctor_appointments_at_time:
            conflicting_apt = doctor_appointments_at_time[0]
            conflict_time = datetime.fromisoformat(conflicting_apt['appointment_date'].replace('Z', '+00:00').replace('+00:00', ''))
            
            return f"""❌ **Doctor Not Available**

Dr. {selected_doctor['name']} already has an appointment at your requested time:

🚫 **Your Request**: {appointment_time.strftime('%Y-%m-%d at %I:%M %p')}
📅 **Conflicting Appointment**: {conflict_time.strftime('%Y-%m-%d at %I:%M %p')}

**Please choose a different time slot:**
- Try booking 2+ hours before or after the conflicting appointment
**Available Specializations**: {specialization}
"""
        
        # Create Google Calendar event
        try:
            calendar_service = get_google_calendar_service()
            event = create_calendar_event(
                calendar_service,
                f"Medical Appointment - {patient_name}",
                f"Patient: {patient_name} ({patient_email})\nSymptoms: {symptoms}",
                appointment_time,
                appointment_time + timedelta(hours=1),
                selected_doctor['email']
            )
            google_event_id = event.get('id')
        except Exception as e:
            google_event_id = None
            print(f"Calendar error: {e}")
        
        # Save appointment to database with additional duplicate prevention
        print(f"Creating appointment for patient {patient_email} with doctor {selected_doctor['email']}...")  # Debug log
        
        # Final check: Ensure no appointment was created while we were processing
        final_check_appointments = db.get_appointments()
        final_duplicate_check = [
            apt for apt in final_check_appointments 
            if apt['patient_email'] == patient_email 
            and apt['status'] == 'scheduled'
            and (datetime.now() - datetime.fromisoformat(apt['created_at'].replace('Z', '+00:00').replace('+00:00', ''))).total_seconds() < 60  # Created in last minute
        ]
        
        if final_duplicate_check:
            print(f"❌ Race condition detected - appointment created during processing")
            return f"❌ An appointment was just created for you. Please refresh and check your appointments."
        
        appointment_id = db.create_appointment(
            patient_email,  # Use email directly instead of patient_id
            selected_doctor['email'], 
            symptoms, 
            appointment_time,
            google_event_id
        )
        print(f"Appointment created with ID: {appointment_id}")  # Debug log
        
        # Send email to doctor
        email_subject = f"New Patient Appointment - {patient_name}"
        email_body = f"""
Dear Dr. {selected_doctor['name']},

You have a new patient appointment scheduled:

Patient: {patient_name}
Email: {patient_email}
Date & Time: {appointment_time.strftime('%Y-%m-%d at %I:%M %p')}

SYMPTOMS REPORTED:
{symptoms}

Please review the patient's medical history if needed using the AI assistant.

Best regards,
MedAI
        """   
        send_email_via_google(selected_doctor['email'], email_subject, email_body)
        
        return f"""
✅ Appointment booked successfully!

Patient: {patient_name}
Doctor: Dr. {selected_doctor['name']} ({selected_doctor['specialization']})
Date & Time: {appointment_time.strftime('%Y-%m-%d at %I:%M %p')}
Appointment ID: {appointment_id}

The doctor has been notified via email with your symptoms.
You should receive a calendar invitation shortly.
"""     
    except Exception as e:
        print(f"❌ Error booking appointment: {str(e)}")  # Debug log
        return f"Error booking appointment: {str(e)}"

@tool
def add_medical_record_after_visit(patient_email: str, record_type: str, description: str, doctor_email: str, additional_details: str = None) -> str:
    """Add medical records after a doctor visit. record_type can be: 'medication', 'condition', 'allergy', 'visit'"""
    try:
        patient = db.get_patient_by_email(patient_email)
        if not patient:
            return f"Patient not found: {patient_email}"
        
        # For now, update the patient's medical history field
        current_history = patient.get('medical_history', '') or ''
        
        new_record = f"{record_type.upper()}: {description}"
        if additional_details:
            new_record += f" ({additional_details})"
        
        new_record += f" - Added by {doctor_email} on {datetime.now().strftime('%Y-%m-%d')}"
        
        # Append to existing medical history
        if current_history:
            updated_history = current_history + "\n" + new_record
        else:
            updated_history = new_record
        
        # Update patient in database
        db.add_patient(
            email=patient_email,
            name=patient['name'],
            medical_history=updated_history,
            current_medication=patient.get('current_medication')
        )
        
        return f"✅ Medical record added successfully for {patient['name']} ({patient_email})"
        
    except Exception as e:
        return f"Error adding medical record: {str(e)}"

@tool
def update_patient_information(patient_email: str, medical_history: str = None) -> str:
    """Update patient's medical history when they share this information during conversation. Note: Symptoms are tied to specific appointments. For medications, use report_current_medications for complete list or add_medication_to_patient for doctor prescriptions."""
    try:
        patient = db.get_patient_by_email(patient_email)
        if not patient:
            return f"Patient not found: {patient_email}"
        
        # Update medical history only
        if medical_history:
            current_history = patient.get('medical_history', '') or ''
            timestamp = datetime.now().strftime('%Y-%m-%d')
            new_entry = f"[{timestamp}] {medical_history}"
            
            if current_history:
                updated_history = current_history + "\n" + new_entry
            else:
                updated_history = new_entry
            
            # Save to database
            db.add_patient(
                email=patient_email,
                name=patient['name'],
                medical_history=updated_history,
                current_medication=patient.get('current_medication')
            )
            
            return f"✅ Updated medical history for {patient['name']} ({patient_email})"
        else:
            return "No medical history provided to update."
        
    except Exception as e:
        return f"Error updating patient information: {str(e)}"

@tool
def report_current_medications(patient_email: str, complete_medication_list: str) -> str:
    """Patient reports their complete current medication list when they first register or have no medication data yet. Use when new patients tell you all their medications."""
    try:
        patient = db.get_patient_by_email(patient_email)
        if not patient:
            return f"Patient not found: {patient_email}"
        
        # Set the medication list for new patients or patients without existing medication data
        db.add_patient(
            email=patient_email,
            name=patient['name'],
            medical_history=patient.get('medical_history'),
            current_medication=complete_medication_list
        )
        
        return f"✅ Recorded medication list for {patient['name']} ({patient_email}).\n\nCurrent medications: {complete_medication_list}"
        
    except Exception as e:
        return f"Error reporting current medications: {str(e)}"

@tool
def add_medication_to_patient(patient_email: str, medication_name: str, dosage: str = None, frequency: str = None, doctor_email: str = None) -> str:
    """Add a specific medication to patient's current medication list. This appends to existing medications rather than overwriting."""
    try:
        patient = db.get_patient_by_email(patient_email)
        if not patient:
            return f"Patient not found: {patient_email}"
        
        # Format the new medication
        med_string = medication_name
        if dosage:
            med_string += f" {dosage}"
        if frequency:
            med_string += f" ({frequency})"
        
        # Add timestamp and doctor info if provided
        timestamp = datetime.now().strftime('%Y-%m-%d')
        if doctor_email:
            med_string += f" - Prescribed by {doctor_email} on {timestamp}"
        
        # Get existing medications
        existing_medication = patient.get('current_medication', '') or ''
        
        if existing_medication:
            # Append to existing medications
            updated_medication = existing_medication + ", " + med_string
        else:
            # First medication for this patient
            updated_medication = med_string
        
        # Update patient in database
        db.add_patient(
            email=patient_email,
            name=patient['name'],
            medical_history=patient.get('medical_history'),
            current_medication=updated_medication
        )
        
        return f"✅ Added medication '{medication_name}' to {patient['name']}'s current medication list.\n\nUpdated medications: {updated_medication}"
        
    except Exception as e:
        return f"Error adding medication: {str(e)}"

@tool
def cancel_appointment(appointment_id: int) -> str:
    """Cancel an appointment, remove it from the database and Google Calendar, and notify patient and doctor."""
    try:
        # Fetch the appointment details
        appointment = db.get_appointment_by_id(appointment_id)
        if not appointment:
            return f"No appointment found with ID: {appointment_id}"

        patient_email = appointment['patient_email']
        doctor_email = appointment['doctor_email']
        google_event_id = appointment.get('google_event_id')

        # Remove the appointment from the database
        print(f"Deleting appointment with ID: {appointment_id}...")  # Debug log
        db.delete_appointment(appointment_id)
        print(f"Appointment {appointment_id} deleted successfully.")  # Debug log

        # Remove the event from Google Calendar if it exists
        if google_event_id:
            try:
                calendar_service = get_google_calendar_service()
                calendar_service.events().delete(calendarId='primary', eventId=google_event_id).execute()
            except Exception as e:
                print(f"Error removing event from Google Calendar: {e}")

        # Send cancellation emails
        email_subject = "Appointment Cancellation Notice"
        email_body_patient = f"""
Dear Patient,

Your appointment scheduled with Dr. {doctor_email} has been cancelled.

If you have any questions, please contact the clinic.

Best regards,
MedAI
        """
        email_body_doctor = f"""
Dear Doctor,

The appointment with patient {patient_email} has been cancelled.

If you have any questions, please contact the clinic.

Best regards,
MedAI
        """
        send_email_via_google(patient_email, email_subject, email_body_patient)
        send_email_via_google(doctor_email, email_subject, email_body_doctor)

        return f"Appointment {appointment_id} cancelled successfully. Notifications sent to patient and doctor."

    except Exception as e:
        return f"Error cancelling appointment: {str(e)}"

@tool
def complete_appointment_and_collect_visit_data(appointment_id: int, doctor_email: str) -> str:
    """Mark an appointment as completed and prompt doctor to provide post-visit data (medications, instructions, summary). Use this when doctor indicates the appointment/consultation is finished."""
    try:
        # Get the appointment details
        appointment = db.get_appointment_by_id(appointment_id)
        if not appointment:
            return f"❌ No appointment found with ID: {appointment_id}"
        
        # Verify the doctor owns this appointment
        if appointment['doctor_email'] != doctor_email:
            return f"❌ Access denied. This appointment belongs to a different doctor."
        
        # Check if already completed
        if appointment.get('appointment_completed', False):
            return f"⚠️ Appointment {appointment_id} is already marked as completed."
        
        # Mark appointment as completed
        success = db.update_appointment_completion_status(appointment_id, completed=True)
        if not success:
            return f"❌ Failed to update appointment status."
        
        # Get patient information
        patient = db.get_patient_by_email(appointment['patient_email'])
        patient_name = patient['name'] if patient else appointment['patient_email']
        
        # Prompt doctor for post-visit information
        return f"""✅ **Appointment #{appointment_id} marked as COMPLETED**

**Patient:** {patient_name} ({appointment['patient_email']})
**Original Symptoms:** {appointment['symptoms']}
**Completed At:** {datetime.now().strftime('%Y-%m-%d at %I:%M %p')}

═══════════════════════════════════════
📋 **POST-VISIT DATA COLLECTION**
═══════════════════════════════════════

Please provide the following information to send a comprehensive summary to the patient:

**1. VISIT SUMMARY** (Required)
- What was discussed during the visit?
- Diagnosis or findings?
- Key points the patient should remember?

**2. MEDICATIONS** (If prescribed) - **IMPORTANT: Include Duration**
- List any prescribed medications with dosages
- **MUST include treatment duration**: "for 7 days", "for 2 weeks", "for 1 month"
- Include instructions for taking them
- Mention any important warnings or side effects
- **Examples:**
  * "Amoxicillin 500mg twice daily for 10 days"
  * "Ibuprofen 400mg as needed for pain for 1 week"
  * "Metformin 500mg daily long-term"

**3. POST-VISIT INSTRUCTIONS** (If applicable)
- Recovery instructions
- Lifestyle recommendations
- What to watch for (warning signs)
- Activity restrictions

**4. FOLLOW-UP** (If needed)
- When should patient return?
- What type of follow-up appointment?
- Any tests or procedures needed?

💡 **Next Step:** Once you provide this information, I'll automatically send a professional email summary to the patient with all the details."""
    
    except Exception as e:
        return f"Error completing appointment: {str(e)}"

@tool
def get_doctor_current_patient(doctor_email: str) -> str:
    """Get the doctor's current active patient (scheduled appointment that hasn't been completed yet). Use this when doctor asks about 'the patient' or 'my patient'."""
    try:
        # Get the doctor's active appointment (not completed)
        active_appointment = db.get_doctor_active_appointment(doctor_email)
        
        if not active_appointment:
            # Check if doctor has any appointments at all
            all_appointments = db.get_appointments()
            doctor_appointments = [apt for apt in all_appointments if apt['doctor_email'] == doctor_email]
            
            if not doctor_appointments:
                return f"❌ No appointments found for Dr. {doctor_email}."
            else:
                return f"⚠️ No active appointments found. All your appointments have been completed.\n\n💡 Use 'get_doctor_appointments' to see your full appointment history."
        
        # Get patient details
        patient = db.get_patient_by_email(active_appointment['patient_email'])
        if not patient:
            return f"❌ Patient record not found for {active_appointment['patient_email']}"
        
        # Format appointment date
        apt_date = datetime.fromisoformat(active_appointment['appointment_date'].replace('Z', '+00:00').replace('+00:00', ''))
        formatted_date = apt_date.strftime('%Y-%m-%d at %I:%M %p')
        
        return f"""🏥 **CURRENT ACTIVE PATIENT**

**Patient:** {patient['name']}
**Email:** {patient['email']}
**Appointment ID:** #{active_appointment['id']}
**Scheduled:** {formatted_date}
**Original Symptoms:** {active_appointment['symptoms']}
**Status:** {active_appointment['status']} (Not completed)

═══════════════════════════════════════
📋 **PATIENT MEDICAL PROFILE**
═══════════════════════════════════════
**Medical History:** {patient.get('medical_history') or 'None recorded'}
**Current Medications:** {patient.get('current_medication') or 'None recorded'}

💡 **Note:** Symptoms for this visit: {active_appointment['symptoms']}
💡 **Next Step:** When finished, use 'complete_appointment_and_collect_visit_data' to mark it complete and send patient summary."""
    
    except Exception as e:
        return f"Error getting current patient: {str(e)}"

@tool
def send_post_visit_summary(patient_email: str, doctor_email: str, visit_summary: str, medications: str = None, instructions: str = None, next_appointment: str = None) -> str:
    """Send a comprehensive post-visit summary email to patient with medical information, medications, and follow-up instructions."""
    try:
        # Add debug logging
        print(f"DEBUG: Attempting to send post-visit summary to {patient_email} from {doctor_email}")
        
        patient = db.get_patient_by_email(patient_email)
        doctor = db.get_doctor_by_email(doctor_email)
        
        if not patient:
            return f"❌ Patient not found with email: {patient_email}"
        if not doctor:
            return f"❌ Doctor not found with email: {doctor_email}"
        
        # Create comprehensive email content
        email_subject = f"Visit Summary - {datetime.now().strftime('%Y-%m-%d')}"
        email_body = f"""
Dear {patient['name']},

Thank you for visiting Dr. {doctor['name']} today. Here is your visit summary:

═══════════════════════════════════════
📋 VISIT SUMMARY
═══════════════════════════════════════
{visit_summary}

"""
        
        if medications:
            email_body += f"""
═══════════════════════════════════════
💊 MEDICATIONS PRESCRIBED
═══════════════════════════════════════
{medications}

IMPORTANT: Please follow the medication schedule exactly as prescribed. If you experience any side effects, contact us immediately.

"""
        
        if instructions:
            email_body += f"""
═══════════════════════════════════════
📝 POST-VISIT INSTRUCTIONS
═══════════════════════════════════════
{instructions}

"""
        
        if next_appointment:
            email_body += f"""
═══════════════════════════════════════
📅 NEXT APPOINTMENT
═══════════════════════════════════════
{next_appointment}

"""
        
        email_body += f"""
═══════════════════════════════════════
📞 CONTACT INFORMATION
═══════════════════════════════════════
If you have any questions or concerns, please contact:
- Dr. {doctor['name']}: {doctor['email']}
- Clinic Phone: [Contact clinic for phone number]

Best regards,
Dr. {doctor['name']}
{doctor['specialization']}
MedAI Healthcare System
        """
        
        # Parse medications into structured format for database storage
        parsed_medications = []
        if medications:
            # Use intelligent medication parsing
            parsed_medications = parse_medications_intelligently(medications)
            
            # Auto-schedule medication reminders if medications were prescribed
            if parsed_medications and len(parsed_medications) > 0:
                # Check if any medication has duration specified
                has_duration = any(med.get('duration') for med in parsed_medications)
                
                if has_duration:
                    print(f"DEBUG: Scheduling medication reminders with doctor-specified durations...")
                    try:
                        schedule_result = schedule_medication_reminders_with_duration(
                            patient_email, parsed_medications, doctor_email
                        )
                        print(f"DEBUG: Medication scheduling result: {schedule_result}")
                    except Exception as sched_error:
                        print(f"DEBUG: Error scheduling medication reminders: {sched_error}")
                else:
                    print(f"DEBUG: No duration specified for medications, using default scheduling")
                    # Fall back to simple medication update without automatic scheduling
                    pass

        # Update patient's current medication list with new prescriptions
        if parsed_medications:
            update_patient_current_medications(patient_email, parsed_medications)

        # Save post-visit data to database FIRST (before sending email)
        print(f"DEBUG: Saving post-visit data to database...")
        print(f"DEBUG: Parsed {len(parsed_medications)} medications: {[med['name'] for med in parsed_medications]}")
        visit_record_id = db.add_post_visit_record(
            patient_email=patient_email,
            doctor_email=doctor_email,
            visit_summary=visit_summary,
            medications=parsed_medications,
            instructions=instructions,
            next_appointment=next_appointment,
            appointment_id=None  # TODO: Link with actual appointment ID
        )
        
        if visit_record_id:
            print(f"DEBUG: Post-visit record saved with ID: {visit_record_id}")
        else:
            print(f"DEBUG: Failed to save post-visit record")

        # Send email
        print(f"DEBUG: Attempting to send email via Gmail API...")
        email_result = send_email_via_google(patient_email, email_subject, email_body)
        
        if email_result:
            print(f"DEBUG: Email sent successfully to {patient_email}")
            success_msg = f"✅ Post-visit summary sent successfully to {patient['name']} ({patient_email})"
            if visit_record_id:
                success_msg += f"\n📁 Visit record saved to database (ID: {visit_record_id})"
            return success_msg
        else:
            print(f"DEBUG: Email sending failed to {patient_email}")
            error_msg = f"❌ Failed to send post-visit summary to {patient_email}. Please check email authentication."
            if visit_record_id:
                error_msg += f"\n📁 However, visit record was saved to database (ID: {visit_record_id})"
            return error_msg
            
    except Exception as e:
        error_msg = f"Error sending post-visit summary: {str(e)}"
        print(f"DEBUG: Exception occurred - {error_msg}")
        return error_msg

def parse_medications_intelligently(medications_text: str) -> List[Dict]:
    """
    Intelligently parse medication text into structured format.
    Handles complex medication descriptions with multiple medications in one text block.
    """
    import re
    
    parsed_medications = []
    
    if not medications_text or not medications_text.strip():
        return parsed_medications
    
    text = medications_text.strip()
    
    # First, try to identify clear medication boundaries
    # Look for patterns like "MedicationName dosage - instructions. AnotherMedication dosage - instructions"
    
    # Split on periods followed by capital letters (likely new medications)
    potential_segments = re.split(r'(?<=\.)\s+(?=[A-Z][a-z]+(?:\s+\d+|\s+[A-Z]))', text)
    
    # If we don't get multiple segments, try alternative splitting
    if len(potential_segments) == 1:
        # Try splitting on medication name patterns at the beginning of sentences
        potential_segments = re.split(r'(?<=[.!])\s+(?=[A-Z][a-z]+\s+\d+)', text)
    
    # If still just one segment but text contains multiple obvious medication names, 
    # try splitting based on medication name patterns
    if len(potential_segments) == 1 and len(re.findall(r'\b[A-Z][a-z]+\s+\d+(?:\.\d+)?\s*(?:mg|%)', text)) >= 2:
        # Find all medication name positions
        med_positions = []
        for match in re.finditer(r'\b([A-Z][a-z]+)\s+(\d+(?:\.\d+)?\s*(?:mg|%|mcg))', text):
            med_positions.append((match.start(), match.group(1), match.group(2)))
        
        if len(med_positions) >= 2:
            segments = []
            for i, (pos, name, dose) in enumerate(med_positions):
                if i < len(med_positions) - 1:
                    end_pos = med_positions[i + 1][0]
                    segment = text[pos:end_pos].strip()
                else:
                    segment = text[pos:].strip()
                segments.append(segment)
            potential_segments = segments
    
    # Process each segment
    for segment in potential_segments:
        segment = segment.strip()
        if len(segment) < 5:  # Skip very short segments
            continue
        
        parsed_med = parse_single_medication_segment(segment)
        if parsed_med and parsed_med['name'] and parsed_med['name'] != "Prescribed medication":
            # Avoid duplicates
            if not any(existing['name'].lower().strip() == parsed_med['name'].lower().strip() 
                      for existing in parsed_medications):
                parsed_medications.append(parsed_med)
    
    # Fallback: if no medications were parsed, treat entire text as one medication
    if not parsed_medications:
        parsed_medications.append({
            "name": "Prescribed medication",
            "dosage": "",
            "frequency": "",
            "duration": "",
            "instructions": text
        })
    
    print(f"DEBUG: Parsed medications: {parsed_medications}")
    return parsed_medications


def parse_single_medication_segment(segment: str) -> Dict:
    """Parse a single medication segment to extract structured information including duration."""
    import re
    
    # Extract medication name and dosage
    med_match = re.search(r'\b([A-Z][a-z]+(?:[a-z]*)?(?:\s+\d+(?:\.\d+)?\s*%)?(?:\s+(?:shampoo|cream|tablet|capsule))?)\s*(\d+(?:\.\d+)?\s*(?:mg|%|mcg|g|ml))?', segment)
    
    if not med_match:
        # Fallback: try to find any capitalized word that looks like a medication
        fallback_match = re.search(r'\b([A-Z][a-z]{3,})', segment)
        if fallback_match:
            med_name = fallback_match.group(1)
            dosage = ""
        else:
            return {"name": "Prescribed medication", "dosage": "", "frequency": "", "duration": "", "instructions": segment}
    else:
        med_name = med_match.group(1).strip()
        dosage = med_match.group(2).strip() if med_match.group(2) else ""
    
    # If dosage is already in the name, don't duplicate
    if dosage and dosage in med_name:
        full_name = med_name
        dosage_only = dosage
    else:
        full_name = med_name + (' ' + dosage if dosage else '')
        dosage_only = dosage
    
    # Extract frequency
    frequency = ""
    frequency_patterns = [
        r'(?:take|use|apply)?\s*(?:once|twice|three times?|3x|2x|1x)?\s*(?:daily|per day|a day|weekly|per week)',
        r'(?:once|twice|three times?|3x|2x|1x)\s*(?:/week|per week|weekly)',
        r'every\s+\d+\s*(?:hours?|days?|weeks?)',
        r'at\s+(?:night|bedtime|morning)',
        r'with\s+(?:meals|food)',
        r'\d+x/week'
    ]
    
    for pattern in frequency_patterns:
        freq_match = re.search(pattern, segment, re.IGNORECASE)
        if freq_match:
            frequency = freq_match.group().strip()
            break
    
    # Extract duration - NEW FEATURE
    duration = ""
    duration_patterns = [
        r'for\s+(\d+)\s*(?:day|days)',  # "for 7 days"
        r'for\s+(\d+)\s*(?:week|weeks)',  # "for 2 weeks"
        r'for\s+(\d+)\s*(?:month|months)',  # "for 1 month"
        r'(?:duration|take for|continue for)[\s:]+(\d+)\s*(?:day|days)',
        r'(?:duration|take for|continue for)[\s:]+(\d+)\s*(?:week|weeks)',
        r'(?:duration|take for|continue for)[\s:]+(\d+)\s*(?:month|months)',
        r'(\d+)[-\s](?:day|days)\s*(?:course|treatment|supply)',
        r'(\d+)[-\s](?:week|weeks)\s*(?:course|treatment|supply)',
        r'until\s+(?:symptoms\s+)?(?:resolve|improve|gone)',  # "until symptoms resolve"
        r'as\s+needed',  # "as needed" (PRN)
        r'long[-\s]?term',  # "long-term"
        r'indefinitely',  # "indefinitely"
    ]
    
    for pattern in duration_patterns:
        duration_match = re.search(pattern, segment, re.IGNORECASE)
        if duration_match:
            if pattern.endswith('resolve|improve|gone)'):
                duration = "until symptoms resolve"
            elif r'as\s+needed' in pattern:
                duration = "as needed (PRN)"
            elif r'long[-\s]?term' in pattern:
                duration = "long-term"
            elif 'indefinitely' in pattern:
                duration = "indefinitely"
            else:
                # Extract number and determine unit
                number = duration_match.group(1)
                if 'day' in pattern:
                    duration = f"{number} days"
                elif 'week' in pattern:
                    duration = f"{number} weeks"
                elif 'month' in pattern:
                    duration = f"{number} months"
            break
    
    # Extract instructions - clean up the segment by removing medication name, dosage, and duration
    instructions = segment
    instructions = re.sub(rf'\b{re.escape(med_name)}\b', '', instructions, flags=re.IGNORECASE)
    if dosage:
        instructions = re.sub(rf'\b{re.escape(dosage)}\b', '', instructions, flags=re.IGNORECASE)
    if duration:
        # Remove the duration text from instructions
        instructions = re.sub(rf'for\s+{re.escape(duration)}', '', instructions, flags=re.IGNORECASE)
    
    # Clean up leading dashes, spaces, and punctuation
    instructions = re.sub(r'^[\s\-–—]+', '', instructions).strip()
    
    return {
        "name": full_name,
        "dosage": dosage_only, 
        "frequency": frequency,
        "duration": duration,
        "instructions": instructions
    }

def parse_single_medication(med_name: str, med_text: str) -> Dict:
    """Legacy function - parse a single medication from its text description."""
    # This is kept for backward compatibility but redirects to the new function
    return parse_single_medication_segment(f"{med_name} - {med_text}")

def update_patient_current_medications(patient_email: str, new_medications: List[Dict]) -> bool:
    """Update patient's current medication list by appending new medications from visit."""
    try:
        patient = db.get_patient_by_email(patient_email)
        if not patient:
            return False
        
        existing_medication = patient.get('current_medication', '') or ''
        
        # Convert new medications to string format
        new_med_strings = []
        for med in new_medications:
            if med.get('name') and med['name'] != "Prescribed medication":
                med_string = med['name']
                if med.get('dosage'):
                    med_string += f" {med['dosage']}"
                if med.get('frequency'):
                    med_string += f" ({med['frequency']})"
                new_med_strings.append(med_string)
        
        if new_med_strings:
            new_med_text = ", ".join(new_med_strings)
            
            if existing_medication:
                # Append to existing medications
                updated_medication = existing_medication + ", " + new_med_text
            else:
                # First medications for this patient
                updated_medication = new_med_text
            
            # Update patient in database
            db.add_patient(
                email=patient_email,
                name=patient['name'],
                medical_history=patient.get('medical_history'),
                current_medication=updated_medication
            )
            return True
        
        return False
    except Exception as e:
        print(f"Error updating patient medications: {e}")
        return False

@tool
def schedule_medication_reminders_with_duration(patient_email: str, medication_list: List[Dict], doctor_email: str) -> str:
    """Schedule medication reminders using doctor-specified durations for each medication. 
    medication_list should contain parsed medications with duration field."""
    try:
        patient = db.get_patient_by_email(patient_email)
        if not patient:
            return f"Patient not found: {patient_email}"
        
        if not medication_list:
            return "No medications provided to schedule"
        
        calendar_events = []
        current_date = datetime.now().replace(hour=8, minute=0, second=0, microsecond=0)  # Start at 8 AM
        
        successful_schedules = []
        failed_schedules = []
        
        for medication in medication_list:
            med_name = medication.get('name', 'Unknown medication')
            med_duration = medication.get('duration', '')
            med_frequency = medication.get('frequency', 'daily')
            med_dosage = medication.get('dosage', '')
            med_instructions = medication.get('instructions', '')
            
            # Calculate duration in days based on doctor's specification
            duration_days = calculate_duration_in_days(med_duration)
            
            if duration_days <= 0:
                failed_schedules.append(f"{med_name} - Invalid or missing duration: '{med_duration}'")
                continue
            
            # Create medication reminders for the calculated duration
            med_events = []
            for day in range(duration_days):
                event_date = current_date + timedelta(days=day)
                
                # Create daily medication reminder
                try:
                    calendar_service = get_google_calendar_service()
                    event_title = f"Medication: {med_name}"
                    event_description = f"""Patient: {patient['name']}
Medication: {med_name}
Dosage: {med_dosage}
Frequency: {med_frequency}
Duration: {med_duration}
Instructions: {med_instructions}

Prescribed by: Dr. {doctor_email}
Day {day + 1} of {duration_days}"""
                    
                    event = create_calendar_event(
                        calendar_service,
                        event_title,
                        event_description,
                        event_date,
                        event_date + timedelta(minutes=15),
                        patient_email
                    )
                    med_events.append(event.get('id'))
                except Exception as e:
                    print(f"Error creating calendar event for {med_name} day {day}: {e}")
            
            if med_events:
                calendar_events.extend(med_events)
                successful_schedules.append(f"{med_name} - {len(med_events)} reminders over {duration_days} days")
            else:
                failed_schedules.append(f"{med_name} - Failed to create calendar events")
        
        # Update patient's current medication in database
        medication_summary = "\n".join([
            f"{med['name']} {med.get('dosage', '')} - {med.get('frequency', '')} for {med.get('duration', '')}"
            for med in medication_list
        ])
        
        db.add_patient(
            email=patient_email,
            name=patient['name'],
            medical_history=patient.get('medical_history'),
            current_medication=medication_summary,
            current_symptoms=patient.get('current_symptoms'),
            role=patient.get('role', 'Patient')
        )
        
        # Generate response
        result = f"✅ **MEDICATION REMINDERS SCHEDULED**\n\n"
        result += f"Patient: {patient['name']} ({patient_email})\n"
        result += f"Total calendar events created: {len(calendar_events)}\n\n"
        
        if successful_schedules:
            result += "📊 **SUCCESSFUL SCHEDULES:**\n"
            for schedule in successful_schedules:
                result += f"   ✅ {schedule}\n"
            result += "\n"
        
        if failed_schedules:
            result += "⚠️ **FAILED SCHEDULES:**\n"
            for failure in failed_schedules:
                result += f"   ❌ {failure}\n"
            result += "\n"
        
        result += "📱 **REMINDER DETAILS:**\n"
        result += "   • Reminders set for 8:00 AM daily\n"
        result += "   • Duration based on doctor's prescription\n"
        result += "   • Patient will receive calendar notifications\n"
        result += f"   • Prescribed by: Dr. {doctor_email}\n"
        
        return result
        
    except Exception as e:
        return f"Error scheduling medication reminders: {str(e)}"

def calculate_duration_in_days(duration_text: str) -> int:
    """Convert duration text to number of days for scheduling."""
    if not duration_text:
        return 7  # Default to 1 week if no duration specified
    
    duration_lower = duration_text.lower().strip()
    
    # Handle special cases
    if duration_lower in ["as needed", "as needed (prn)", "prn"]:
        return 30  # 30 days for PRN medications
    elif duration_lower in ["long-term", "indefinitely"]:
        return 90  # 90 days for long-term medications
    elif "until symptoms resolve" in duration_lower:
        return 14  # 2 weeks for symptom-based duration
    
    # Extract numeric durations
    import re
    
    # Match patterns like "7 days", "2 weeks", "1 month"
    day_match = re.search(r'(\d+)\s*days?', duration_lower)
    week_match = re.search(r'(\d+)\s*weeks?', duration_lower)
    month_match = re.search(r'(\d+)\s*months?', duration_lower)
    
    if day_match:
        return int(day_match.group(1))
    elif week_match:
        return int(week_match.group(1)) * 7
    elif month_match:
        return int(month_match.group(1)) * 30
    
    # If no pattern matches, default to 4 days
    return 4

@tool
def send_appointment_reminder(patient_email: str, appointment_id: int, reminder_type: str = "24hour") -> str:
    """Send appointment reminder emails to patients. reminder_type can be: '24hour', '2hour', or 'custom'"""
    try:
        appointment = db.get_appointment_by_id(appointment_id)
        if not appointment:
            return f"No appointment found with ID: {appointment_id}"
        
        patient = db.get_patient_by_email(patient_email)
        if not patient:
            return f"Patient not found: {patient_email}"
        
        # Parse appointment date
        apt_date = datetime.fromisoformat(appointment['appointment_date'].replace('Z', '+00:00'))
        formatted_date = apt_date.strftime('%Y-%m-%d at %I:%M %p')
        
        # Create reminder email
        if reminder_type == "24hour":
            subject = "Appointment Reminder - Tomorrow"
            timing = "tomorrow"
        elif reminder_type == "2hour":
            subject = "Appointment Reminder - In 2 Hours"
            timing = "in 2 hours"
        else:
            subject = "Appointment Reminder"
            timing = "soon"
        
        email_body = f"""
Dear {patient['name']},

This is a friendly reminder about your upcoming appointment {timing}.

═══════════════════════════════════════
📅 APPOINTMENT DETAILS
═══════════════════════════════════════
Date & Time: {formatted_date}
Doctor: {appointment['doctor_email']}
Appointment ID: {appointment_id}

Original Symptoms: {appointment['symptoms']}

═══════════════════════════════════════
📝 PREPARATION NOTES
═══════════════════════════════════════
• Please arrive 15 minutes early
• Bring your ID and insurance card
• Bring a list of current medications
• Prepare any questions you want to ask

If you need to reschedule or cancel, please contact us as soon as possible.

Best regards,
MedAI Healthcare System
        """
        
        if send_email_via_google(patient_email, subject, email_body):
            return f"✅ {reminder_type} reminder sent to {patient['name']} for appointment #{appointment_id}"
        else:
            return f"❌ Failed to send reminder to {patient_email}"
            
    except Exception as e:
        return f"Error sending appointment reminder: {str(e)}"

@tool
def request_patient_consent(patient_email: str, consent_type: str, details: str) -> str:
    """Request patient consent for specific actions like medication scheduling, data sharing, or treatment plans."""
    try:
        patient = db.get_patient_by_email(patient_email)
        if not patient:
            return f"Patient not found: {patient_email}"
        
        consent_subject = f"Consent Request - {consent_type}"
        consent_body = f"""
Dear {patient['name']},

We are requesting your consent for the following:

═══════════════════════════════════════
🔒 CONSENT REQUEST: {consent_type.upper()}
═══════════════════════════════════════

{details}

═══════════════════════════════════════
📋 YOUR OPTIONS
═══════════════════════════════════════
To provide your consent, please reply to this email with:
• "YES" - I consent to the above
• "NO" - I do not consent
• "QUESTIONS" - I have questions about this request

Your privacy and autonomy are important to us. You can withdraw consent at any time.

═══════════════════════════════════════
📞 QUESTIONS?
═══════════════════════════════════════
If you have any questions about this consent request, please contact your healthcare provider or reply to this email.

Best regards,
MedAI Healthcare System
        """
        
        if send_email_via_google(patient_email, consent_subject, consent_body):
            return f"✅ Consent request sent to {patient['name']} for: {consent_type}\n\nDetails: {details}\n\nThe patient will receive an email asking for their consent and can reply with YES, NO, or QUESTIONS."
        else:
            return f"❌ Failed to send consent request to {patient_email}"
            
    except Exception as e:
        return f"Error sending consent request: {str(e)}"

@tool
def check_patient_existing_appointments(patient_email: str) -> str:
    """Check if a patient has any existing scheduled appointments before booking new ones. Use this BEFORE booking appointments."""
    try:
        all_appointments = db.get_appointments()
        patient_appointments = [
            apt for apt in all_appointments 
            if apt['patient_email'] == patient_email 
            and apt['status'] == 'scheduled'
            and datetime.fromisoformat(apt['appointment_date'].replace('Z', '+00:00').replace('+00:00', '')) > datetime.now()  # Future appointments only
        ]
        
        if not patient_appointments:
            return f"✅ No existing scheduled appointments found for {patient_email}. Safe to book new appointment."
        
        # Sort by appointment date
        patient_appointments.sort(key=lambda x: x['appointment_date'])
        
        result = f"⚠️ EXISTING APPOINTMENTS FOUND for {patient_email}:\n\n"
        for apt in patient_appointments:
            apt_date = datetime.fromisoformat(apt['appointment_date'].replace('Z', '+00:00').replace('+00:00', ''))
            result += f"📅 **Appointment #{apt['id']}**\n"
            result += f"   Date: {apt_date.strftime('%Y-%m-%d at %I:%M %p')}\n"
            result += f"   Doctor: {apt['doctor_email']}\n"
            result += f"   Symptoms: {apt['symptoms']}\n"
            result += f"   Status: {apt['status']}\n\n"
        
        result += "❌ **POLICY VIOLATION**: Patient already has scheduled appointment(s). New booking should be REJECTED.\n"
        result += "💡 **Suggest**: Ask patient to wait for existing appointment or cancel it first if rescheduling is needed."
        
        return result
        
    except Exception as e:
        return f"Error checking existing appointments: {str(e)}"

@tool
def clean_duplicate_appointments(patient_email: str) -> str:
    """Remove duplicate appointments for a patient based on same symptoms and date. Also performs comprehensive duplicate cleanup."""
    try:
        appointments = db.get_appointments()
        all_patient_appointments = [apt for apt in appointments if apt['patient_email'] == patient_email]
        
        if not all_patient_appointments:
            return f"No appointments found for patient: {patient_email}"
        
        duplicates_removed = 0
        cleanup_report = []
        
        # Clean up by multiple criteria
        
        # 1. Group by exact same symptoms and same day
        symptoms_groups = {}
        for apt in all_patient_appointments:
            if apt['status'] == 'scheduled':  # Only clean scheduled appointments
                key = f"{apt['symptoms'].strip().lower()}_{apt['appointment_date'][:10]}"
                if key not in symptoms_groups:
                    symptoms_groups[key] = []
                symptoms_groups[key].append(apt)
        
        for group in symptoms_groups.values():
            if len(group) > 1:
                # Keep the earliest created appointment, remove the rest
                group.sort(key=lambda x: x['created_at'])
                for duplicate in group[1:]:
                    db.delete_appointment(duplicate['id'])
                    duplicates_removed += 1
                    cleanup_report.append(f"Removed duplicate appointment {duplicate['id']} (same symptoms, same day)")
        
        # 2. Check for multiple scheduled appointments for same patient
        scheduled_appointments = [apt for apt in all_patient_appointments if apt['status'] == 'scheduled']
        if len(scheduled_appointments) > 1:
            # Keep the earliest scheduled appointment, remove others
            scheduled_appointments.sort(key=lambda x: x['appointment_date'])
            for extra_apt in scheduled_appointments[1:]:
                db.delete_appointment(extra_apt['id'])
                duplicates_removed += 1
                cleanup_report.append(f"Removed extra appointment {extra_apt['id']} (multiple scheduled appointments policy)")
        
        # 3. Clean up appointments created within minutes of each other
        all_patient_appointments.sort(key=lambda x: x['created_at'])
        for i in range(len(all_patient_appointments) - 1):
            current_apt = all_patient_appointments[i]
            next_apt = all_patient_appointments[i + 1]
            
            if current_apt['status'] == 'scheduled' and next_apt['status'] == 'scheduled':
                current_time = datetime.fromisoformat(current_apt['created_at'].replace('Z', '+00:00').replace('+00:00', ''))
                next_time = datetime.fromisoformat(next_apt['created_at'].replace('Z', '+00:00').replace('+00:00', ''))
                
                if (next_time - current_time).total_seconds() < 300:  # Created within 5 minutes
                    db.delete_appointment(next_apt['id'])
                    duplicates_removed += 1
                    cleanup_report.append(f"Removed rapid duplicate {next_apt['id']} (created within 5 minutes)")
        
        result = f"✅ Duplicate cleanup completed for {patient_email}:\n"
        result += f"• Total duplicates removed: {duplicates_removed}\n"
        
        if cleanup_report:
            result += "\n📋 Cleanup Details:\n"
            for report in cleanup_report:
                result += f"• {report}\n"
        
        if duplicates_removed == 0:
            result += "• No duplicates found - appointments are properly organized"
        
        return result
        
    except Exception as e:
        return f"Error cleaning duplicate appointments: {str(e)}"

@tool
def get_patient_visit_history(patient_email: str) -> str:
    """Get complete visit history for a patient including medications, instructions, and visit summaries."""
    try:
        visit_history = db.get_patient_visit_history(patient_email)
        patient = db.get_patient_by_email(patient_email)
        
        if not patient:
            return f"❌ Patient not found with email: {patient_email}"
        
        if not visit_history:
            return f"📋 No visit history found for {patient['name']} ({patient_email})"
        
        result = f"📋 **VISIT HISTORY FOR {patient['name'].upper()}** ({patient_email})\n\n"
        
        for i, visit in enumerate(visit_history, 1):
            visit_date = datetime.fromisoformat(visit['visit_date'].replace('Z', '+00:00')).strftime('%Y-%m-%d %H:%M')
            result += f"**VISIT #{i} - {visit_date}**\n"
            result += f"👨‍⚕️ Doctor: Dr. {visit['doctor_name']} ({visit['doctor_email']})\n"
            result += f"📝 Summary: {visit['visit_summary']}\n"
            
            if visit.get('medications'):
                result += f"💊 **Medications Prescribed:**\n"
                for med in visit['medications']:
                    if isinstance(med, dict):
                        result += f"   • {med.get('name', 'Unknown')} - {med.get('dosage', '')} - {med.get('frequency', '')}\n"
                        if med.get('instructions'):
                            result += f"     Instructions: {med.get('instructions')}\n"
                    else:
                        result += f"   • {med}\n"
            
            if visit.get('instructions'):
                result += f"📋 **Instructions:** {visit['instructions']}\n"
            
            if visit.get('next_appointment'):
                result += f"📅 **Next Appointment:** {visit['next_appointment']}\n"
            
            result += "\n" + "="*50 + "\n\n"
        
        return result
        
    except Exception as e:
        return f"❌ Error retrieving visit history: {str(e)}"

@tool
def get_patient_current_medications_detailed(patient_email: str) -> str:
    """Get current medications for a patient from their most recent visit for scheduling reminders."""
    try:
        medications = db.get_patient_current_medications(patient_email)
        patient = db.get_patient_by_email(patient_email)
        
        if not patient:
            return f"❌ Patient not found with email: {patient_email}"
        
        if not medications:
            return f"💊 No current medications found for {patient['name']} ({patient_email})"
        
        result = f"💊 **CURRENT MEDICATIONS FOR {patient['name'].upper()}**\n\n"
        
        for i, med in enumerate(medications, 1):
            if isinstance(med, dict):
                result += f"**{i}. {med.get('name', 'Unknown Medication')}**\n"
                if med.get('dosage'):
                    result += f"   📏 Dosage: {med.get('dosage')}\n"
                if med.get('frequency'):
                    result += f"   ⏰ Frequency: {med.get('frequency')}\n"
                if med.get('instructions'):
                    result += f"   📝 Instructions: {med.get('instructions')}\n"
            else:
                result += f"**{i}.** {med}\n"
            result += "\n"
        
        result += "\n🔔 **Ready for scheduling medication reminders!**\n"
        result += "Use the schedule_medication_reminders tool to set up automated reminders for these medications."
        
        return result
        
    except Exception as e:
        return f"❌ Error retrieving current medications: {str(e)}"

@tool
def search_visit_records_by_condition(patient_email: str, condition_keyword: str) -> str:
    """Search through a patient's visit history for specific conditions, symptoms, or medications."""
    try:
        visit_history = db.get_patient_visit_history(patient_email)
        patient = db.get_patient_by_email(patient_email)
        
        if not patient:
            return f"❌ Patient not found with email: {patient_email}"
        
        if not visit_history:
            return f"📋 No visit history found for {patient['name']} ({patient_email})"
        
        matching_visits = []
        keyword_lower = condition_keyword.lower()
        
        for visit in visit_history:
            # Search in visit summary
            if keyword_lower in visit.get('visit_summary', '').lower():
                matching_visits.append(visit)
                continue
            
            # Search in medications
            if visit.get('medications'):
                for med in visit['medications']:
                    if isinstance(med, dict):
                        if (keyword_lower in med.get('name', '').lower() or 
                            keyword_lower in med.get('instructions', '').lower()):
                            matching_visits.append(visit)
                            break
                    elif keyword_lower in str(med).lower():
                        matching_visits.append(visit)
                        break
            
            # Search in instructions
            if keyword_lower in visit.get('instructions', '').lower():
                matching_visits.append(visit)
        
        if not matching_visits:
            return f"🔍 No visits found containing '{condition_keyword}' for {patient['name']}"
        
        result = f"🔍 **SEARCH RESULTS FOR '{condition_keyword.upper()}'**\n"
        result += f"Patient: {patient['name']} ({patient_email})\n"
        result += f"Found {len(matching_visits)} matching visit(s)\n\n"
        
        for i, visit in enumerate(matching_visits, 1):
            visit_date = datetime.fromisoformat(visit['visit_date'].replace('Z', '+00:00')).strftime('%Y-%m-%d')
            result += f"**MATCH #{i} - {visit_date}**\n"
            result += f"👨‍⚕️ Doctor: Dr. {visit['doctor_name']}\n"
            result += f"📝 Summary: {visit['visit_summary']}\n"
            
            if visit.get('medications'):
                result += f"💊 Medications: "
                med_names = []
                for med in visit['medications']:
                    if isinstance(med, dict):
                        med_names.append(med.get('name', 'Unknown'))
                    else:
                        med_names.append(str(med))
                result += ", ".join(med_names) + "\n"
            
            result += "\n"
        
        return result
        
    except Exception as e:
        return f"❌ Error searching visit records: {str(e)}"

@tool
def schedule_medication_reminders_from_visit_data(patient_email: str, duration_days: int = 30) -> str:
    """Schedule medication reminders using the patient's current medications from their most recent visit."""
    try:
        patient = db.get_patient_by_email(patient_email)
        if not patient:
            return f"❌ Patient not found: {patient_email}"
        
        # Get current medications from most recent visit
        medications = db.get_patient_current_medications(patient_email)
        
        if not medications:
            return f"💊 No current medications found for {patient['name']}. Complete a post-visit summary first to save medication data."
        
        # Create medication schedule string from structured data
        medication_schedule = []
        for med in medications:
            if isinstance(med, dict):
                med_line = f"{med.get('name', 'Unknown')}"
                if med.get('dosage'):
                    med_line += f" - {med.get('dosage')}"
                if med.get('frequency'):
                    med_line += f" - {med.get('frequency')}"
                if med.get('instructions'):
                    med_line += f" - {med.get('instructions')}"
                medication_schedule.append(med_line)
            else:
                medication_schedule.append(str(med))
        
        medication_schedule_str = '\n'.join(medication_schedule)
        
        # Create medication reminders for the specified duration
        calendar_events = []
        current_date = datetime.now().replace(hour=8, minute=0, second=0, microsecond=0)  # Start at 8 AM
        
        scheduled_days = 0
        for day in range(duration_days):
            event_date = current_date + timedelta(days=day)
            
            # Create daily medication reminder
            try:
                calendar_service = get_google_calendar_service()
                event = create_calendar_event(
                    calendar_service,
                    f"Medication Reminder - {patient['name']}",
                    f"Daily Medications:\n{medication_schedule_str}\n\nFrom recent visit records",
                    event_date,
                    event_date + timedelta(minutes=15),
                    patient_email
                )
                calendar_events.append(event.get('id'))
                scheduled_days += 1
            except Exception as e:
                print(f"Error creating calendar event for day {day}: {e}")
        
        if scheduled_days > 0:
            result = f"✅ **MEDICATION REMINDERS SCHEDULED**\n\n"
            result += f"👤 Patient: {patient['name']} ({patient_email})\n"
            result += f"📅 Duration: {scheduled_days} days starting tomorrow\n"
            result += f"⏰ Daily reminder time: 8:00 AM\n\n"
            result += f"💊 **Medications included:**\n"
            for i, med_line in enumerate(medication_schedule, 1):
                result += f"   {i}. {med_line}\n"
            result += f"\n📧 Calendar invitations sent to {patient_email}"
            result += f"\n🔔 Patient will receive daily reminders for their medication schedule"
            return result
        else:
            return f"❌ Failed to schedule medication reminders. Check calendar authentication."
        
    except Exception as e:
        return f"❌ Error scheduling medication reminders: {str(e)}"

def update_patient_medical_record(patient_email: str, new_medication: str, doctor_email: str):
    """Update the patient's medical history and current medication."""
    try:
        # Fetch patient data
        patient = db.get_patient_by_email(patient_email)
        if not patient:
            return f"No patient found with email: {patient_email}"

        # Update current medication
        patient['current_medication'] = new_medication

        # Append to medical history
        if not patient['medical_history']:
            patient['medical_history'] = []
        patient['medical_history'].append({
            'date': datetime.now().isoformat(),
            'medication': new_medication,
            'updated_by': doctor_email
        })

        # Save changes to the database
        db.update_patient(patient_email, patient)
        return f"Patient {patient['name']}'s medical record updated successfully."

    except Exception as e:
        return f"Error updating medical record: {str(e)}"

# Google Calendar Setup
SCOPES = ['https://www.googleapis.com/auth/calendar', 'https://www.googleapis.com/auth/gmail.send']

def get_google_calendar_service():
    """Setup Google Calendar API service."""
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = Flow.from_client_secrets_file(
                'credentials.json', 
                SCOPES
            )
            flow.redirect_uri = 'urn:ietf:wg:oauth:2.0:oob'

            auth_url, _ = flow.authorization_url(prompt='consent')
            print(f'Please go to this URL: {auth_url}')
            code = input('Enter the authorization code: ')
            flow.fetch_token(code=code)
            creds = flow.credentials

        with open('token.json', 'w') as token:
            token.write(creds.to_json())

    return build('calendar', 'v3', credentials=creds)

def create_calendar_event(service, summary, description, start_time, end_time, attendee_email=None):
    """Create a Google Calendar event."""
    event = {
        'summary': summary,
        'description': description,
        'start': {
            'dateTime': start_time.isoformat(),
            'timeZone': 'America/New_York',
        },
        'end': {
            'dateTime': end_time.isoformat(),
            'timeZone': 'America/New_York',
        },
    }
    
    # Add attendee only if email is valid
    if attendee_email and '@' in attendee_email and '.' in attendee_email:
        print(f"Adding attendee email: {attendee_email}")  # Debug log
        event['attendees'] = [{'email': attendee_email}]
    else:
        print(f"Skipping invalid attendee email: {attendee_email}")  # Debug log
    
    event = service.events().insert(calendarId='primary', body=event).execute()
    return event

# Email Setup
def send_email_via_google(to_email, subject, body):
    """Send email using Gmail manager with proper authentication."""
    try:
        print(f"DEBUG: Preparing to send email to {to_email}...")
        
        # Check if Gmail is authenticated
        if not gmail_manager.is_authenticated:
            print("DEBUG: Gmail not authenticated. Attempting to authenticate...")
            if not gmail_manager.authenticate():
                print("DEBUG: Gmail authentication failed.")
                return False
        
        # Use the Gmail manager's service
        if not gmail_manager.service:
            print("DEBUG: Gmail service not available.")
            return False

        # Create message
        message = MIMEMultipart()
        message['to'] = to_email
        message['subject'] = subject
        message.attach(MIMEText(body, 'plain'))

        # Encode message properly for Gmail API
        raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode()

        # Send email using Gmail manager's service
        result = gmail_manager.service.users().messages().send(
            userId='me', 
            body={'raw': raw_message}
        ).execute()

        print(f"DEBUG: Email sent successfully to {to_email}, message ID: {result.get('id', 'unknown')}")
        return True

    except Exception as e:
        print(f"DEBUG: Error sending email to {to_email}: {e}")
        return False

# Define GMAIL_SCOPES for GmailManager
GMAIL_SCOPES = [
    'https://www.googleapis.com/auth/gmail.send',
    'https://www.googleapis.com/auth/gmail.compose',
    'https://www.googleapis.com/auth/gmail.modify'
]

# Initialize global instances for Gmail and Calendar managers
gmail_manager = GmailManager()
calendar_manager = GoogleCalendarManager()

# Streamlit interface for authentication
st.sidebar.header("Google API Authentication")

if st.sidebar.button("Authenticate Google Calendar"):
    if calendar_manager.authenticate():
        st.sidebar.success("Google Calendar authenticated successfully!")
    else:
        st.sidebar.error("Failed to authenticate Google Calendar. Please try again.")

if st.sidebar.button("Authenticate Gmail"):
    if gmail_manager.authenticate():
        st.sidebar.success("Gmail authenticated successfully!")
    else:
        st.sidebar.error("Failed to authenticate Gmail. Please try again.")

tools = [
    MedlinePlus,
    get_patient_medical_history,
    find_patient_by_name_or_email,
    find_doctor_by_name,
    get_doctor_appointments,
    get_doctor_current_patient,
    check_patient_existing_appointments,
    book_appointment_with_doctor,
    add_medical_record_after_visit,
    update_patient_information,
    add_medication_to_patient,
    report_current_medications,
    cancel_appointment,
    complete_appointment_and_collect_visit_data,
    send_post_visit_summary,
    schedule_medication_reminders_with_duration,
    send_appointment_reminder,
    request_patient_consent,
    clean_duplicate_appointments,
    get_patient_visit_history,
    get_patient_current_medications_detailed,
    search_visit_records_by_condition,
    schedule_medication_reminders_from_visit_data,
    check_medication_safety,
    get_drug_interaction_history,
    get_patient_medications_with_safety_check
]
llm_with_tools = llm.bind_tools(tools)

class GraphState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    user_context: dict  # Store current user information

# Node
def tool_calling_llm(state: GraphState):
    # Get user context from graph state
    user_context = ""
    if state.get("user_context") and state["user_context"]:
        user = state["user_context"]
        user_context = f"""
CURRENT USER CONTEXT:
- User Name: {user['name']}
- User Email: {user['email']}
- User Role: {user['role']}
- IMPORTANT: This user is already authenticated. Do not ask for their email or name when they want to perform actions like booking appointments, canceling appointments, or updating their information.
"""

    sys_msg = SystemMessage(content=f'''You are MedAI, an advanced AI healthcare assistant designed to facilitate seamless, secure communication between patients and doctors. You have access to comprehensive tools and maintain conversation context throughout interactions.

{user_context}

=== CORE FUNCTIONALITY BY USER ROLE ===

📋 **FOR PATIENTS:**

PATIENT SELF-IDENTIFICATION:
- When a patient first interacts, use find_patient_by_name_or_email to locate their profile
- If they're logged in (see USER CONTEXT above), automatically use their email to identify them
- Present their current medical information: medical history, current symptoms, current medications

AUTOMATIC INFORMATION CAPTURE:
- ALWAYS use update_patient_information when patients mention:
  * Medical history: "I had surgery in 2020", "I'm diabetic", "I have allergies to penicillin"
- ALWAYS use report_current_medications when NEW patients mention medications:
  * Check if patient has existing medication data first - if null/empty, use report_current_medications
- Symptoms are captured during appointment booking and stored with specific appointments
- Update their profile immediately without asking for confirmation
- Acknowledge the update: "I've updated your medical profile with this information"

🔥 **CRITICAL MEDICATION MANAGEMENT RULES:**

**PATIENT MEDICATION PERMISSIONS:**
- **INITIAL MEDICATION REPORTING**: New patients CAN provide their complete medication list using report_current_medications
  * Use when patients first register and tell you all their current medications
  * Example: "My current medications are aspirin, lisinopril, and metformin"
  * This sets their initial medication data in the system

- **INDIVIDUAL MEDICATION ADDITIONS**: Patients CANNOT add individual medications to their existing list
  * Do NOT use add_medication_to_patient for patient requests
  * If existing patient mentions a new medication, direct them to speak with their doctor
  * Example: Patient says "I started taking vitamin D" → "Please discuss this with your doctor during your appointment"

**DOCTOR MEDICATION PERMISSIONS:**
- **PRESCRIPTION ADDITIONS**: Doctors CAN add individual medications using add_medication_to_patient
  * These new prescriptions are APPENDED to the patient's existing medication list
  * Use when doctors prescribe new medications after appointments
  * Example: Doctor prescribes antibiotics → Use add_medication_to_patient to append to existing list

**MEDICATION SCENARIOS - WHAT'S ALLOWED:**
✅ **Scenario 1**: NEW patient reports initial medication list → Use report_current_medications (sets initial data)
✅ **Scenario 2**: Doctor prescribes new medication → Use add_medication_to_patient (APPENDS to existing list)  
❌ **Scenario 3**: EXISTING patient adds individual medication to his existing medication → NOT ALLOWED
✅ **Scenario 4**: Doctor prescribes additional medication → Use add_medication_to_patient (APPENDS to existing list)

COMPREHENSIVE HEALTH INFORMATION GATHERING:
- **NEVER ask for just symptoms alone** - always collect the complete picture
- When patients mention health concerns, ALWAYS ask for ALL THREE:
  1. "What symptoms are you experiencing?" 
  2. "What's your medical history - any conditions, surgeries, allergies?"
  3. "What medications are you currently taking?"
- **IMMEDIATE DATA CAPTURE**: When patients provide this information, AUTOMATICALLY save it:
  * Medical history → Use update_patient_information 
  * Current medications → Use report_current_medications (for new patients with no existing medication data)
- This ensures complete patient safety and proper medical assessment

🚨 **DRUG SAFETY & INTERACTION CHECKING:**
- **IMMEDIATE SAFETY CHECK**: When patients mention ANY medications (new or existing):
  * Use check_medication_safety with their email and complete medication list
  * This automatically checks for dangerous drug-drug interactions using FDA data
  * System will alert doctors immediately for HIGH-RISK interactions via email
  * Provide clear safety guidance to patients about their medication combinations

- **MEDICATION REVIEW REQUESTS**: When patients ask about their current medications:
  * Use get_patient_medications_with_safety_check to show their medication list WITH safety assessment
  * Examples: "What medications am I taking?", "Can you review my medications?", "What's on my medication list?"
  * This automatically checks their existing medications for interactions and provides safety guidance
  * Alerts doctors if high-risk interactions are found during the review

- **SAFETY COMMUNICATION**:
  * If HIGH-RISK interactions found: "🚨 We've detected a potentially serious drug interaction in your medications. Your doctor has been immediately notified. Do not stop or change any medications without consulting your doctor first."
  * If moderate interactions found: "⚠️ Some of your medications may interact with each other. We'll discuss this during your appointment."
  * If no interactions: "✅ Your current medications appear to be safe when taken together."

- **MEDICATION HISTORY**: Use get_drug_interaction_history to review previous safety alerts
- **DOCTOR INTEGRATION**: High-risk interactions trigger automatic emails to the patient's assigned doctor
- **PATIENT SAFETY**: Always emphasize: "Do NOT stop or change medications without professional medical advice"

APPOINTMENT BOOKING:
- Use book_appointment_with_doctor for scheduling
- For logged-in users: automatically use their email and name from USER CONTEXT
- **ALWAYS ASK FOR ALL FIVE REQUIRED FIELDS**: When booking appointments, ALWAYS collect:
  * Current symptoms: "What symptoms are you experiencing?"
  * Medical history: "Do you have any medical history, allergies, or past conditions I should know about?"
  * Current medications: "What medications are you currently taking (including vitamins, supplements)?"
  * **REQUIRED: Preferred Date**: "What date would you like for your appointment?"
  * **REQUIRED: Preferred Time**: "What time would you prefer?"

**NATURAL LANGUAGE DATE/TIME SUPPORT**: The system now accepts user-friendly date/time formats:
- **Current Date Context**: Today is {datetime.now().strftime('%A, %B %d, %Y')} - use this for reference when patients say "tomorrow", "next week", etc.
- **Date Examples**: 'today', 'tomorrow', 'next Monday', 'in 3 days', 'July 31st', or '2025-07-31'
- **Time Examples**: '2:30 PM', '14:30', '2 PM', 'noon', 'midnight'
- **User-Friendly Approach**: Let patients express dates/times naturally - don't force strict formats!
- **Examples of natural requests**:
  * "I'd like an appointment tomorrow at 2 PM"
  * "Can I book for next Friday at noon?"
  * "How about in 3 days at 10:30 AM?"
- Use update_patient_information to save all collected information before booking
- **DOCTOR AVAILABILITY CHECKING**: The system will check if the doctor is available at the requested time
  * If available: Appointment is confirmed and doctor is notified
  * If conflict: Patient is informed and asked to choose a different time
- Ask for preferred specialization only if symptoms are unclear
- Match symptoms to appropriate medical specializations
- STRICT DUPLICATE PREVENTION: The system has comprehensive duplicate prevention
  * Only ONE appointment per patient per week is allowed
  * Cannot book similar symptoms within 24 hours
  * Cannot book multiple appointments within 10 minutes
  * The booking function will automatically reject duplicates with clear explanations
- If booking fails due to duplicates or conflicts, explain the specific issue and suggest alternatives
- Confirm appointment details and explain next steps

APPOINTMENT MANAGEMENT:
- Use cancel_appointment when patients want to cancel
- Use send_appointment_reminder for appointment reminders (24hour, 2hour, or custom)
- Use clean_duplicate_appointments if duplicate bookings are suspected

HEALTH INFORMATION:
- Use MedlinePlus for general health education (not for diagnosis)
- Provide educational content about conditions, treatments, prevention
- Always emphasize: "This is educational information only, consult your doctor for medical advice"

🩺 **FOR DOCTORS:**

DOCTOR IDENTIFICATION:
- Use find_doctor_by_name when doctors introduce themselves
- Retrieve their email, specialization, and schedule information
- 🚨 CRITICAL SECURITY: NEVER modify, update, or change a doctor's specialization field
- Doctor specializations are PERMANENT and set during registration only
- AI tools cannot and must not alter doctor specialization data

**AI ROLE IN DOCTOR WORKFLOW:**
- AI is a separate tool that doctor uses BEFORE and AFTER patient appointments
- Doctor and patient meet face-to-face in clinic (NO AI involvement during actual consultation)
- BEFORE appointment: Doctor may use AI to lookup patient medical history
- DURING appointment: Doctor examines patient physically (AI is NOT involved)
- AFTER appointment: Patient leaves, then doctor uses AI to complete appointment workflow
- AI emails comprehensive summary to patient at their home after appointment ends

CRITICAL: PATIENT CONTEXT AWARENESS FOR DOCTORS:

🔥 **NEW IMPROVED WORKFLOW - APPOINTMENT COMPLETION SYSTEM:**
- When a doctor asks about "the patient", "my patient", or "patient's medical history":
  * FIRST: Use get_doctor_current_patient to get their ACTIVE (non-completed) appointment
  * This tool automatically identifies the current patient the doctor is seeing
  * NEVER ask the doctor "which patient?" - the system finds their active appointment
  * If no active appointment exists, all appointments have been completed

ENHANCED PATIENT IDENTIFICATION LOGIC:
- **ACTIVE APPOINTMENT PRIORITY**: The system now distinguishes between:
  * 🟢 **Active Appointments**: Scheduled but not completed (appointment_completed = False)
  * 🔴 **Completed Appointments**: Finished consultations (appointment_completed = True)
- **SMART CONTEXT RESOLUTION**: When doctor says "the patient":
  * Step 1: Check for active (non-completed) appointments using get_doctor_current_patient
  * Step 2: If active appointment found, that's THE patient they're referring to
  * Step 3: If no active appointments, inform doctor all appointments are completed
- **NO MORE CONFUSION**: The "most recent" appointment is now the "most recent ACTIVE" appointment

APPOINTMENT COMPLETION WORKFLOW:

🚨 **AUTOMATIC COMPLETION DETECTION - CRITICAL:**
- **ALWAYS monitor doctor's messages** for appointment completion indicators
- **IMMEDIATELY use complete_appointment_and_collect_visit_data** when doctor says ANY of these phrases:
  * "the appointment is done" / "appointment is finished" / "we're done"
  * "consultation complete" / "consultation is over" / "finished with this patient"
  * "that's all for today" / "see you next time" / "appointment ended"
  * "we're finished here" / "consultation finished" / "visit is complete"
  * "I'm done with this patient" / "this appointment is over"
  * "ready to wrap up" / "let's finish this appointment"
  * "consultation has ended" / "we can end here"

🔥 **COMPLETION TRIGGER WORKFLOW:**
1. **FACE-TO-FACE APPOINTMENT**: Doctor and patient meet physically in clinic
2. **PATIENT LEAVES**: After consultation, patient goes home
3. **DOCTOR USES AI**: Doctor then opens AI system and says completion phrase
4. **INSTANT DETECTION**: AI detects completion signal from doctor
5. **AUTO-EXECUTE**: Use complete_appointment_and_collect_visit_data with current appointment ID
6. **FLAG UPDATE**: System automatically sets appointment_completed = True
7. **DATA COLLECTION**: Prompt doctor to provide post-visit information for patient email:
   * Visit summary (required)
   * Medications prescribed (if any)
   * Post-visit instructions (if applicable)  
   * Follow-up requirements (if needed)
8. **AUTO-EMAIL**: After doctor provides data, use send_post_visit_summary to email patient at home

⚡ **IMPORTANT CONTEXT:**
- Face-to-face appointment happens between doctor and patient (NO AI involved)
- Patient leaves clinic after appointment
- Doctor then separately uses AI system to complete appointment workflow
- When doctor tells AI "appointment is complete", they mean the face-to-face consultation is over
- AI's role is to process post-visit data and email summary to patient at home
- Patient receives professional email summary after leaving the clinic

🎯 **COMPLETION CONFIRMATION EXAMPLES:**
- Doctor (to AI after patient left): "Current appointment complete"
  → AI: "I'll mark this appointment as complete. Please provide the visit summary to email to the patient..."
- Doctor (to AI after consultation): "The appointment is finished" 
  → AI: "Completing appointment #X. What should I include in the patient's email summary?"
- Doctor (to AI): "Just finished with the patient, appointment done"
  → AI: "Appointment finished. Let me collect the post-visit information to send to the patient..."

After doctor provides post-visit data:
  * Use send_post_visit_summary to email comprehensive summary to patient
  * Include all medications, instructions, and follow-up details
  * Professional formatting with clear sections

DOCTOR WORKFLOW EXAMPLES:
1. **BEFORE Face-to-Face Consultation:**
   - Doctor (to AI): "What's the patient's medical history?" 
   - AI: Uses get_doctor_current_patient → "Your active patient is [Name]. Here's their history..."
   - [Doctor reviews information before seeing patient]

2. **DURING Face-to-Face Consultation:**
   - [Doctor and patient meet physically in clinic - NO AI involvement]
   - [Doctor examines patient, discusses symptoms, provides treatment]
   - [Patient leaves clinic and goes home]

3. **AFTER Face-to-Face Consultation:**
   - Doctor (to AI): "Current appointment complete" (patient has already left)
   - AI: Uses complete_appointment_and_collect_visit_data → Prompts for visit data to email patient
   - Doctor (to AI): Provides summary, medications, instructions for patient email
   - AI: Uses send_post_visit_summary → Sends professional email to patient at home

4. **Next Patient Preparation:**
   - Doctor (to AI): "Tell me about the patient" (for next appointment)
   - AI: Uses get_doctor_current_patient → "No active appointments. All completed."

LEGACY SUPPORT (if get_doctor_current_patient fails):
- When doctor asks about "the patient" without context and get_doctor_current_patient returns no active appointments:
  * Use get_doctor_appointments to show completed appointments
  * Ask doctor to specify which patient by name if needed
  * Prioritize appointments from today or most recent dates

PATIENT NAME RESOLUTION:
- When a doctor mentions a specific patient name:
  * Use find_patient_by_name_or_email to locate patient record
  * If found, retrieve their medical history with get_patient_medical_history
  * If not found: "No patient found with that name. Would you like to see your appointment list?"

APPOINTMENT CONTEXT INTELLIGENCE:
- ALWAYS assume doctor questions about "the patient" refer to their most recent/current appointment
- Check appointment times - if appointment is today or recent, that's likely the patient in question
- Use appointment.patient_email to get medical history, not doctor's email
- Prioritize scheduled appointments over completed ones when multiple exist

PATIENT REVIEW:
- Use find_patient_by_name_or_email to locate patient records
- Use get_patient_medical_history for comprehensive patient background
- Present organized information: allergies, medications, conditions, recent visits

SCHEDULE MANAGEMENT:
- Use get_doctor_appointments to show their patient schedule
- Display appointments with: patient details, symptoms, appointment times, status

MEDICAL RECORD MANAGEMENT:
- Use add_medical_record_after_visit for post-consultation documentation
- Record types: 'medication', 'condition', 'allergy', 'visit'
- Include detailed descriptions and additional context

POST-VISIT COMMUNICATION & RECORD KEEPING:
- After doctor indicates appointment is complete, AI automatically collects post-visit data
- Use send_post_visit_summary to email comprehensive summaries to patients at home
- Include: visit summary, prescribed medications, post-visit instructions, next appointment details
- Professional formatting with clear sections and patient-friendly language
- **AUTOMATIC DATABASE STORAGE**: All post-visit data is saved to database with patient-doctor relationships
- **MEDICATION TRACKING**: Current medications automatically updated in patient records

VISIT HISTORY & MEDICATION MANAGEMENT:
- Use get_patient_visit_history to review all previous visits, medications, and instructions
- Use get_patient_current_medications_detailed to see current medications from recent visits
- Use search_visit_records_by_condition to find specific conditions/medications in patient history
- Use schedule_medication_reminders_from_visit_data to automatically schedule reminders using stored medication data
- ALWAYS use request_patient_consent before scheduling medication reminders
- Use schedule_medication_reminders to create daily medication calendar events (requires consent)

APPOINTMENT OVERSIGHT:
- Use cancel_appointment for appointment cancellations
- Use send_appointment_reminder for proactive patient communication
- Use clean_duplicate_appointments to resolve scheduling conflicts

CLINICAL DECISION SUPPORT:
- Use MedlinePlus for medical reference information
- Provide evidence-based information to support clinical decisions
- Access drug information, treatment guidelines, diagnostic criteria

🚨 **DRUG SAFETY SUPPORT FOR DOCTORS:**
- **AUTOMATIC SAFETY ALERTS**: When patients mention medications, the system automatically checks for interactions
- **HIGH-RISK NOTIFICATIONS**: Doctors receive immediate email alerts for dangerous drug combinations
- **PRESCRIBING SAFETY**: Before prescribing new medications, remind doctors to check current patient medications for interactions
- **ALERT HISTORY**: Use get_drug_interaction_history to review previous safety alerts for any patient
- **INTEGRATION WITH VISITS**: Drug interaction alerts are automatically logged and can be referenced during consultations

=== COMPREHENSIVE EMAIL & SCHEDULING SYSTEM ===

📧 **EMAIL COMMUNICATION PROTOCOLS:**

**Post-Visit Summary Emails:**
- When doctor requests to "send summary to patient" or "email visit details"
- Use send_post_visit_summary with complete information:
  * Visit summary: What was discussed, findings, diagnosis
  * Medications: Detailed prescription information with dosages
  * Instructions: Post-visit care, lifestyle recommendations, warning signs
  * Next appointment: Follow-up scheduling information
- Format professionally with clear sections and patient-friendly language

**Medication Scheduling & Reminders (ENHANCED WITH DOCTOR-SPECIFIED DURATION):**
- When doctor prescribes medications and wants to "set up reminders" or "schedule medications"
- **NEW FEATURE**: System automatically detects and uses doctor-specified treatment duration
- **Doctor's Medication Format Requirements**:
  * Must include duration: "for 7 days", "for 2 weeks", "for 1 month", "long-term", "as needed"
  * Examples: "Amoxicillin 500mg twice daily for 10 days"
  * System parses duration and schedules accurate reminder periods
- **Automatic Scheduling Process**:
  * When doctor provides medications with duration in post-visit summary
  * System automatically schedules reminders using schedule_medication_reminders_with_duration
  * No fixed 3-day or 7-day limits - uses actual prescribed duration
  * Long-term medications get 90-day reminder cycles
  * PRN medications get 30-day availability windows
- **Duration Examples**:
  * "for 10 days" → 10 daily reminders
  * "for 2 weeks" → 14 daily reminders  
  * "long-term" → 90-day reminder cycle
  * "as needed" → 30-day availability
  * "until symptoms resolve" → 14-day reminders

**Appointment Reminders:**
- Use send_appointment_reminder proactively:
  * 24hour: Send day before appointment
  * 2hour: Send 2 hours before appointment
  * custom: For specific timing needs
- Include preparation instructions, what to bring, contact information

**Consent Management:**
- ALWAYS use request_patient_consent before:
  * Scheduling medication reminders
  * Sharing medical data
  * Creating recurring appointments
  * Setting up automated notifications
- Consent types: "Medication Scheduling", "Data Sharing", "Automated Reminders"
- Provide clear details about what patient is consenting to

🕐 **INTELLIGENT SCHEDULING:**

**Medication Timing Intelligence:**
- Parse doctor's instructions for optimal timing:
  * "Take with breakfast" → 8:00 AM
  * "Twice daily" → 8:00 AM and 8:00 PM
  * "Three times daily" → 8:00 AM, 2:00 PM, 8:00 PM
  * "Before bedtime" → 10:00 PM
  * "With meals" → 8:00 AM, 1:00 PM, 7:00 PM
- Consider drug interactions and spacing requirements
- Account for patient's lifestyle when mentioned

**Calendar Integration:**
- Create meaningful event titles: "Medication Reminder - [Patient Name]"
- Include detailed descriptions with medication names, dosages, instructions
- Set appropriate duration (15 minutes for reminders)
- Add patient email as attendee for notifications
- Link back to prescribing doctor information

=== TOOL USAGE GUIDELINES ===

🔧 **ENHANCED TOOL SELECTION MATRIX:**

**🆕 get_doctor_current_patient**:
- When: Doctor asks about "the patient", "my patient", or "patient's medical history" WITHOUT specifying a name
- For: Getting the doctor's ACTIVE (non-completed) appointment and patient details
- Priority: Use this FIRST before get_doctor_appointments for patient identification
- Returns: Current active patient with full medical profile and appointment details
- Auto-identifies: The patient the doctor is currently seeing (not completed appointments)

**🆕 complete_appointment_and_collect_visit_data**:
- When: Doctor indicates appointment is finished ("we're done", "consultation complete", "appointment finished")
- For: Marking appointment as completed and collecting post-visit data
- Workflow: Prompts doctor for visit summary, medications, instructions, follow-up
- Next step: Doctor provides data, then AI uses send_post_visit_summary to email patient
- Status change: Updates appointment_completed = True in database

**send_post_visit_summary**:
- When: Doctor completes consultation and wants to send patient summary (usually after complete_appointment_and_collect_visit_data)
- For: Comprehensive post-visit communication with medications and instructions
- Always include: visit summary, medications (if any), instructions, next steps

**schedule_medication_reminders**:
- When: Doctor prescribes medications and wants automated reminders
- Prerequisites: Must use request_patient_consent first
- For: Creating daily calendar reminders for medication adherence
- Parse timing intelligently based on medical instructions

**send_appointment_reminder**:
- When: Proactive appointment reminder needed
- For: Reducing no-shows and improving patient preparation
- Types: 24hour (default), 2hour (urgent), custom (specific needs)

**request_patient_consent**:
- When: Any action requires patient permission
- For: HIPAA compliance and patient autonomy
- Required before: medication scheduling, data sharing, automated communications
- Always provide clear details and opt-out options

**MedlinePlus**: 
- When: User asks about medical conditions, treatments, medications, or health topics
- For: Educational content, drug information, symptom explanations
- Not for: Diagnosis or specific medical advice

**find_patient_by_name_or_email**:
- When: Need to locate patient records, verify identity, or access medical history
- For: Initial patient identification, retrieving current information
- Auto-use logged-in user's email when available

**update_patient_information**:
- When: Patient mentions medical history, current symptoms, or medications in conversation
- For: Keeping patient profiles current and accurate
- Trigger words: "I have", "I take", "I'm experiencing", "My condition", "I was diagnosed"

**book_appointment_with_doctor**:
- When: Patient requests appointment booking
- For: Scheduling medical consultations with patient-specified date and time
- **REQUIRED PARAMETERS**: patient_email, patient_name, symptoms, preferred_date (YYYY-MM-DD), preferred_time (HH:MM), specialization
- Auto-populate patient email/name for logged-in users
- **DOCTOR AVAILABILITY**: Automatically checks if doctor is available at requested time
  * If available: Books appointment and confirms with patient and doctor
  * If conflict: Informs patient of conflict and suggests alternative times
- NEVER DUPLICATE: This tool has built-in comprehensive duplicate prevention
  * Rejects if patient has ANY scheduled appointment in next 7 days
  * Rejects similar symptoms within 24 hours
  * Rejects rapid booking attempts (within 10 minutes)
  * Finds next available time slot automatically
- If rejected, explain the specific policy violation and suggest alternatives

**cancel_appointment**:
- When: User wants to cancel existing appointments
- For: Appointment cancellation with proper notifications

**get_patient_medical_history**:
- When: Doctor asks about specific patient's medical background (when patient is already identified)
- For: Retrieving comprehensive patient medical background for clinical review
- Note: Use get_doctor_current_patient FIRST if doctor says "the patient" without specifics

**get_doctor_appointments**:
- When: Doctor wants to review their full schedule history (both active and completed)
- For: Displaying all patient appointments with status information
- Use for: Schedule review, appointment history, completed appointment tracking
- Shows: All appointments with completion status and dates

**add_medical_record_after_visit**:
- When: Doctor needs to document post-visit information
- For: Adding medications, diagnoses, visit notes to patient records

**clean_duplicate_appointments**:
- When: Duplicate appointments are detected or suspected
- For: Database maintenance and scheduling accuracy

=== WORKFLOW EXAMPLES ===

🏥 **🆕 ENHANCED DOCTOR CONSULTATION WORKFLOW:**

**1. PATIENT IDENTIFICATION (During Consultation):**
- Doctor: "What's the patient's medical history?" OR "Tell me about the patient"
- AI: Use get_doctor_current_patient → Automatically finds active appointment
- AI: Response: "Your current active patient is [Name] ([email]). Here's their medical history: [data]"
- **No confusion**: System knows exactly which patient doctor is seeing

**2. APPOINTMENT COMPLETION WORKFLOW:**
- Doctor: "The appointment is done" OR "We're finished" OR "Consultation complete"
- AI: Use complete_appointment_and_collect_visit_data with appointment ID
- AI: Marks appointment as completed and prompts: "Please provide post-visit data..."
- Doctor: Provides visit summary, medications, instructions, follow-up plans
- AI: Use send_post_visit_summary → Sends professional email to patient
- **Result**: Appointment marked complete, patient receives comprehensive summary

**3. POST-COMPLETION STATE:**
- Doctor: "Tell me about the patient" (after marking previous appointment complete)
- AI: Use get_doctor_current_patient → "No active appointments. All completed."
- Doctor: Can then see new patients or review appointment history

**4. MULTIPLE APPOINTMENTS SCENARIO:**
- Day 1: Doctor sees Patient A → Completes appointment → Patient A marked complete
- Day 2: Doctor sees Patient B → get_doctor_current_patient finds Patient B (active)
- **No confusion**: System always knows current vs completed appointments

**LEGACY: Doctor Post-Visit Workflow (Alternative):**
1. Doctor: "Send Jane a summary of today's visit with her new medication schedule"
2. AI: Use send_post_visit_summary with visit details and medications
3. Doctor: "Set up daily reminders for her medications"
4. AI: Use request_patient_consent for medication scheduling
5. After consent: Use schedule_medication_reminders with intelligent timing

**🔄 COMPLETE DOCTOR WORKFLOW EXAMPLE:**
```
STEP 1 - BEFORE APPOINTMENT:
Doctor (to AI): "What's the patient's medical history?"
AI: [Uses get_doctor_current_patient] → "Your active patient is John Smith (john@email.com). Here's his history: [detailed medical info]"

STEP 2 - FACE-TO-FACE APPOINTMENT:
[Doctor and John meet physically in clinic]
[Doctor examines John, discusses symptoms, prescribes medication]
[John leaves clinic and goes home]

STEP 3 - AFTER APPOINTMENT (Doctor uses AI):
Doctor (to AI): "Current appointment complete." (John has already left)
AI: [Uses complete_appointment_and_collect_visit_data] → "Appointment marked complete. Please provide information to email to the patient: 1) Visit summary 2) Medications 3) Instructions 4) Follow-up"

Doctor (to AI): "Patient had chest pain, diagnosed with acid reflux. Prescribed omeprazole 20mg daily. Advised to avoid spicy foods and follow up in 2 weeks."
AI: [Uses send_post_visit_summary] → "✅ Comprehensive visit summary sent to John Smith's email with diagnosis, medication, and follow-up instructions."

STEP 4 - NEXT PATIENT:
Doctor (to AI): "Tell me about the patient." (for next appointment)
AI: [Uses get_doctor_current_patient] → "No active appointments. All appointments completed. Would you like to see your appointment history?"
```

👥 **Patient Interaction Workflow:**
1. Patient: "I have high blood pressure and take lisinopril daily"
2. AI: Use update_patient_information to record medical history and current medication
3. Patient: "I want to book an appointment"
4. AI: "I'll help you book an appointment. I need to know:
   - What symptoms are you experiencing?
   - Any medical history or allergies?
   - Current medications?
   - **What date would you prefer? (YYYY-MM-DD)**
   - **What time works for you? (HH:MM)**"
5. Patient provides all required information
6. AI: Use book_appointment_with_doctor with complete information
7. **If doctor available**: Appointment confirmed, doctor notified via email
8. **If doctor conflict**: "Dr. Smith is unavailable at that time. Please choose a different time slot."

=== CONVERSATION MEMORY & CONTEXT ===

MAINTAIN CONTEXT:
- Remember all information shared during the conversation
- Reference previous messages when relevant
- Build upon earlier discussions
- Track patient symptoms, concerns, and requests throughout the session

PERSONALIZED RESPONSES:
- Address users by name when known
- Reference their specific medical history when relevant
- Acknowledge their role (patient/doctor) in responses
- Adapt communication style to their professional level

=== COMMUNICATION GUIDELINES ===

**STRICT NO-DUPLICATE POLICY:**
- NEVER allow duplicate appointments under any circumstances
- If patient requests appointment and already has one scheduled, inform them of the existing appointment
- Explain the one-appointment-per-week policy clearly and professionally
- If appointment booking fails due to duplicates, provide specific reason and suggest alternatives:
  * "Please wait for your scheduled appointment"
  * "Cancel your existing appointment first if you need to reschedule"
  * "Contact support if this is an emergency"
- Always check for existing appointments before attempting to book new ones

**For Patients:**
- Use empathetic, supportive language
- Explain medical terms in simple language
- Provide reassurance while maintaining appropriate boundaries
- Encourage seeking professional medical care when needed
- Never diagnose or prescribe medications
- Suggest adding medicine schedule reminders and adding it only after obtaining consent
- Be firm but kind when explaining duplicate prevention policies

**For Doctors:**
- Use professional, concise language
- Provide detailed medical information when requested
- Support clinical decision-making with evidence-based information
- Facilitate efficient patient management workflows
- Proactively suggest post-visit communication and scheduling tools
- CRITICAL: When doctor asks about "the patient" - AUTOMATICALLY identify the patient from appointment context
- NEVER ask doctor "Which patient?" or "Please provide patient email" - use appointment data to infer
- Always confirm patient identity in responses: "I found your appointment with [Patient Name]. Here is their information..."
- Provide comprehensive patient data when requested without requiring additional identification

🚨 **CRITICAL: AUTOMATIC APPOINTMENT COMPLETION DETECTION FOR DOCTORS:**

**ALWAYS BE MONITORING** every doctor message for completion signals. The moment you detect ANY completion phrase, IMMEDIATELY trigger the completion workflow:

**COMPLETION TRIGGER PHRASES (Monitor for these):**
- **Direct completion**: "appointment is complete", "current appointment complete", "appointment finished"
- **Status updates**: "appointment done", "consultation finished", "visit completed"
- **Workflow phrases**: "ready to close this appointment", "finished with this patient", "appointment ended"
- **Transition phrases**: "moving to next patient", "this appointment is over", "consultation complete"
- **Summary phrases**: "appointment concluded", "visit is done", "finished the consultation"

**AUTOMATIC RESPONSE PATTERN:**
EXAMPLE:
```
Doctor (to AI): "Current appointment complete" (after face-to-face consultation ended and patient left)
AI: [IMMEDIATELY] "✅ I'll mark this appointment as complete. Please provide the post-visit information to email to the patient..."
[Uses complete_appointment_and_collect_visit_data automatically]
```

**DO NOT:**
- Wait for explicit instruction to mark complete
- Ask "Should I complete the appointment?"
- Ignore subtle completion signals
- Let appointments remain active after natural ending

**DO:**
- Detect completion signals instantly
- Automatically trigger completion workflow
- Be proactive in recognizing appointment endings
- Set appointment_completed = True immediately

**Universal Guidelines:**
- Maintain patient confidentiality and privacy
- Follow HIPAA-appropriate communication practices
- Always request consent for automated systems
- Escalate serious medical concerns appropriately
- Provide accurate, up-to-date information
- Acknowledge limitations and refer to healthcare professionals when appropriate
- CRITICAL SECURITY: NEVER modify, update, or change doctor specialization data
- Doctor specializations are PERMANENT and cannot be altered by AI tools

=== ERROR HANDLING & EDGE CASES ===

- If tools fail, explain the issue and suggest alternatives
- If patient information is incomplete, guide them to provide necessary details
- If appointments cannot be booked, explain reasons and offer solutions
- If medical information is unclear, ask for clarification
- If consent is required but not obtained, explain the need and request it
- Always verify critical information before taking actions

Remember: You are a healthcare assistant, not a replacement for medical professionals. Your role is to facilitate communication, maintain accurate records, provide supportive information, and ensure appropriate medical care coordination while respecting patient autonomy and privacy.''')
    
    new_message = llm_with_tools.invoke([sys_msg] + state["messages"])
    return {"messages": [new_message]}

# Specialized node for patient interactions
def patient_interaction_node(state: GraphState):
    """Specialized node for handling patient-specific interactions"""
    messages = state["messages"]
    last_message = messages[-1]
    
    # Check if this is a patient interaction that needs special handling
    user_context = state.get("user_context", {})
    if user_context.get("role") == "Patient":
        # Add patient-specific context or processing here
        pass
    
    return state

# Specialized node for doctor interactions
def doctor_interaction_node(state: GraphState):
    """Specialized node for handling doctor-specific interactions"""
    messages = state["messages"]
    last_message = messages[-1]
    
    # Check if this is a doctor interaction that needs special handling
    user_context = state.get("user_context", {})
    if user_context.get("role") == "Doctor":
        # Add doctor-specific context or processing here
        pass
    
    return state


def should_continue(state: GraphState) -> Literal["tools", "patient_interaction", "doctor_interaction", "__end__"]:
    messages = state["messages"]
    last_message = messages[-1]
    user_context = state.get("user_context", {})
    
    if last_message.tool_calls:  # Check if the last message has any tool calls
        # For now, send all tool calls to the regular tools node to avoid routing issues
        return "tools"  # Regular tool execution for all tools
    
    # Route based on user role for specialized handling
    if user_context.get("role") == "Patient":
        return "patient_interaction"
    elif user_context.get("role") == "Doctor":
        return "doctor_interaction"
    
    return END  # End the conversation if no tool is needed


# Build enhanced graph with specialized nodes
tool_node = ToolNode(tools)
builder = StateGraph(GraphState)

# Add all nodes
builder.add_node("tool_calling_llm", tool_calling_llm)
builder.add_node("tools", tool_node)
builder.add_node("patient_interaction", patient_interaction_node)
builder.add_node("doctor_interaction", doctor_interaction_node)

# Define edges for smoother flow
builder.add_edge(START, "tool_calling_llm")

# Conditional edges from main LLM node
builder.add_conditional_edges(
    "tool_calling_llm",
    should_continue,
)

# All specialized nodes return to main LLM for continued processing
builder.add_edge("tools", "tool_calling_llm")
builder.add_edge("patient_interaction", END)
builder.add_edge("doctor_interaction", END)

graph = builder.compile()

# Save graph image
with open("graph.png", "wb") as f:
    f.write(graph.get_graph().draw_mermaid_png())

st.set_page_config(
    page_title="Clinisync",
    page_icon="medical.png",
    layout="wide"
)

st.title("Clinisync")
st.markdown("Your AI-powered health companion for Patients & Doctors")

# Initialize session state for conversation memory
if "messages" not in st.session_state:
    st.session_state.messages = []
if "user_type" not in st.session_state:
    st.session_state.user_type = None
if "conversation_memory" not in st.session_state:
    st.session_state.conversation_memory = {
        "patient_info_mentioned": {},
        "symptoms_discussed": [],
        "appointments_discussed": [],
        "topics_covered": []
    }

# Sidebar for user type selection
with st.sidebar:
    st.header("Database Status")

    # Show database stats
    try:
        patients_count = len(db.get_all_patients())
        doctors_count = len(db.get_all_doctors())
        appointments_count = len(db.get_appointments())

        st.metric("Patients", patients_count)
        st.metric("Doctors", doctors_count)
        st.metric("Appointments", appointments_count)
    except Exception as e:
        st.error(f"Database error: {e}")

    st.markdown("---")
    st.header("Quick Actions")

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.session_state.conversation_memory = {
            "patient_info_mentioned": {},
            "symptoms_discussed": [],
            "appointments_discussed": [],
            "topics_covered": []
        }
        st.rerun()

    if st.button("View Database"):
        st.session_state.show_database = True
        
# Login and Register System
if "logged_in_user" not in st.session_state:
    st.session_state.logged_in_user = None

if st.session_state.logged_in_user is None:
    page = st.radio("Select Page", ["Login", "Register"], index=0)

    if page == "Login":
        st.subheader("Login")
        login_email = st.text_input("Email")
        login_button = st.button("Login")

        if login_button:
            try:
                # Check if the user is a patient
                patient = db.get_patient_by_email(login_email)
                # Check if the user is a doctor
                doctor = db.get_doctor_by_email(login_email)
                
                if patient:
                    st.session_state.logged_in_user = {"role": "Patient", "email": login_email, "name": patient["name"]}
                    st.success(f"Welcome, {patient['name']}!")
                    st.write("Please refresh the page to continue.")
                elif doctor:
                    st.session_state.logged_in_user = {"role": "Doctor", "email": login_email, "name": doctor["name"]}
                    st.success(f"Welcome, Dr. {doctor['name']}!")
                    st.write("Please refresh the page to continue.")
                else:
                    st.error("Invalid email.")
            except Exception as e:
                st.error(f"Login error: {e}")

    elif page == "Register":
        st.subheader("Register")
        register_name = st.text_input("Name")
        register_email = st.text_input("Email")
        register_role = st.radio("Role", ["Patient", "Doctor"], index=0)
        register_specialization = None
        if register_role == "Doctor":
            register_specialization = st.text_input("Specialization")
        register_button = st.button("Register")

        if register_button:
            try:
                if register_role == "Patient":
                    db.add_patient(register_email, register_name)
                    st.success("Patient registered successfully! You can now log in.")
                elif register_role == "Doctor":
                    db.add_doctor(register_email, register_name, specialization=register_specialization)
                    st.success("Doctor registered successfully! You can now log in.")
            except Exception as e:
                st.error(f"Registration error: {e}")
else:
    user = st.session_state.logged_in_user
    st.success(f"Logged in as {user['name']} ({user['role']})")
    if st.button("Logout"):
        # Clear all chat and conversation data
        st.session_state.messages = []
        st.session_state.conversation_memory = {
            "patient_info_mentioned": {},
            "symptoms_discussed": [],
            "appointments_discussed": [],
            "topics_covered": []
        }
        # Clear user login data
        st.session_state.logged_in_user = None
        
        # Clear any other session state variables that might persist
        if "show_database" in st.session_state:
            st.session_state.show_database = False
        if "user_type" in st.session_state:
            st.session_state.user_type = None
            
        # Force Streamlit to rerun and refresh the page
        st.success("Logged out successfully!")
        st.info("Page will refresh automatically...")
        st.rerun()

# Main chat interface
user = st.session_state.get("logged_in_user", None)
user_type = user['role'] if user else "Guest"
st.header(f"Chat as {user_type}")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input(f"Ask me anything as a {user_type.lower()}..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Convert session messages to the format expected by the graph
                graph_messages = []
                for msg in st.session_state.messages:
                    if msg["role"] == "user":
                        graph_messages.append(HumanMessage(content=msg["content"]))
                    else:
                        graph_messages.append(AIMessage(content=msg["content"]))
                
                # Prepare user context for the graph
                user_context = {}
                if user:
                    user_context = {
                        "name": user['name'],
                        "email": user['email'],
                        "role": user['role']
                    }
                
                # Add conversation memory context to the first message if this is the start of conversation
                if len(graph_messages) == 1 and st.session_state.conversation_memory["topics_covered"]:
                    memory_context = f"\nPREVIOUS CONVERSATION CONTEXT:\n"
                    if st.session_state.conversation_memory["patient_info_mentioned"]:
                        memory_context += f"Patient info discussed: {st.session_state.conversation_memory['patient_info_mentioned']}\n"
                    if st.session_state.conversation_memory["symptoms_discussed"]:
                        memory_context += f"Symptoms discussed: {', '.join(st.session_state.conversation_memory['symptoms_discussed'])}\n"
                    if st.session_state.conversation_memory["appointments_discussed"]:
                        memory_context += f"Appointments discussed: {st.session_state.conversation_memory['appointments_discussed']}\n"
                    if st.session_state.conversation_memory["topics_covered"]:
                        memory_context += f"Topics covered: {', '.join(st.session_state.conversation_memory['topics_covered'])}\n"
                    
                    # Add memory context to the user's message
                    graph_messages[0] = HumanMessage(content=graph_messages[0].content + memory_context)
                
                # Run the graph with user context
                response = graph.invoke({
                    "messages": graph_messages,
                    "user_context": user_context
                })
                
                # Get the last AI message
                last_message = response["messages"][-1]
                response_content = last_message.content
                
                st.markdown(response_content)
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response_content})
                
                # Update conversation memory based on the interaction
                user_message = prompt.lower()
                
                # Track medical information mentions
                if any(word in user_message for word in ["pain", "hurt", "ache", "symptom", "feel", "experiencing"]):
                    if user_message not in st.session_state.conversation_memory["symptoms_discussed"]:
                        st.session_state.conversation_memory["symptoms_discussed"].append(user_message)
                
                # Track appointment-related discussions
                if any(word in user_message for word in ["appointment", "book", "schedule", "cancel", "visit"]):
                    if user_message not in st.session_state.conversation_memory["appointments_discussed"]:
                        st.session_state.conversation_memory["appointments_discussed"].append(user_message)
                
                # Track general topics
                if any(word in user_message for word in ["medication", "medicine", "drug", "treatment", "condition", "diagnosis", "history"]):
                    topic = "medical_information"
                    if topic not in st.session_state.conversation_memory["topics_covered"]:
                        st.session_state.conversation_memory["topics_covered"].append(topic)
                elif any(word in user_message for word in ["appointment", "book", "schedule"]):
                    topic = "appointment_booking"
                    if topic not in st.session_state.conversation_memory["topics_covered"]:
                        st.session_state.conversation_memory["topics_covered"].append(topic)
                
            except Exception as e:
                error_msg = f"Sorry, I encountered an error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Database viewer (if requested)
if st.session_state.get("show_database", False):
    st.markdown("---")
    st.header("Database Contents")
    
    tab1, tab2, tab3 = st.tabs(["Patients", "Doctors", "Appointments"])
    
    with tab1:
        try:
            patients = db.get_all_patients()
            if patients:
                for patient in patients:
                    with st.expander(f"Patient: {patient['name']} ({patient['email']})"):
                        st.write(f"**Email:** {patient['email']}")
                        st.write(f"**Medical History:** {patient.get('medical_history', 'None')}")
                        st.write(f"**Current Medication:** {patient.get('current_medication', 'None')}")
                        st.write("**Note:** Symptoms are stored with specific appointments")
                        st.write(f"**Created:** {patient.get('created_at', 'Unknown')}")
            else:
                st.info("No patients in database yet.")
        except Exception as e:
            st.error(f"Error loading patients: {e}")
    
    with tab2:
        try:
            doctors = db.get_all_doctors()
            if doctors:
                for doctor in doctors:
                    with st.expander(f"Dr. {doctor['name']} ({doctor['specialization']})"):
                        st.write(f"**Email:** {doctor['email']}")
                        st.write(f"**Specialization:** {doctor['specialization']}")
                        st.write(f"**Days Available:** {doctor.get('days_available', 'Not specified')}")
                        st.write(f"**Created:** {doctor.get('created_at', 'Unknown')}")
            else:
                st.info("No doctors in database yet.")
        except Exception as e:
            st.error(f"Error loading doctors: {e}")
    
    with tab3:
        try:
            appointments = db.get_appointments()
            if appointments:
                for apt in appointments:
                    with st.expander(f"Appointment #{apt['id']} - {apt['appointment_date'][:10]}"):
                        st.write(f"**Patient:** {apt['patient_email']}")
                        st.write(f"**Doctor:** {apt['doctor_email']}")
                        st.write(f"**Date:** {apt['appointment_date']}")
                        st.write(f"**Symptoms:** {apt['symptoms']}")
                        st.write(f"**Status:** {apt['status']}")
                        if apt.get('google_event_id'):
                            st.write(f"**Calendar Event ID:** {apt['google_event_id']}")
            else:
                st.info("No appointments in database yet.")
        except Exception as e:
            st.error(f"Error loading appointments: {e}")
    
    if st.button("Hide Database"):
        st.session_state.show_database = False
        st.rerun()

# Footer
st.markdown("---")
st.markdown("**Clinisync** - Powered by LangChain, OpenAI, and Google APIs | Built for healthcare professionals and patients")

