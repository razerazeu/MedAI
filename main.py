import streamlit as st
import os
import requests
from typing import TypedDict, Literal, Annotated
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
from database import db
import pickle
from google_auth_oauthlib.flow import InstalledAppFlow
import base64
import pytz

from google_calendar_auth import GoogleCalendarManager, authenticate_google_calendar, is_calendar_authenticated
from gmail_integration import GmailManager, authenticate_gmail, is_gmail_authenticated

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
        if patient.get('current_symptoms'):
            result += "🩺 CURRENT SYMPTOMS:\n"
            result += f"- {patient['current_symptoms']}\n\n"
        
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
            return f"Found patient: {patient['name']}\nEmail: {patient['email']}\nMedical History: {patient.get('medical_history', 'None')}\nCurrent Symptoms: {patient.get('current_symptoms', 'None')}"

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
        
        if not doctor_appointments:
            return f"No appointments found for Dr. {doctor_name} ({doctor_email})"
        
        result = f"📅 Appointments for Dr. {doctor_name}:\n\n"
        
        # Sort appointments by date
        doctor_appointments.sort(key=lambda x: x['appointment_date'])
        
        for apt in doctor_appointments:
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
def book_appointment_with_doctor(patient_email: str, patient_name: str, symptoms: str, specialization: str = "General Medicine") -> str:
    """Book an appointment for a patient with an available doctor based on symptoms and specialization needed."""
    try:
        print(f"🔍 Comprehensive duplicate check for {patient_email}...")  # Debug log
        
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
        
        # Smart scheduling: Find next available slot
        appointment_time = datetime.now() + timedelta(days=1)  # Start with tomorrow
        appointment_time = appointment_time.replace(hour=10, minute=0, second=0, microsecond=0)  # 10 AM
        
        # Check if the selected time slot is already taken by the doctor
        doctor_appointments_at_time = [
            apt for apt in all_appointments
            if apt['doctor_email'] == selected_doctor['email']
            and apt['status'] == 'scheduled'
            and abs((datetime.fromisoformat(apt['appointment_date'].replace('Z', '+00:00').replace('+00:00', '')) - appointment_time).total_seconds()) < 3600  # Within 1 hour
        ]
        
        # If slot is taken, find next available slot
        while doctor_appointments_at_time and appointment_time < datetime.now() + timedelta(days=30):  # Look up to 30 days ahead
            appointment_time += timedelta(hours=1)  # Try next hour
            doctor_appointments_at_time = [
                apt for apt in all_appointments
                if apt['doctor_email'] == selected_doctor['email']
                and apt['status'] == 'scheduled'
                and abs((datetime.fromisoformat(apt['appointment_date'].replace('Z', '+00:00').replace('+00:00', '')) - appointment_time).total_seconds()) < 3600
            ]
        
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
            current_medication=patient.get('current_medication'),
            current_symptoms=patient.get('current_symptoms')
        )
        
        return f"✅ Medical record added successfully for {patient['name']} ({patient_email})"
        
    except Exception as e:
        return f"Error adding medical record: {str(e)}"

@tool
def update_patient_information(patient_email: str, medical_history: str = None, current_symptoms: str = None, current_medication: str = None) -> str:
    """Update patient's medical history, current symptoms, or current medication when they share this information during conversation."""
    try:
        patient = db.get_patient_by_email(patient_email)
        if not patient:
            return f"Patient not found: {patient_email}"
        
        updated_fields = []
        
        # Update medical history
        if medical_history:
            current_history = patient.get('medical_history', '') or ''
            timestamp = datetime.now().strftime('%Y-%m-%d')
            new_entry = f"[{timestamp}] {medical_history}"
            
            if current_history:
                updated_history = current_history + "\n" + new_entry
            else:
                updated_history = new_entry
            
            patient['medical_history'] = updated_history
            updated_fields.append("medical history")
        
        # Update current symptoms
        if current_symptoms:
            patient['current_symptoms'] = current_symptoms
            updated_fields.append("current symptoms")
        
        # Update current medication
        if current_medication:
            patient['current_medication'] = current_medication
            updated_fields.append("current medication")
        
        # Save to database
        if updated_fields:
            db.add_patient(
                email=patient_email,
                name=patient['name'],
                medical_history=patient.get('medical_history'),
                current_medication=patient.get('current_medication'),
                current_symptoms=patient.get('current_symptoms'),
                role=patient.get('role', 'Patient')
            )
            
            fields_str = ", ".join(updated_fields)
            return f"✅ Updated {fields_str} for {patient['name']} ({patient_email})"
        else:
            return "No information provided to update."
        
    except Exception as e:
        return f"Error updating patient information: {str(e)}"

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
def send_post_visit_summary(patient_email: str, doctor_email: str, visit_summary: str, medications: str = None, instructions: str = None, next_appointment: str = None) -> str:
    """Send a comprehensive post-visit summary email to patient with medical information, medications, and follow-up instructions."""
    try:
        patient = db.get_patient_by_email(patient_email)
        doctor = db.get_doctor_by_email(doctor_email)
        
        if not patient or not doctor:
            return f"Patient or doctor not found in database"
        
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
        
        # Send email
        if send_email_via_google(patient_email, email_subject, email_body):
            return f"✅ Post-visit summary sent successfully to {patient['name']} ({patient_email})"
        else:
            return f"❌ Failed to send post-visit summary to {patient_email}"
            
    except Exception as e:
        return f"Error sending post-visit summary: {str(e)}"

@tool
def schedule_medication_reminders(patient_email: str, medication_schedule: str, doctor_email: str, duration_days: int = 30) -> str:
    """Schedule medication reminders and add them to patient's treatment plan. Requires patient consent before scheduling."""
    try:
        patient = db.get_patient_by_email(patient_email)
        if not patient:
            return f"Patient not found: {patient_email}"
        
        # Parse medication schedule and create calendar events
        medications = []
        schedule_lines = medication_schedule.split('\n')
        
        for line in schedule_lines:
            if line.strip():
                medications.append(line.strip())
        
        if not medications:
            return "No medications found in the schedule"
        
        # Create medication reminders for the specified duration
        calendar_events = []
        current_date = datetime.now().replace(hour=8, minute=0, second=0, microsecond=0)  # Start at 8 AM
        
        for day in range(duration_days):
            event_date = current_date + timedelta(days=day)
            
            # Create daily medication reminder
            try:
                calendar_service = get_google_calendar_service()
                event = create_calendar_event(
                    calendar_service,
                    f"Medication Reminder - {patient['name']}",
                    f"Daily Medications:\n{medication_schedule}\n\nPrescribed by: Dr. {doctor_email}",
                    event_date,
                    event_date + timedelta(minutes=15),
                    patient_email
                )
                calendar_events.append(event.get('id'))
            except Exception as e:
                print(f"Error creating calendar event for day {day}: {e}")
        
        # Update patient's current medication in database
        db.add_patient(
            email=patient_email,
            name=patient['name'],
            medical_history=patient.get('medical_history'),
            current_medication=medication_schedule,
            current_symptoms=patient.get('current_symptoms'),
            role=patient.get('role', 'Patient')
        )
        
        return f"✅ Scheduled {len(calendar_events)} medication reminders for {patient['name']} over {duration_days} days.\n\nMedication Schedule:\n{medication_schedule}\n\nReminders will appear in their calendar daily at 8:00 AM."
        
    except Exception as e:
        return f"Error scheduling medication reminders: {str(e)}"

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

        # Save the credentials for the next run
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
        print(f"Preparing to send email to {to_email}...")  # Debug log
        
        # Check if Gmail is authenticated
        if not gmail_manager.is_authenticated:
            print("Gmail not authenticated. Attempting to authenticate...")
            if not gmail_manager.authenticate():
                print("Gmail authentication failed.")
                return False
        
        # Use the Gmail manager's service
        if not gmail_manager.service:
            print("Gmail service not available.")
            return False

        # Create message
        message = MIMEMultipart()
        message['to'] = to_email
        message['subject'] = subject
        message.attach(MIMEText(body, 'plain'))

        # Encode message properly for Gmail API
        raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode()

        # Send email using Gmail manager's service
        gmail_manager.service.users().messages().send(
            userId='me', 
            body={'raw': raw_message}
        ).execute()

        print(f"Email sent successfully to {to_email}")  # Debug log
        return True

    except Exception as e:
        print(f"Error sending email to {to_email}: {e}")  # Debug log
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
    check_patient_existing_appointments,
    book_appointment_with_doctor,
    add_medical_record_after_visit,
    update_patient_information,
    cancel_appointment,
    send_post_visit_summary,
    schedule_medication_reminders,
    send_appointment_reminder,
    request_patient_consent,
    clean_duplicate_appointments
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
  * Current symptoms: "I have a headache", "My knee hurts", "I've been feeling dizzy"
  * Current medications: "I take aspirin daily", "I'm on insulin", "I use an inhaler"
- Update their profile immediately without asking for confirmation
- Acknowledge the update: "I've updated your medical profile with this information"

APPOINTMENT BOOKING:
- Use book_appointment_with_doctor for scheduling
- For logged-in users: automatically use their email and name from USER CONTEXT
- Ask only for: symptoms description, preferred specialization (if specific)
- Match symptoms to appropriate medical specializations
- STRICT DUPLICATE PREVENTION: The system has comprehensive duplicate prevention
  * Only ONE appointment per patient per week is allowed
  * Cannot book similar symptoms within 24 hours
  * Cannot book multiple appointments within 10 minutes
  * The booking function will automatically reject duplicates with clear explanations
- If booking fails due to duplicates, explain the policy and suggest alternatives
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

CRITICAL: PATIENT CONTEXT AWARENESS FOR DOCTORS:
- When a doctor asks about "the patient" or "patient's medical history" without specifying a name/email:
  * First check the doctor's appointments using get_doctor_appointments
  * Look for TODAY's appointments or the most recent scheduled appointment
  * Automatically identify the patient from the appointment context
  * Use that patient's email to retrieve their medical history with get_patient_medical_history
  * DO NOT ask the doctor for the patient's email or name - infer it from appointment context
- When doctor asks: "Is there any medical history available for the patient?"
  * Response flow: "Let me check your recent appointments to identify the patient... I found your appointment with [Patient Name]. Here is their medical history: [data]"
- When doctor asks about "this patient" or "my patient":
  * Automatically look up the doctor's current/recent appointments
  * Use the appointment data to identify which patient they're referring to
  * Retrieve that specific patient's information without asking for clarification
- When doctor asks about "the patient" without context:
    * Use get_doctor_appointments to find their most recent appointment
    * If multiple appointments exist, prioritize the most recent scheduled one
    * Use that patient's email to retrieve medical history
- When a doctor mentions the name of the patient:
    * Use find_patient_by_name_or_email to locate the patient record
    * If found, retrieve their medical history with get_patient_medical_history
    * If not found, inform the doctor: "There is no patient found with that name in the system." and ask him if he wants a list of his previous appointments to identify the patient.

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

POST-VISIT COMMUNICATION:
- Use send_post_visit_summary to send comprehensive visit summaries to patients
- Include: visit summary, prescribed medications, post-visit instructions, next appointment details
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

**Medication Scheduling & Reminders:**
- When doctor prescribes medications and wants to "set up reminders" or "schedule medications"
- STEP 1: Use request_patient_consent for medication scheduling
  * Explain: "Daily medication reminders via calendar invitations"
  * Details: Medication names, timing, duration, calendar access
- STEP 2: After consent received, use schedule_medication_reminders
  * Parse medication schedule (e.g., "Aspirin 81mg daily at 8 AM, Metformin 500mg twice daily at 8 AM and 8 PM")
  * Create calendar events for specified duration (default 30 days)
  * Set appropriate timing based on prescription

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

**send_post_visit_summary**:
- When: Doctor completes consultation and wants to send patient summary
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
- For: Scheduling medical consultations
- Auto-populate patient email/name for logged-in users
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
- When: Doctor asks about patient's medical background, history, medications, or conditions
- CRITICAL: If doctor asks about "the patient" without specifying name/email:
  * First use get_doctor_appointments to find the doctor's recent appointments
  * Identify the most relevant patient (today's appointment or most recent scheduled)
  * Use that patient's email to retrieve their medical history
  * DO NOT ask doctor for patient email - infer it from appointment context
- For: Retrieving comprehensive patient medical background for clinical review
- Auto-identify patient from appointment context when doctor references "the patient"

**get_doctor_appointments**:
- When: Doctor wants to review their schedule OR when doctor asks about "the patient" without specifics
- For: Displaying upcoming patient appointments AND identifying patient context
- Use for context: When doctor says "the patient" - check appointments to identify which patient
- Priority: Today's appointments first, then upcoming scheduled appointments

**add_medical_record_after_visit**:
- When: Doctor needs to document post-visit information
- For: Adding medications, diagnoses, visit notes to patient records

**clean_duplicate_appointments**:
- When: Duplicate appointments are detected or suspected
- For: Database maintenance and scheduling accuracy

=== WORKFLOW EXAMPLES ===

🏥 **Doctor Post-Visit Workflow:**
1. Doctor: "Send Jane a summary of today's visit with her new medication schedule"
2. AI: Use send_post_visit_summary with visit details and medications
3. Doctor: "Set up daily reminders for her medications"
4. AI: Use request_patient_consent for medication scheduling
5. After consent: Use schedule_medication_reminders with intelligent timing

**CRITICAL: Doctor Patient Context Workflow:**
1. Doctor: "Is there any medical history available for the patient?"
2. AI: First use get_doctor_appointments to check doctor's recent appointments
3. AI: Identify the most relevant patient (today's appointment or recent scheduled)
4. AI: Use get_patient_medical_history with that patient's email
5. AI: Response: "I found your appointment with [Patient Name] ([email]). Here is their medical history: [data]"
- NEVER ask doctor for patient email/name when they say "the patient"
- ALWAYS infer patient context from appointment data
- ALWAYS provide patient identification in response for clarity

**Doctor Schedule Review Workflow:**
1. Doctor: "What appointments do I have today?"
2. AI: Use get_doctor_appointments with doctor's email
3. AI: Present organized schedule with patient details, symptoms, times
4. Doctor: "Tell me about the patient's background" (referring to appointment list)
5. AI: Use patient email from the appointment to get medical history (no need to ask which patient)

👥 **Patient Interaction Workflow:**
1. Patient: "I have high blood pressure and take lisinopril daily"
2. AI: Use update_patient_information to record medical history and current medication
3. Patient: "I want to book an appointment"
4. AI: Use book_appointment_with_doctor with logged-in user's info
5. AI: Automatically send appointment confirmation and preparation instructions

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

**Universal Guidelines:**
- Maintain patient confidentiality and privacy
- Follow HIPAA-appropriate communication practices
- Always request consent for automated systems
- Escalate serious medical concerns appropriately
- Provide accurate, up-to-date information
- Acknowledge limitations and refer to healthcare professionals when appropriate

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

# Node for handling email and scheduling workflows
def communication_workflow_node(state: GraphState):
    """Specialized node for handling complex email and scheduling workflows"""
    messages = state["messages"]
    last_message = messages[-1]
    
    # Check if the last interaction involved email or scheduling tools
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        for tool_call in last_message.tool_calls:
            tool_name = tool_call.get('name', '')
            if tool_name in ['send_post_visit_summary', 'schedule_medication_reminders', 'request_patient_consent']:
                # Add workflow-specific processing here
                pass
    
    return state


def should_continue(state: GraphState) -> Literal["tools", "patient_interaction", "doctor_interaction", "communication_workflow", "__end__"]:
    messages = state["messages"]
    last_message = messages[-1]
    user_context = state.get("user_context", {})
    
    if last_message.tool_calls:  # Check if the last message has any tool calls
        # Check if this involves communication/scheduling tools
        for tool_call in last_message.tool_calls:
            tool_name = tool_call.get('name', '')
            if tool_name in ['send_post_visit_summary', 'schedule_medication_reminders', 'request_patient_consent', 'send_appointment_reminder']:
                return "communication_workflow"
        
        return "tools"  # Regular tool execution
    
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
builder.add_node("communication_workflow", communication_workflow_node)

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
builder.add_edge("communication_workflow", "tool_calling_llm")

graph = builder.compile()

# Save graph image
with open("graph.png", "wb") as f:
    f.write(graph.get_graph().draw_mermaid_png())

st.set_page_config(
    page_title="Medical Assistant AI",
    page_icon="medical.png",
    layout="wide"
)

st.title("Medical Assistant AI")
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
        st.session_state.logged_in_user = None
        st.write("Please refresh the page to log out.")

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
                        st.write(f"**Current Symptoms:** {patient.get('current_symptoms', 'None')}")
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
st.markdown("**MedAI** - Powered by LangChain, OpenAI, and Google APIs | Built for healthcare professionals and patients")

