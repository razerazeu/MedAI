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
        
        result = f"Medical History for {patient['name']} ({patient_email}):\n\n"
        
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
        print(f"🔍 Checking for duplicate appointments for {patient_email}...")  # Debug log
        
        # Check for existing appointments for this patient with similar symptoms in the last 24 hours
        all_appointments = db.get_appointments()
        recent_appointments = [
            apt for apt in all_appointments 
            if apt['patient_email'] == patient_email 
            and apt['symptoms'].strip().lower() == symptoms.strip().lower()  # More precise comparison
            and apt['status'] == 'scheduled'
            and (datetime.now() - datetime.fromisoformat(apt['created_at'].replace('Z', '+00:00').replace('+00:00', ''))).total_seconds() < 3600  # 1 hour instead of 24
        ]
        
        if recent_appointments:
            print(f"❌ Duplicate appointment detected for {patient_email}")  # Debug log
            return f"❌ Duplicate appointment detected! You already have a similar appointment scheduled:\n\nAppointment ID: {recent_appointments[0]['id']}\nSymptoms: {recent_appointments[0]['symptoms']}\nDate: {recent_appointments[0]['appointment_date']}\n\nIf you need to modify this appointment, please contact us."
        
        print(f"✅ No duplicates found. Proceeding with booking...")  # Debug log
        
        # Add patient to database if not exists
        patient_id = db.add_patient(patient_email, patient_name)
        
        # Find doctors by specialization
        doctors = db.get_doctors_by_specialization(specialization)
        if not doctors:
            return f"No doctors found for specialization: {specialization}"
        
        selected_doctor = doctors[0]
        
        appointment_time = datetime.now() + timedelta(days=1)  # Tomorrow at same time
        appointment_time = appointment_time.replace(hour=10, minute=0, second=0, microsecond=0)  # 10 AM
        
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
        
        # Save appointment to database
        print(f"Creating appointment for patient {patient_email} with doctor {selected_doctor['email']}...")  # Debug log
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
def clean_duplicate_appointments(patient_email: str) -> str:
    """Remove duplicate appointments for a patient based on same symptoms and date."""
    try:
        appointments = db.get_appointments()
        patient_appointments = [apt for apt in appointments if apt['patient_email'] == patient_email]
        
        if not patient_appointments:
            return f"No appointments found for patient: {patient_email}"
        
        # Group appointments by symptoms and date
        appointment_groups = {}
        for apt in patient_appointments:
            key = f"{apt['symptoms']}_{apt['appointment_date'][:10]}"  # Group by symptoms and date
            if key not in appointment_groups:
                appointment_groups[key] = []
            appointment_groups[key].append(apt)
        
        duplicates_removed = 0
        for group in appointment_groups.values():
            if len(group) > 1:
                # Keep the first appointment, remove the rest
                for duplicate in group[1:]:
                    db.delete_appointment(duplicate['id'])
                    duplicates_removed += 1
        
        return f"✅ Cleaned up {duplicates_removed} duplicate appointments for {patient_email}"
        
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
    
    if attendee_email:
        event['attendees'] = [{'email': attendee_email}]
    
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
    book_appointment_with_doctor,
    add_medical_record_after_visit,
    cancel_appointment,
    clean_duplicate_appointments
]
llm_with_tools = llm.bind_tools(tools)

class GraphState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

# Node
def tool_calling_llm(state: GraphState):
    sys_msg = SystemMessage(content='''You are a responsible and helpful AI healthcare assistant designed to facilitate safe, efficient communication between patients and doctors.

Your core responsibilities include:

PATIENT IDENTIFICATION:
- Use the find_patient_by_name_or_email tool to identify patients based on their name or email.
- Confirm their identity and provide relevant information like medical history or appointments.


DOCTOR IDENTIFICATION:
- Use the find_doctor_by_name tool to identify doctors and retrieve their email and schedule.
- Provide their upcoming appointments using the get_doctor_appointments tool.

PATIENT DATA COLLECTION:
- Ask about symptoms, duration, severity, and relevant medical history (allergies or chronic conditions), current medications.
- Ensure clarity and completeness before proceeding.
                            
APPOINTMENT COORDINATION:
- Match symptoms to a relevant specialization.
- Book one appointment using Google Calendar.
- Send confirmation emails to both patient and doctor with appointment details.


DOCTOR SUPPORT:
- Retrieve patient medical histories and summarize symptoms when requested.
- Use MedlinePlus to provide medical information for doctors (not patients).

POST-VISIT PROCESSING:
- Add structured data like medications or follow-up plans to the patient record.
- Send patient-friendly summaries via email.

TASK SCHEDULING:
- Schedule reminders for medications, follow-ups, or treatment plans.

GENERAL GUIDELINES:
- Be empathetic and clear with patients, concise and professional with doctors.
- Do not diagnose or suggest medications; act only on doctor input or patient instructions.''')
    new_message = llm_with_tools.invoke([sys_msg] + state["messages"])
    return {"messages": [new_message]}


def should_continue(state: GraphState) -> Literal["tools", "__end__"]:
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:  # Check if the last message has any tool calls
        return "tools"  # Continue to tool execution
    return END  # End the conversation if no tool is needed


# Build graph
tool_node = ToolNode(tools)
builder = StateGraph(GraphState)
builder.add_node("tool_calling_llm", tool_calling_llm)
builder.add_node("tools", tool_node)
builder.add_edge(START, "tool_calling_llm")
builder.add_conditional_edges(
    "tool_calling_llm",
    should_continue,
)
builder.add_edge("tools", "tool_calling_llm")
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

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "user_type" not in st.session_state:
    st.session_state.user_type = None

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
                
                # Run the graph
                response = graph.invoke({"messages": graph_messages})
                
                # Get the last AI message
                last_message = response["messages"][-1]
                response_content = last_message.content
                
                st.markdown(response_content)
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response_content})
                
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

