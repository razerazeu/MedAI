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
        
        records = db.get_patient_medical_history(patient['id'])
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
def book_appointment_with_doctor(patient_email: str, patient_name: str, symptoms: str, specialization: str = "General Medicine") -> str:
    """Book an appointment for a patient with an available doctor based on symptoms and specialization needed."""
    try:
        # Add patient to database if not exists
        patient_id = db.add_patient(patient_email, patient_name)
        
        # Find doctors by specialization
        doctors = db.get_doctors_by_specialization(specialization)
        if not doctors:
            return f"No doctors found for specialization: {specialization}"
        
        # For hackathon - just pick the first available doctor
        selected_doctor = doctors[0]
        
        # Create appointment time (next available slot - simplified for hackathon)
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
        appointment_id = db.create_appointment(
            patient_id, 
            selected_doctor['email'], 
            symptoms, 
            appointment_time,
            google_event_id
        )
        
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
MedAI System
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
        return f"Error booking appointment: {str(e)}"

@tool
def add_medical_record_after_visit(patient_email: str, record_type: str, description: str, doctor_email: str, additional_details: str = None) -> str:
    """Add medical records after a doctor visit. record_type can be: 'medication', 'condition', 'allergy', 'visit'"""
    try:
        patient = db.get_patient_by_email(patient_email)
        if not patient:
            return f"Patient not found: {patient_email}"
        
        details = {}
        if additional_details:
            # Try to parse additional details as key:value pairs
            try:
                for line in additional_details.split('\n'):
                    if ':' in line:
                        key, value = line.split(':', 1)
                        details[key.strip()] = value.strip()
            except:
                details['notes'] = additional_details
        
        record_id = db.add_medical_record(
            patient['id'], 
            record_type, 
            description, 
            details, 
            doctor_email
        )
        
        return f"✅ Medical record added successfully (ID: {record_id}) for {patient['name']}"
        
    except Exception as e:
        return f"Error adding medical record: {str(e)}"

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
                'client_secret_539955399427-4veqqjtgvoq4jn90789l2on5uj14mv69.apps.googleusercontent.com.json', 
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
    """Send email using Google API with OAuth credentials."""
    try:
        # Get Google credentials (same as calendar)
        creds = None
        if os.path.exists('token.json'):
            creds = Credentials.from_authorized_user_file('token.json', SCOPES)
        
        if not creds or not creds.valid:
            print("Google credentials not found or invalid. Please run calendar setup first.")
            return False
        
        # Build Gmail service
        gmail_service = build('gmail', 'v1', credentials=creds)
        
        # Create message
        message = MIMEMultipart()
        message['to'] = to_email
        message['subject'] = subject
        message.attach(MIMEText(body, 'plain'))
        
        # Encode message properly for Gmail API
        import base64
        raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode()
        
        # Send email
        gmail_service.users().messages().send(
            userId='me', 
            body={'raw': raw_message}
        ).execute()
        
        print(f"Email sent successfully to {to_email}")
        return True
        
    except Exception as e:
        print(f"Error sending email via Google API: {e}")
        return False

tools = [MedlinePlus, get_patient_medical_history, book_appointment_with_doctor, add_medical_record_after_visit]
llm_with_tools = llm.bind_tools(tools)

class GraphState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

# Node
def tool_calling_llm(state: GraphState):
    sys_msg = SystemMessage(content='''You are a responsible and helpful Agentic AI healthcare assistant designed to facilitate safe, efficient communication between patients and doctors — without requiring in-person visits.

Your core responsibilities include:
                            
Collecting Patient Data

Ask the user how they are feeling, their current symptoms, duration, severity, and any relevant past medical conditions.

If needed, ask about allergies, medications, or chronic illnesses.

Ensure clarity and completeness before proceeding.

Appointment Coordination

Match the patient's symptoms to a relevant medical specialization.

Book an appointment using Google Calendar with an available doctor.

Send a confirmation email to the doctor, including all symptom details provided by the patient.

Doctor-Side Support

When the doctor requests it, retrieve old medical records from the patient's database using appropriate tools.

Help the doctor by summarizing the patient's current symptoms and history in a concise format.

ONLY when assisting doctors (not patients), use the MedlinePlus tool to search for medical information that can help narrow down potential diagnoses based on the symptoms provided. This tool should NEVER be used for patients directly.
                            
Post-Visit Processing

Accept structured data from the doctor (e.g. medications prescribed, dosage instructions, follow-up plans).

Convert this into a patient-friendly summary and send it via email.

Task & Reminder Scheduling

If the patient requests, schedule medication reminders, follow-up visits, or alerts via email.

Add treatment plans to the patient's calendar when relevant.
                            
You do not diagnose or suggest medications yourself. You act only on doctor input or patient instruction.
Speak clearly, respectfully, and with empathy.''')
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