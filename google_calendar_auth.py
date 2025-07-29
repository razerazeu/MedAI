"""
Google Calendar Authentication and Management for Medical Assistant
Handles authentication, calendar operations, and medical-related event management.
"""
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import os
import pickle

# Comprehensive scopes for medical assistant functionality
SCOPES = [
    'https://www.googleapis.com/auth/calendar',           # Full calendar access
    'https://www.googleapis.com/auth/calendar.events',   # Event management
    'https://www.googleapis.com/auth/calendar.readonly',  # Read calendar data
    'https://www.googleapis.com/auth/calendar.settings.readonly'  # Read calendar settings
]

# Medical-related keywords for filtering (only manage medical events)
MEDICAL_KEYWORDS = [
    'medical', 'doctor', 'appointment', 'medication', 'medicine', 'pill', 'tablet',
    'pharmacy', 'prescription', 'dose', 'dosage', 'treatment', 'therapy', 'clinic',
    'hospital', 'checkup', 'physical', 'exam', 'consultation', 'specialist',
    'cardiology', 'dermatology', 'neurology', 'oncology', 'psychiatry', 'surgery',
    'dental', 'dentist', 'orthodontist', 'optometry', 'ophthalmology', 'radiology',
    'lab', 'blood test', 'x-ray', 'mri', 'ct scan', 'ultrasound', 'biopsy',
    'vaccine', 'vaccination', 'immunization', 'flu shot', 'health', 'wellness',
    'diabetic', 'insulin', 'blood pressure', 'cholesterol', 'antibiotic', 'vitamin',
    'supplement', 'allergy', 'asthma', 'hypertension', 'diabetes', 'chronic'
]

class GoogleCalendarManager:
    """Manages Google Calendar authentication and medical-related operations."""
    
    def __init__(self, credentials_file: str = 'credentials.json', token_file: str = 'token.pickle'):
        self.credentials_file = credentials_file
        self.token_file = token_file
        self.service = None
        self.is_authenticated = False
        
    def authenticate(self) -> bool:
        """
        Authenticate with Google Calendar API.
        Returns True if successful, False otherwise.
        """
        try:
            creds = None
            
            # Load existing token
            if os.path.exists(self.token_file):
                with open(self.token_file, 'rb') as token:
                    creds = pickle.load(token)
            
            # If there are no valid credentials available, request authorization
            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                else:
                    return self._request_new_authorization()
            
            # Save the credentials for the next run
            with open(self.token_file, 'wb') as token:
                pickle.dump(creds, token)
            
            # Build the service
            self.service = build('calendar', 'v3', credentials=creds)
            self.is_authenticated = True
            
            # Test the connection
            calendar_list = self.service.calendarList().list().execute()
            print(f"Successfully connected to Google Calendar! Found {len(calendar_list.get('items', []))} calendars.")
            
            return True
            
        except FileNotFoundError:
            print("credentials.json file not found. Please download it from Google Cloud Console.")
            return False
        except Exception as e:
            print(f"Authentication error: {str(e)}")
            return False
    
    def _request_new_authorization(self) -> bool:
        """Request new authorization from user."""
        try:
            flow = InstalledAppFlow.from_client_secrets_file(self.credentials_file, SCOPES)
            creds = flow.run_local_server(port=8081, prompt='consent')
            
            # Save the credentials for the next run
            with open(self.token_file, 'wb') as token:
                pickle.dump(creds, token)
            
            self.service = build('calendar', 'v3', credentials=creds)
            self.is_authenticated = True
            return True
            
        except Exception as e:
            print(f"Error during new authorization: {e}")
            return False

# Global calendar manager instance
calendar_manager = GoogleCalendarManager()

def get_calendar_manager() -> GoogleCalendarManager:
    """Get the global calendar manager instance."""
    return calendar_manager

def authenticate_google_calendar() -> bool:
    """Simple function to authenticate with Google Calendar."""
    return calendar_manager.authenticate()

def is_calendar_authenticated() -> bool:
    """Check if Google Calendar is authenticated."""
    return calendar_manager.is_authenticated
