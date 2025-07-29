"""
Gmail API Integration for Medical Assistant
Handles email notifications for medication reminders and appointment alerts.
"""
from datetime import datetime, timedelta
import pickle
import os  # Ensure os is imported for file path checks
from typing import Optional, Dict, Any, List
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# Gmail API scopes
GMAIL_SCOPES = [
    'https://www.googleapis.com/auth/gmail.send',        # Send emails
    'https://www.googleapis.com/auth/gmail.compose',     # Compose emails
    'https://www.googleapis.com/auth/gmail.modify'       # Modify emails (for drafts)
]

class GmailManager:
    """Manages Gmail API authentication and email operations for medical reminders."""
    
    def __init__(self, credentials_file: str = 'credentials.json', token_file: str = 'gmail_token.pickle'):
        self.credentials_file = credentials_file
        self.token_file = token_file
        self.service = None
        self.is_authenticated = False
        self.user_email = None
        
    def authenticate(self) -> bool:
        """
        Authenticate with Gmail API.
        Returns True if successful, False otherwise.
        """
        try:
            print("Starting Gmail authentication...")  # Debug log
            creds = None

            # Load existing token
            if os.path.exists(self.token_file):
                print(f"Loading token from {self.token_file}...")  # Debug log
                with open(self.token_file, 'rb') as token:
                    creds = pickle.load(token)

            # If there are no valid credentials available, request authorization
            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    print("Refreshing expired credentials...")  # Debug log
                    creds.refresh(Request())
                else:
                    print("Requesting new authorization...")  # Debug log
                    return self._request_new_authorization()

            # Save the credentials for the next run
            with open(self.token_file, 'wb') as token:
                pickle.dump(creds, token)

            # Build the service
            print("Building Gmail service...")  # Debug log
            self.service = build('gmail', 'v1', credentials=creds)
            self.is_authenticated = True

            # Get user email address
            profile = self.service.users().getProfile(userId='me').execute()
            self.user_email = profile.get('emailAddress', 'Unknown')
            print(f"Authenticated as {self.user_email}")  # Debug log
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
            print("Starting new authorization flow...")  # Debug log
            flow = InstalledAppFlow.from_client_secrets_file(self.credentials_file, GMAIL_SCOPES)
            creds = flow.run_local_server(port=8081, prompt='consent')

            # Save the credentials for the next run
            with open(self.token_file, 'wb') as token:
                pickle.dump(creds, token)

            self.service = build('gmail', 'v1', credentials=creds)
            self.is_authenticated = True
            print("New authorization successful.")  # Debug log
            return True

        except Exception as e:
            print(f"Error during new authorization: {e}")
            return False

# Global Gmail manager instance
gmail_manager = GmailManager()

def get_gmail_manager() -> GmailManager:
    """Get the global Gmail manager instance."""
    return gmail_manager

def authenticate_gmail() -> bool:
    """Simple function to authenticate with Gmail."""
    return gmail_manager.authenticate()

def is_gmail_authenticated() -> bool:
    """Check if Gmail is authenticated."""
    return gmail_manager.is_authenticated()

def get_user_email() -> str:
    """Get the authenticated user's email address."""
    return gmail_manager.user_email or "Unknown"
