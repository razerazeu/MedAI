import os
import json
import requests
import uuid
from datetime import datetime
from typing import List, Optional, Dict, Tuple
from langchain_core.tools import tool
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import smtplib
from database import db

def normalize_drug_name(name: str) -> str:
    """Normalize drug names to standard forms for better interaction detection."""
    synonym_map = {
        "paracetamol": "acetaminophen",
        "acetaminophen": "acetaminophen",
        "tylenol": "acetaminophen",
        "panadol": "acetaminophen",
        "ibuprofen": "ibuprofen",
        "advil": "ibuprofen",
        "motrin": "ibuprofen",
        "aspirin": "aspirin",
        "acetylsalicylic acid": "aspirin",
        "naproxen": "naproxen",
        "aleve": "naproxen",
        "lisinopril": "lisinopril",
        "prinivil": "lisinopril",
        "zestril": "lisinopril",
        "losartan": "losartan",
        "cozaar": "losartan",
        "metoprolol": "metoprolol",
        "lopressor": "metoprolol",
        "amoxicillin": "amoxicillin",
        "augmentin": "amoxicillin",
        "azithromycin": "azithromycin",
        "zithromax": "azithromycin",
        "ciprofloxacin": "ciprofloxacin",
        "cipro": "ciprofloxacin",
        "vitamin d": "cholecalciferol",
        "vit d": "cholecalciferol",
        "cholecalciferol": "cholecalciferol",
        "vitamin c": "ascorbic acid",
        "ascorbic acid": "ascorbic acid",
        "diphenhydramine": "diphenhydramine",
        "benadryl": "diphenhydramine",
        "loratadine": "loratadine",
        "claritin": "loratadine",
        "cetirizine": "cetirizine",
        "zyrtec": "cetirizine",
        "omeprazole": "omeprazole",
        "prilosec": "omeprazole",
        "esomeprazole": "esomeprazole",
        "nexium": "esomeprazole",
        "metformin": "metformin",
        "glucophage": "metformin",
        "warfarin": "warfarin",
        "coumadin": "warfarin",
        "atorvastatin": "atorvastatin",
        "lipitor": "atorvastatin",
        "simvastatin": "simvastatin",
        "zocor": "simvastatin",
        "sertraline": "sertraline",
        "zoloft": "sertraline",
        "fluoxetine": "fluoxetine",
        "prozac": "fluoxetine",
        "citalopram": "citalopram",
        "celexa": "citalopram",
        "salbutamol": "albuterol",
        "albuterol": "albuterol",
        "ventolin": "albuterol",
        "fluticasone": "fluticasone",
        "salmeterol": "salmeterol",
        "seretide": "fluticasone+salmeterol",
        "clarithromycin": "clarithromycin",
        "biaxin": "clarithromycin"
    }
    key = name.strip().lower()
    return synonym_map.get(key, key)

def get_fda_drug_label(drug_name: str) -> Optional[str]:
    """Fetch drug label information from OpenFDA API."""
    try:
        normalized = normalize_drug_name(drug_name)
        url = "https://api.fda.gov/drug/label.json"
        params = {"search": f"openfda.generic_name:{normalized}", "limit": 1}
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()

        if "results" not in data or not data["results"]:
            return None
            
        info = data["results"][0]
        interaction_text = " ".join([
            *info.get("drug_interactions", []),
            *info.get("warnings", []),
            *info.get("contraindications", [])
        ])
        return interaction_text.lower() if interaction_text else None
    except Exception as e:
        print(f"⚠️ OpenFDA lookup failed for {drug_name}: {e}")
        return None

def check_drug_interactions(medications: List[str]) -> Tuple[List[str], bool]:
    """
    Check for drug-drug interactions using OpenFDA data.
    Returns (alerts, has_high_risk_interactions)
    """
    alerts = []
    high_risk_found = False
    drug_labels = {}
    
    # Fetch FDA labels for all medications
    for med in medications:
        label = get_fda_drug_label(med)
        drug_labels[med] = label
        print(f"Fetched FDA label for {med} ✅")

    # Check for interactions between drugs
    for i, med1 in enumerate(medications):
        label1 = drug_labels.get(med1, "")
        if not label1:
            continue
            
        for j, med2 in enumerate(medications):
            if i >= j:  # Avoid checking the same pair twice
                continue
                
            label2 = drug_labels.get(med2, "")
            normalized_med2 = normalize_drug_name(med2)
            normalized_med1 = normalize_drug_name(med1)
            
            # Check if med2 is mentioned in med1's warnings/interactions
            if normalized_med2 in label1 or med2.lower() in label1:
                alert = f"{med1} may interact with {med2} (based on FDA label warnings)"
                alerts.append(alert)
                
                # Check for high-risk interactions
                if is_high_risk_interaction(med1, med2, label1):
                    high_risk_found = True
            
            # Check the reverse (med1 in med2's warnings)
            if label2 and (normalized_med1 in label2 or med1.lower() in label2):
                alert = f"{med2} may interact with {med1} (based on FDA label warnings)"
                if alert not in alerts:  # Avoid duplicates
                    alerts.append(alert)
                    
                if is_high_risk_interaction(med2, med1, label2):
                    high_risk_found = True

    return alerts, high_risk_found

def is_high_risk_interaction(drug1: str, drug2: str, warning_text: str) -> bool:
    """Determine if an interaction is high-risk based on warning text and known combinations."""
    warning_text = warning_text.lower()
    drug1_norm = normalize_drug_name(drug1).lower()
    drug2_norm = normalize_drug_name(drug2).lower()
    
    # High-risk indicators in warning text
    high_risk_keywords = [
        "contraindicated", "serious", "severe", "fatal", "death", 
        "hemorrhage", "bleeding", "toxicity", "arrhythmia",
        "respiratory depression", "coma", "seizure"
    ]
    
    # Known high-risk combinations
    high_risk_pairs = [
        ("warfarin", "aspirin"),
        ("warfarin", "ibuprofen"),
        ("warfarin", "naproxen"),
        ("simvastatin", "clarithromycin"),
        ("atorvastatin", "clarithromycin"),
        ("metformin", "contrast"),
        ("digoxin", "amiodarone")
    ]
    
    # Check for high-risk keywords in warning text
    for keyword in high_risk_keywords:
        if keyword in warning_text:
            return True
    
    # Check for known high-risk drug pairs
    for pair in high_risk_pairs:
        if (drug1_norm == pair[0] and drug2_norm == pair[1]) or \
           (drug1_norm == pair[1] and drug2_norm == pair[0]):
            return True
    
    return False

def send_drug_safety_email(to_email: str, subject: str, body: str, is_doctor: bool = False) -> bool:
    """Send drug safety alert email to doctor or patient."""
    try:
        # Email configuration (you'll need to set these environment variables)
        smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
        smtp_port = int(os.getenv("SMTP_PORT", "587"))
        email_user = os.getenv("EMAIL_USER")
        email_password = os.getenv("EMAIL_PASSWORD")
        
        if not all([email_user, email_password]):
            print("⚠️ Email credentials not configured")
            return False
        
        # Create message
        msg = MIMEMultipart()
        msg['From'] = email_user
        msg['To'] = to_email
        msg['Subject'] = subject
        
        # Add body
        msg.attach(MIMEText(body, 'html' if is_doctor else 'plain'))
        
        # Send email
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(email_user, email_password)
            server.send_message(msg)
        
        print(f"✅ Drug safety email sent to {to_email}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to send email to {to_email}: {e}")
        return False

def log_drug_interaction_alert(patient_email: str, patient_name: str, doctor_email: str, 
                             medications: List[str], interactions: List[str], 
                             high_risk: bool) -> str:
    """Log drug interaction alert to the alerts system."""
    try:
        # Create logs directory if it doesn't exist
        os.makedirs("logs", exist_ok=True)
        
        # Create alert record
        alert_id = str(uuid.uuid4())
        alert_record = {
            "id": alert_id,
            "type": "drug_interaction",
            "patient_email": patient_email,
            "patient_name": patient_name,
            "doctor_email": doctor_email,
            "medications": medications,
            "interactions": interactions,
            "high_risk": high_risk,
            "timestamp": datetime.now().isoformat(),
            "status": "sent"
        }
        
        # Load existing alerts
        alerts_file = "logs/drug_safety_alerts.json"
        try:
            if os.path.exists(alerts_file):
                with open(alerts_file, "r", encoding="utf-8") as f:
                    existing_alerts = json.load(f)
            else:
                existing_alerts = []
        except json.JSONDecodeError:
            print("⚠️ Corrupted drug_safety_alerts.json detected. Resetting file.")
            existing_alerts = []
        
        # Check for recent duplicates (same patient, same specific interactions in last 24 hours)
        recent_cutoff = datetime.now().timestamp() - 86400  # 24 hours ago
        
        for existing_alert in existing_alerts:
            try:
                existing_time = datetime.fromisoformat(existing_alert["timestamp"]).timestamp()
                if (existing_time > recent_cutoff and 
                    existing_alert["patient_email"] == patient_email and
                    set(existing_alert["interactions"]) == set(interactions) and
                    existing_alert["high_risk"] == high_risk):
                    print(f"⚠️ Duplicate drug interaction alert for {patient_name} with same interactions within 24 hours. Skipping.")
                    return existing_alert["id"]
            except:
                continue
        
        # Add new alert
        existing_alerts.append(alert_record)
        
        # Write back to file
        with open(alerts_file, "w", encoding="utf-8") as f:
            json.dump(existing_alerts, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Drug interaction alert logged: {alert_id}")
        return alert_id
        
    except Exception as e:
        print(f"❌ Failed to log drug interaction alert: {e}")
        return None

@tool
def check_medication_safety(patient_email: str, medications_list: str) -> str:
    """
    Check for drug-drug interactions when a patient provides their medication list.
    This tool automatically alerts doctors for high-risk interactions and provides patient guidance.
    
    Args:
        patient_email (str): Email of the patient
        medications_list (str): Comma-separated list of medications the patient is taking
    
    Returns:
        str: Safety assessment and guidance for the patient
    """
    try:
        # Parse medications
        medications = [med.strip() for med in medications_list.split(",") if med.strip()]
        if not medications:
            return "No medications provided to check."
        
        # Get patient info
        patient = db.get_patient_by_email(patient_email)
        if not patient:
            return "Patient not found in database."
        
        patient_name = patient.get("name", "Unknown Patient")
        
        # Always check for interactions and show warnings in text
        # (Email alerts will still be deduplicated separately)
        interactions, high_risk = check_drug_interactions(medications)
        
        # Update patient's current medications in database
        medications_str = ", ".join(medications)
        db.update_patient(patient_email, {"current_medication": medications_str})
        
        # Prepare response for patient
        response = f"🏥 **Medication Safety Check for {patient_name}**\n\n"
        response += f"📋 **Your Current Medications:**\n"
        for i, med in enumerate(medications, 1):
            response += f"  {i}. {med}\n"
        response += "\n"
        
        if not interactions:
            response += "✅ **Good News!** No significant drug interactions detected in your current medications.\n\n"
            response += "💡 **Reminder:** Always inform your healthcare providers about all medications, supplements, and vitamins you're taking."
            return response
        
        # Handle interactions found
        response += f"⚠️  **Drug Interaction Alert** ⚠️\n\n"
        response += f"We found {len(interactions)} potential interaction(s) in your medications:\n\n"
        
        for i, interaction in enumerate(interactions, 1):
            response += f"  {i}. {interaction}\n"
        
        if high_risk:
            response += "\n🚨 **HIGH RISK INTERACTION DETECTED** 🚨\n"
            response += "Your doctor has been immediately notified about this high-risk medication combination.\n\n"
            
            # Find patient's doctor or use a default doctor for emergencies
            doctor_email = None
            
            # Try to find the patient's most recent doctor from appointments
            appointments = db.get_appointments()
            patient_appointments = [
                apt for apt in appointments 
                if apt["patient_email"] == patient_email
            ]
            
            if patient_appointments:
                # Get most recent appointment
                patient_appointments.sort(key=lambda x: x["appointment_date"], reverse=True)
                doctor_email = patient_appointments[0]["doctor_email"]
            
            # If no doctor found, use system default or emergency contact
            if not doctor_email:
                # You can set a default doctor email for emergency cases
                doctor_email = os.getenv("EMERGENCY_DOCTOR_EMAIL", "doctor@clinic.com")
            
            # Send alert to doctor
            doctor_subject = f"URGENT: High-Risk Drug Interaction - {patient_name}"
            doctor_body = f"""
            <h2>🚨 HIGH-RISK DRUG INTERACTION ALERT 🚨</h2>
            
            <p><strong>Patient:</strong> {patient_name} ({patient_email})</p>
            <p><strong>Alert Time:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h3>Current Medications:</h3>
            <ul>
            {''.join([f'<li>{med}</li>' for med in medications])}
            </ul>
            
            <h3>Detected Interactions:</h3>
            <ul>
            {''.join([f'<li style="color: red;"><strong>{interaction}</strong></li>' for interaction in interactions])}
            </ul>
            
            <h3>Recommended Actions:</h3>
            <ul>
            <li>Review patient's medication list immediately</li>
            <li>Consider alternative medications if necessary</li>
            <li>Contact patient to discuss medication adjustments</li>
            <li>Document any changes in patient records</li>
            </ul>
            
            <p><em>This is an automated alert from MedAI Drug Safety System.</em></p>
            """
            
            # Send doctor alert
            send_drug_safety_email(doctor_email, doctor_subject, doctor_body, is_doctor=True)
            
            # Log the alert
            log_drug_interaction_alert(
                patient_email, patient_name, doctor_email, 
                medications, interactions, high_risk
            )
            
        else:
            response += "\n⚠️  **Moderate Risk Detected**\n"
            response += "While not immediately dangerous, these interactions should be monitored.\n\n"
        
        # Patient guidance
        response += "**🔒 Important Safety Instructions:**\n"
        response += "• Do NOT stop or change any medications without consulting your doctor\n"
        response += "• Inform all healthcare providers about this interaction alert\n"
        response += "• Watch for unusual symptoms and report them immediately\n"
        response += "• Keep this medication list updated and bring it to all appointments\n\n"
        
        if high_risk:
            response += "• 🚨 **Contact your doctor immediately** - they have been notified\n"
            response += "• If you experience any unusual symptoms, seek medical attention right away\n"
        else:
            response += "• Schedule a follow-up appointment to discuss these medications\n"
        
        return response
        
    except Exception as e:
        print(f"❌ Error in medication safety check: {e}")
        return f"❌ Unable to complete medication safety check. Please try again or contact support. Error: {str(e)}"

@tool
def get_patient_medications_with_safety_check(patient_email: str) -> str:
    """
    Get patient's current medications with automatic safety interaction checking.
    Use this when patients ask about their medications, want to review their list, 
    or need their medication information.
    
    Args:
        patient_email (str): Email of the patient
    
    Returns:
        str: Patient's medication list with safety assessment
    """
    try:
        # Get patient info
        patient = db.get_patient_by_email(patient_email)
        if not patient:
            return "I don't have your information in our system. Please provide your email or contact information."
        
        patient_name = patient.get("name", "Unknown Patient")
        current_medications = patient.get("current_medication")
        
        if not current_medications:
            return f"📋 **Current Medications for {patient_name}**\n\n✅ No medications currently listed in your profile.\n\n💡 **Tip**: If you're taking any medications, vitamins, or supplements, please let me know so I can update your profile and check for any interactions."
        
        # Parse the medication list
        medications = [med.strip() for med in current_medications.split(",") if med.strip()]
        
        response = f"📋 **Current Medications for {patient_name}**\n\n"
        response += f"📝 **Your medications on file:**\n"
        for i, med in enumerate(medications, 1):
            response += f"  {i}. {med}\n"
        response += "\n"
        
        # Check for interactions
        interactions, high_risk = check_drug_interactions(medications)
        
        if not interactions:
            response += "✅ **Safety Status: GOOD**\n"
            response += "No significant drug interactions detected in your current medications.\n\n"
            response += "💡 **Reminders:**\n"
            response += "• Always inform healthcare providers about all your medications\n"
            response += "• Include vitamins, supplements, and over-the-counter drugs\n"
            response += "• Keep this list updated with any changes\n"
            return response
        
        # Handle interactions found
        response += f"⚠️  **Safety Status: INTERACTIONS DETECTED** ⚠️\n\n"
        response += f"I found {len(interactions)} potential interaction(s) in your current medications:\n\n"
        
        for i, interaction in enumerate(interactions, 1):
            response += f"  {i}. {interaction}\n"
        response += "\n"
        
        if high_risk:
            response += "🚨 **HIGH RISK INTERACTIONS PRESENT** 🚨\n\n"
            response += "**⚠️ IMPORTANT**: These medication combinations can be dangerous.\n\n"
            
            # Check if doctor has already been notified recently
            try:
                alerts_file = "logs/drug_safety_alerts.json"
                if os.path.exists(alerts_file):
                    with open(alerts_file, "r", encoding="utf-8") as f:
                        existing_alerts = json.load(f)
                    
                    # Check for recent alerts (within 24 hours) with same specific interactions
                    recent_cutoff = datetime.now().timestamp() - 86400
                    recent_alert = False
                    
                    for alert in existing_alerts:
                        try:
                            alert_time = datetime.fromisoformat(alert["timestamp"]).timestamp()
                            if (alert_time > recent_cutoff and 
                                alert["patient_email"] == patient_email and
                                set(alert["interactions"]) == set(interactions) and
                                alert["high_risk"] == high_risk):
                                recent_alert = True
                                break
                        except:
                            continue
                    
                    if recent_alert:
                        response += "✅ **Your doctor has already been notified** about these interactions.\n\n"
                    else:
                        response += "📧 **Your doctor is being notified immediately** about these high-risk interactions.\n\n"
                        
                        # Find patient's doctor and send alert
                        doctor_email = None
                        appointments = db.get_appointments()
                        patient_appointments = [
                            apt for apt in appointments 
                            if apt["patient_email"] == patient_email
                        ]
                        
                        if patient_appointments:
                            patient_appointments.sort(key=lambda x: x["appointment_date"], reverse=True)
                            doctor_email = patient_appointments[0]["doctor_email"]
                        
                        if not doctor_email:
                            doctor_email = os.getenv("EMERGENCY_DOCTOR_EMAIL", "doctor@clinic.com")
                        
                        # Send alert and log
                        log_drug_interaction_alert(
                            patient_email, patient_name, doctor_email,
                            medications, interactions, high_risk
                        )
                        
                        # Send email alert
                        doctor_subject = f"HIGH-RISK Drug Interaction Review - {patient_name}"
                        doctor_body = f"""
                        <h2>🚨 PATIENT MEDICATION REVIEW ALERT</h2>
                        
                        <p><strong>Patient:</strong> {patient_name} ({patient_email})</p>
                        <p><strong>Request Time:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                        <p><strong>Trigger:</strong> Patient requested medication list review</p>
                        
                        <h3>Current Medications:</h3>
                        <ul>
                        {''.join([f'<li>{med}</li>' for med in medications])}
                        </ul>
                        
                        <h3>High-Risk Interactions Detected:</h3>
                        <ul>
                        {''.join([f'<li style="color: red;"><strong>{interaction}</strong></li>' for interaction in interactions])}
                        </ul>
                        
                        <h3>Recommended Actions:</h3>
                        <ul>
                        <li>Contact patient to schedule medication review appointment</li>
                        <li>Consider medication alternatives or adjustments</li>
                        <li>Provide patient education on interaction risks</li>
                        <li>Document any medication changes</li>
                        </ul>
                        
                        <p><em>This alert was triggered when the patient reviewed their medication list.</em></p>
                        """
                        
                        send_drug_safety_email(doctor_email, doctor_subject, doctor_body, is_doctor=True)
                        
            except Exception as e:
                print(f"Error checking recent alerts: {e}")
                response += "📧 **Your doctor will be notified** about these interactions.\n\n"
            
        else:
            response += "⚠️  **Moderate Risk Interactions**\n\n"
            response += "While not immediately dangerous, these interactions should be monitored.\n\n"
        
        # Safety instructions
        response += "**🔒 Safety Instructions:**\n"
        response += "• **DO NOT** stop or change any medications without consulting your doctor first\n"
        response += "• Show this interaction alert to any healthcare provider you visit\n"
        response += "• Watch for unusual symptoms (bleeding, dizziness, nausea, etc.)\n"
        response += "• Keep a written list of all your medications with you\n"
        response += "• Report any new side effects to your doctor immediately\n\n"
        
        if high_risk:
            response += "🚨 **URGENT**: Contact your doctor as soon as possible to discuss these interactions.\n"
            response += "💊 **Emergency**: If you experience unusual bleeding, severe dizziness, or other concerning symptoms, seek immediate medical care.\n\n"
        else:
            response += "📞 **Next Steps**: Discuss these interactions at your next appointment or contact your doctor if you have concerns.\n\n"
        
        response += "📅 **Medication Management Tip**: Bring this list to all medical appointments and when picking up prescriptions.\n"
        
        return response
        
    except Exception as e:
        print(f"❌ Error in medication review with safety check: {e}")
        return f"❌ I'm having trouble accessing your medication information. Please try again or contact support."

@tool  
def get_drug_interaction_history(patient_email: str) -> str:
    """
    Get the drug interaction alert history for a patient.
    
    Args:
        patient_email (str): Email of the patient
    
    Returns:
        str: History of drug interaction alerts for the patient
    """
    try:
        alerts_file = "logs/drug_safety_alerts.json"
        if not os.path.exists(alerts_file):
            return "No drug interaction alerts on record."
        
        with open(alerts_file, "r", encoding="utf-8") as f:
            all_alerts = json.load(f)
        
        # Filter alerts for this patient
        patient_alerts = [
            alert for alert in all_alerts 
            if alert.get("patient_email") == patient_email
        ]
        
        if not patient_alerts:
            return "No drug interaction alerts found for this patient."
        
        # Sort by timestamp (most recent first)
        patient_alerts.sort(key=lambda x: x["timestamp"], reverse=True)
        
        response = f"📋 **Drug Interaction Alert History**\n\n"
        
        for i, alert in enumerate(patient_alerts, 1):
            alert_date = datetime.fromisoformat(alert["timestamp"]).strftime('%Y-%m-%d %H:%M')
            risk_level = "🚨 HIGH RISK" if alert.get("high_risk") else "⚠️ Moderate Risk"
            
            response += f"**Alert #{i} - {alert_date}**\n"
            response += f"Risk Level: {risk_level}\n"
            response += f"Medications: {', '.join(alert['medications'])}\n"
            response += f"Interactions Found: {len(alert['interactions'])}\n"
            
            for interaction in alert['interactions']:
                response += f"  • {interaction}\n"
            response += "\n"
        
        return response
        
    except Exception as e:
        print(f"❌ Error retrieving drug interaction history: {e}")
        return "❌ Unable to retrieve drug interaction history."

# Initialize drug safety alerts file if it doesn't exist
def initialize_drug_safety_system():
    """Initialize the drug safety alert system."""
    os.makedirs("logs", exist_ok=True)
    alerts_file = "logs/drug_safety_alerts.json"
    if not os.path.exists(alerts_file):
        with open(alerts_file, "w", encoding="utf-8") as f:
            json.dump([], f)
        print("✅ Drug safety alert system initialized")

# Call initialization when module is imported
initialize_drug_safety_system()