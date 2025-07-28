import streamlit as st
import os
import requests
import json
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

# Sample patient database for testing
PATIENT_DATA = {
    "john_doe": {
        "name": "John Doe",
        "age": 45,
        "gender": "Male",
        "allergies": ["Penicillin", "Shellfish"],
        "chronic_conditions": ["Type 2 Diabetes", "Hypertension"],
        "current_medications": [
            {"name": "Metformin", "dosage": "500mg", "frequency": "twice daily"},
            {"name": "Lisinopril", "dosage": "10mg", "frequency": "once daily"}
        ],
        "medical_history": [
            {"date": "2023-06-15", "condition": "Diagnosed with Type 2 Diabetes"},
            {"date": "2022-03-10", "condition": "Diagnosed with Hypertension"},
            {"date": "2021-11-22", "procedure": "Annual physical exam - normal"}
        ],
        "emergency_contact": {"name": "Jane Doe", "relationship": "Wife", "phone": "555-0123"}
    },
    "sarah_smith": {
        "name": "Sarah Smith", 
        "age": 32,
        "gender": "Female",
        "allergies": ["Latex"],
        "chronic_conditions": ["Asthma"],
        "current_medications": [
            {"name": "Albuterol Inhaler", "dosage": "90mcg", "frequency": "as needed"},
            {"name": "Fluticasone", "dosage": "50mcg", "frequency": "twice daily"}
        ],
        "medical_history": [
            {"date": "2024-01-08", "condition": "Asthma exacerbation - treated"},
            {"date": "2023-07-20", "procedure": "Pulmonary function test - mild obstruction"},
            {"date": "2023-05-15", "condition": "Annual gynecological exam - normal"}
        ],
        "emergency_contact": {"name": "Mike Smith", "relationship": "Husband", "phone": "555-0456"}
    }
}

@tool
def get_patient_records(patient_name: str) -> str:
    """Retrieve patient medical records from the database."""
    # Simple name matching (in real app, would use proper patient ID)
    patient_key = patient_name.lower().replace(" ", "_")
    
    if patient_key in PATIENT_DATA:
        patient = PATIENT_DATA[patient_key]
        
        records = f"""
**Patient: {patient['name']}**
Age: {patient['age']}, Gender: {patient['gender']}

**Allergies:** {', '.join(patient['allergies']) if patient['allergies'] else 'None known'}

**Chronic Conditions:** {', '.join(patient['chronic_conditions']) if patient['chronic_conditions'] else 'None'}

**Current Medications:**
"""
        for med in patient['current_medications']:
            records += f"- {med['name']} {med['dosage']} {med['frequency']}\n"
        
        records += "\n**Medical History:**\n"
        for entry in patient['medical_history']:
            date = entry['date']
            if 'condition' in entry:
                records += f"- {date}: {entry['condition']}\n"
            elif 'procedure' in entry:
                records += f"- {date}: {entry['procedure']}\n"
        
        records += f"\n**Emergency Contact:** {patient['emergency_contact']['name']} ({patient['emergency_contact']['relationship']}) - {patient['emergency_contact']['phone']}"
        
        return records
    else:
        return f"No patient records found for '{patient_name}'. Available patients: John Doe, Sarah Smith"

tools = [MedlinePlus, get_patient_records]
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

When the doctor requests it, retrieve old medical records from the patient's database using the get_patient_records tool. Available test patients: John Doe, Sarah Smith.

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

# Simple console chatbot for testing
def run_console_chat():
    print("Healthcare AI Assistant - Console Chat")
    print("Type 'quit' to exit\n")
    
    messages = []
    
    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
            
        if not user_input:
            continue
            
        # Add user message
        messages.append(HumanMessage(content=user_input))
        
        # Run the graph
        try:
            response = graph.invoke({"messages": messages})
            
            for m in response['messages']:
                m.pretty_print()
            
            # Update messages with AI response
            messages = response["messages"]
            
        except Exception as e:
            print(f"Error: {e}\n")

if __name__ == "__main__":
    run_console_chat()