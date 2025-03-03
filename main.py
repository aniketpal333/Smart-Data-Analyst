import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import openai
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# --- Initialization ---
st.set_page_config(page_title="Agentic Chatbot Smart Data Analyst", layout="wide")
st.title("Agentic Chatbot Smart Data Analyst")

# Initialize session state for conversation and data
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [{"role": "system", "content": (
        "You are an intelligent data analyst agent. You can load data, generate summaries, create visualizations, "
        "train machine learning models, and learn from feedback. Engage in a friendly, conversational style while "
        "providing technical insights and suggestions. If a command is detected (like 'data summary', 'visualize', or 'train model'), "
        "perform that action and report the results."
    )}]
if "dataframe" not in st.session_state:
    st.session_state.dataframe = None

# --- OpenAI API Key Input ---
openai_api_key = st.sidebar.text_input("Enter your OpenAI API key", type="password")
if openai_api_key:
    openai.api_key = openai_api_key
else:
    st.sidebar.warning("An OpenAI API key is required for language processing and feedback functions.")

# --- File Uploader ---
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is not None:
    try:
        st.session_state.dataframe = pd.read_csv(uploaded_file)
        st.sidebar.success("Dataset loaded successfully!")
    except Exception as e:
        st.sidebar.error(f"Error loading file: {e}")

# --- Define Agentic Actions ---

def generate_data_summary(df):
    summary = f"Data Preview:\n{df.head().to_string()}\n\n" \
              f"Data Description:\n{df.describe().to_string()}\n\n" \
              f"Missing Values:\n{df.isnull().sum().to_string()}"
    return summary

def plot_correlation_heatmap(df):
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
    plt.tight_layout()
    return fig

def train_classification_model(df, target_column):
    if target_column not in df.columns:
        return "Target column not found in the dataset."
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    report = classification_report(y_test, predictions)
    return report

def call_openai_chat(messages, max_tokens=200, temperature=0.7):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        return f"Error calling OpenAI API: {e}"

def process_command(user_input):
    """
    Check if the input has specific commands and process them.
    Otherwise, defer to the LLM for a general response.
    """
    df = st.session_state.dataframe
    lower_input = user_input.lower()
    response = ""
    
    if "load data" in lower_input:
        if df is not None:
            response = "Data is already loaded. Use 'data summary' to see an overview."
        else:
            response = "Please upload a CSV file using the sidebar."
    
    elif "data summary" in lower_input or "summarize" in lower_input:
        if df is None:
            response = "No dataset available. Please upload a CSV file first."
        else:
            summary = generate_data_summary(df)
            response = f"Here is the data summary:\n\n{summary}"
    
    elif "visualize" in lower_input or "heatmap" in lower_input:
        if df is None:
            response = "No dataset available. Please upload a CSV file first."
        else:
            fig = plot_correlation_heatmap(df)
            st.image(fig, caption="Correlation Heatmap", use_column_width=True)
            response = "Displayed the correlation heatmap."
    
    elif "train model" in lower_input:
        if df is None:
            response = "No dataset available. Please upload a CSV file first."
        else:
            # Determine target column (default to 'target' if available)
            target_column = "target" if "target" in df.columns else None
            if not target_column:
                # Ask user to specify target column if not found
                response = "No default 'target' column found. Please specify the target column in your command, e.g., 'train model on column X'."
            else:
                report = train_classification_model(df, target_column)
                response = f"Model trained on column '{target_column}'.\n\nClassification Report:\n{report}"
    
    elif "feedback" in lower_input:
        # This branch can be triggered if the user gives direct feedback
        response = "Thank you for your feedback. Please type your detailed feedback message."
    
    return response

# --- Chat Interface ---
st.markdown("### Chat with the Agent")
user_input = st.text_input("Your message", key="user_input")

if st.button("Send") and user_input:
    # Append user message to chat history
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    
    # First check for command keywords in the user input
    command_response = process_command(user_input)
    
    if command_response:
        # If a command was processed, add its response to history
        st.session_state.chat_history.append({"role": "assistant", "content": command_response})
    else:
        # Otherwise, call the OpenAI LLM with the conversation history
        assistant_reply = call_openai_chat(st.session_state.chat_history)
        st.session_state.chat_history.append({"role": "assistant", "content": assistant_reply})
    
    # Clear the input box after sending
    st.experimental_rerun()

# Display chat history
st.markdown("#### Conversation History")
for msg in st.session_state.chat_history[1:]:
    role = "🧠 Analyst" if msg["role"] == "assistant" else "👤 You"
    st.markdown(f"**{role}:** {msg['content']}")

# --- Feedback Section ---
st.markdown("### Provide Feedback to Help the Agent Learn")
feedback_text = st.text_area("Enter your feedback on the analysis or suggestions for improvements", key="feedback")
if st.button("Submit Feedback"):
    if feedback_text.strip() == "":
        st.warning("Please provide some feedback before submitting.")
    else:
        # Save feedback locally (simulate learning by saving to file)
        with open("feedback_history.txt", "a") as f:
            f.write(feedback_text + "\n")
        st.success("Feedback submitted! Thank you.")
        
        # Let the agent generate suggestions based on the feedback
        feedback_prompt = f"The user provided the following feedback for improving the data analysis: \"{feedback_text}\". " \
                          "Please suggest improvements or further steps to enhance the analysis process."
        feedback_response = call_openai_chat([{"role": "user", "content": feedback_prompt}], max_tokens=150)
        st.markdown("#### Suggestions Based on Your Feedback")
        st.write(feedback_response)