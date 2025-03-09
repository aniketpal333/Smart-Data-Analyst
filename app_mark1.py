import io
import base64
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Any, Dict, List
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph
from pydantic import BaseModel
from PIL import Image

# --- Initialize the LLM ---
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key="AIzaSyDrmiz2LgB8lWR1T3OmJM9kp9VnrUFIr50",
    streaming=True,
)

# --- Memory Log for Conversation & File Data ---
if "memory" not in st.session_state:
    st.session_state.memory = {"chat_history": [], "file_data": None, "file_name": None}

# --- Shared State Definition ---
class LLMState(BaseModel):
    query: str
    response: str = ""
    agent: str = "general_ai_agent"
    file: Optional[Any] = None
    memory: Dict[str, Any] = {}

    model_config = {"arbitrary_types_allowed": True}

# --- LLM-Based Decision Making with Chat History Awareness ---
def determine_agent(state: LLMState) -> str:
    chat_history = "\n".join([f"User: {chat['user']}\nAI: {chat['bot']}" for chat in state.memory.get("chat_history", [])])
    file_status = "yes" if state.memory.get("file_data") is not None else "no"

    prompt = f"""
    You are an AI assistant selecting the best tool for the user's query. Consider chat history for context.
    
    **Available Tools:**
    - `data_visualization_agent`: For charts, graphs, and data visualizations.
    - `data_loader_agent`: For file analysis (e.g., retrieving column names).
    - `data_summarization_agent`: For data insights and statistics.
    - `general_ai_agent`: For all other queries.

    **Previous Chat History:**  
    {chat_history if chat_history else "No previous history available."}

    **User Query:** "{state.query.strip()}"  
    **File Available in Memory:** {file_status}  

    **Examples for Reference:**
    - "Show a heatmap of correlations." → `data_visualization_agent`
    - "Summarize this dataset." → `data_summarization_agent`
    - "What are the columns in this file?" → `data_loader_agent`
    - "Tell me a joke!" → `general_ai_agent`
    - "Plot the trends in sales data." → `data_visualization_agent`
    - "Find the average and median of the dataset." → `data_summarization_agent`
    - "What type of data is in this file?" → `data_loader_agent`
    
    Select **one tool** based on the context. **Do not explain. Respond only with the tool name.**
    """

    decision = "".join(chunk.content for chunk in llm.stream(prompt)).strip().lower()

    valid_agents = {
        "data_visualization_agent",
        "data_loader_agent",
        "data_summarization_agent",
        "general_ai_agent"
    }

    return decision if decision in valid_agents else "general_ai_agent"

# --- Supervisor Function ---
def supervisor(state: LLMState):
    state.agent = determine_agent(state)
    return {"agent": state.agent}

# --- File Loader (Loads Once, Then Stores in Memory) ---
def load_file(file) -> Optional[pd.DataFrame]:
    if file is None:
        return None

    try:
        file.seek(0)
        if file.name.endswith(".csv"):
            return pd.read_csv(io.StringIO(file.getvalue().decode("utf-8")), sep=None, engine='python')
        elif file.name.endswith((".xls", ".xlsx")):
            return pd.read_excel(file)
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

# --- Data Loader Agent ---
def data_loader_agent(state: LLMState):
    if state.memory.get("file_data") is None:
        return {"response": "No file uploaded. Please upload a file to process."}
    
    df = state.memory["file_data"]
    response_text = f"File contains {df.shape[0]} rows and {df.shape[1]} columns.\n"
    response_text += f"Columns: {', '.join(df.columns)}"
    
    return {"response": response_text}

# --- Data Summarization Agent ---
def data_summarization_agent(state: LLMState):
    if state.memory.get("file_data") is None:
        return {"response": "No file uploaded. Please upload a file for summarization."}
    
    df = state.memory["file_data"]
    response_text = "Summary Statistics:\n" + df.describe().to_string()
    
    return {"response": response_text}

# --- Data Visualization Agent ---
def data_visualization_agent(state: LLMState):
    if state.memory.get("file_data") is None:
        return {"response": "No file uploaded. Please upload a file for visualization."}
    
    df = state.memory["file_data"]
    numeric_columns = df.select_dtypes(include=['number']).columns

    if numeric_columns.empty:
        return {"response": "No numeric columns available for visualization."}

    corr = df[numeric_columns].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Correlation Heatmap")

    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')

    return {"response": "Heatmap generated.", "image": f"data:image/png;base64,{image_base64}"}

# --- General AI Agent with Chat Memory ---
def general_ai_agent(state: LLMState):
    chat_history = "\n".join([f"User: {chat['user']}\nAI: {chat['bot']}" for chat in state.memory.get("chat_history", [])])

    prompt = f"""
    You are an AI chatbot that understands and remembers past conversations.
    
    **Previous Chat History:**  
    {chat_history if chat_history else "No previous history available."}

    **User Query:** "{state.query.strip()}"
    
    Respond accordingly, keeping in mind the previous interactions.
    """

    response_text = "".join(chunk.content for chunk in llm.stream(prompt))
    return {"response": response_text}

# --- Dispatcher ---
def dispatcher(state: LLMState):
    return globals()[state.agent](state)

# --- Build the Graph ---
graph = StateGraph(LLMState)
graph.add_node("supervisor", supervisor)
graph.add_node("dispatcher", dispatcher)
graph.set_entry_point("supervisor")
graph.add_edge("supervisor", "dispatcher")
executable = graph.compile()

# --- Streamlit UI ---
st.title("AI-Powered Data Assistant with Chat Memory")

uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xls", "xlsx"])
user_query = st.text_input("Enter your query:")

# Store file in memory only once
if uploaded_file is not None and uploaded_file.name != st.session_state.memory["file_name"]:
    st.session_state.memory["file_data"] = load_file(uploaded_file)
    st.session_state.memory["file_name"] = uploaded_file.name

if st.button("Submit"):
    state = LLMState(
        query=user_query,
        file=uploaded_file,
        memory=st.session_state.memory
    )

    result = executable.invoke(state)

    # Save conversation history
    st.session_state.memory["chat_history"].append({"user": user_query, "bot": result["response"]})

    st.write(f"**Chatbot:** {result['response']}")
    if "image" in result:
        st.image(result["image"], caption="Generated Heatmap", use_column_width=True)

# --- Display Chat History ---
if st.session_state.memory["chat_history"]:
    st.write("### Chat History:")
    for chat in st.session_state.memory["chat_history"]:
        st.write(f"**You:** {chat['user']}")
        st.write(f"**AI:** {chat['bot']}")