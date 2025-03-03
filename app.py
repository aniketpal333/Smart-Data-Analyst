import io
import base64
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph
from pydantic import BaseModel

# Initialize the LLM (used for decision making and general responses)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key="AIzaSyDrmiz2LgB8lWR1T3OmJM9kp9VnrUFIr50",
    streaming=True,
)

# Shared state definition: allow arbitrary types so file objects pass through.
class LLMState(BaseModel):
    query: str
    response: str = ""
    agent: str = "general_ai_agent"
    file: Optional[Any] = None

    model_config = {"arbitrary_types_allowed": True}

# --- Supervisor: Decide which agent to invoke ---
def determine_agent(state: LLMState) -> str:
    q_lower = state.query.lower()
    # Forced rule: if file is uploaded and query contains "heatmap", use visualization.
    if state.file is not None and "heatmap" in q_lower:
        return "data_visualization_agent"
    # Forced rule: if file is uploaded and query contains any summarization keyword, use summarization.
    if state.file is not None and any(word in q_lower for word in ["summarize", "summary", "describe", "stats"]):
        return "data_summarization_agent"
    # If file is uploaded and query indicates reading the file structure, use loader.
    if state.file is not None and any(word in q_lower for word in ["read", "import", "load", "structure"]):
        return "data_loader_agent"
    
    file_status = "yes" if state.file is not None else "no"
    prompt = (
        "You are an intelligent assistant that chooses which tool to invoke based on the user's query and file status. "
        "Available tools are:\n"
        "  - data_visualization_agent: to visualize data (e.g., plot heatmaps or other charts) from an uploaded file.\n"
        "  - data_loader_agent: to simply read and summarize an uploaded file's structure.\n"
        "  - data_summarization_agent: to compute summary statistics and provide a detailed summary of the data.\n"
        "  - general_ai_agent: to handle general conversation and queries.\n\n"
        f"User Query: \"{state.query.strip()}\"\n"
        f"File Uploaded: {file_status}\n\n"
        "Based solely on this information, respond with exactly one option: data_visualization_agent, data_loader_agent, data_summarization_agent, or general_ai_agent."
    )
    decision = "".join(chunk.content for chunk in llm.stream(prompt)).strip().lower()
    if "data_visualization_agent" in decision:
        return "data_visualization_agent"
    elif "data_loader_agent" in decision:
        return "data_loader_agent"
    elif "data_summarization_agent" in decision:
        return "data_summarization_agent"
    else:
        return "general_ai_agent"

def supervisor(state: LLMState):
    state.agent = determine_agent(state)
    return {"agent": state.agent}

# --- Data Loader Agent: Summarizes file structure ---
def data_loader_agent(state: LLMState):
    if state.file is None:
        return {"response": "No file uploaded. Please upload a file to process."}
    file = state.file
    response_text = f"Loaded file: {file.name}\n"
    try:
        file.seek(0)
        if file.name.endswith(".csv"):
            decoded = io.StringIO(file.getvalue().decode("utf-8"))
            df = pd.read_csv(decoded, sep=None, engine='python')
        elif file.name.endswith((".xls", ".xlsx")):
            file.seek(0)
            df = pd.read_excel(file)
        else:
            return {"response": "Unsupported file type. Please upload a CSV or Excel file."}
        response_text += f"\nFile contains {df.shape[0]} rows and {df.shape[1]} columns."
        response_text += f"\nColumns: {', '.join(df.columns)}"
    except Exception as e:
        return {"response": f"Error loading file: {str(e)}"}
    return {"response": response_text}

# --- Data Summarization Agent: Computes summary statistics ---
def data_summarization_agent(state: LLMState):
    if state.file is None:
        return {"response": "No file uploaded. Please upload a file for summarization."}
    file = state.file
    response_text = f"Loaded file: {file.name}\n"
    try:
        file.seek(0)
        if file.name.endswith(".csv"):
            decoded = io.StringIO(file.getvalue().decode("utf-8"))
            df = pd.read_csv(decoded, sep=None, engine='python')
        elif file.name.endswith((".xls", ".xlsx")):
            file.seek(0)
            df = pd.read_excel(file)
        else:
            return {"response": "Unsupported file type. Please upload a CSV or Excel file."}
        response_text += f"\nFile contains {df.shape[0]} rows and {df.shape[1]} columns."
        response_text += f"\nColumns: {', '.join(df.columns)}"
        
        # Cleaning: Remove extra quotes and attempt to convert columns.
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].str.replace('"', '', regex=False)
            df[col] = pd.to_numeric(df[col], errors='ignore')
        
        # Summarize numeric columns.
        numeric_columns = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
        summary_text = ""
        if numeric_columns:
            summary_text += "\nNumeric Summary:\n" + df[numeric_columns].describe().to_string()
        # Summarize non-numeric columns (top 5 value counts).
        non_numeric_columns = [col for col in df.columns if not pd.api.types.is_numeric_dtype(df[col])]
        if non_numeric_columns:
            for col in non_numeric_columns:
                counts = df[col].value_counts().head(5).to_string()
                summary_text += f"\n\nTop 5 counts for {col}:\n{counts}"
        if not summary_text:
            summary_text = "\nNo summary available."
        response_text += summary_text
        return {"response": response_text}
    except Exception as e:
        return {"response": f"Error summarizing file: {str(e)}"}

# --- Data Visualization Agent: Plots a heatmap ---
def data_visualization_agent(state: LLMState):
    if state.file is None:
        return {"response": "No file uploaded. Please upload a file for visualization."}
    file = state.file
    response_text = f"Loaded file: {file.name}\n"
    try:
        file.seek(0)
        if file.name.endswith(".csv"):
            decoded = io.StringIO(file.getvalue().decode("utf-8"))
            df = pd.read_csv(decoded, sep=None, engine='python')
        elif file.name.endswith((".xls", ".xlsx")):
            file.seek(0)
            df = pd.read_excel(file)
        else:
            return {"response": "Unsupported file type. Please upload a CSV or Excel file."}
        response_text += f"\nFile contains {df.shape[0]} rows and {df.shape[1]} columns."
        response_text += f"\nColumns: {', '.join(df.columns)}"
        
        # Cleaning: Remove extra quotes and convert to numeric.
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].str.replace('"', '', regex=False)
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        numeric_columns = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
        if not numeric_columns:
            response_text += "\nNo numeric columns available for visualization after conversion."
            return {"response": response_text}
        
        corr = df[numeric_columns].corr()
        if corr.empty:
            response_text += "\nCorrelation matrix is empty. Check numeric conversion."
            return {"response": response_text}
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        ax.set_title("Correlation Heatmap")
        
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        image_data = f"data:image/png;base64,{image_base64}"
        
        return {"response": response_text, "image": image_data}
    except Exception as e:
        return {"response": f"Error visualizing file: {str(e)}"}

# --- General AI Agent: Handles other queries ---
def general_ai_agent(state: LLMState):
    system_prompt = "You are a helpful assistant. Provide a concise answer to the user's query."
    full_prompt = f"System: {system_prompt}\nUser Query: {state.query.strip()}"
    response_text = "".join(chunk.content for chunk in llm.stream(full_prompt))
    return {"response": response_text}

# --- Dispatcher: Route the state to the chosen agent ---
def dispatcher(state: LLMState):
    if state.agent == "data_loader_agent":
        return data_loader_agent(state)
    elif state.agent == "data_visualization_agent":
        return data_visualization_agent(state)
    elif state.agent == "data_summarization_agent":
        return data_summarization_agent(state)
    else:
        return general_ai_agent(state)

# --- Build the Graph ---
graph = StateGraph(LLMState)
graph.add_node("supervisor", supervisor)
graph.add_node("dispatcher", dispatcher)
graph.add_node("data_loader_agent", data_loader_agent)
graph.add_node("data_visualization_agent", data_visualization_agent)
graph.add_node("data_summarization_agent", data_summarization_agent)
graph.add_node("general_ai_agent", general_ai_agent)

graph.set_entry_point("supervisor")
graph.add_edge("supervisor", "dispatcher")

executable = graph.compile()

# --- Streamlit UI ---
st.title("AI-Powered Data Assistant")

uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xls", "xlsx"])
user_query = st.text_input("Enter your query:")

if st.button("Submit"):
    state = LLMState(query=user_query, file=uploaded_file)
    result = executable.invoke(state)
    st.write(f"**Chatbot:** {result['response']}")
    if "image" in result:
        st.image(result["image"])