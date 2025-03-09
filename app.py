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

# --- Initialize the Head LLM (used for all communications) ---
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

# --- Helper: Convert chat history to text ---
def chat_history_text(history: List[Dict[str, str]]) -> str:
    if not history:
        return "No prior conversation."
    return "\n".join([f"User: {entry['user']}\nAI: {entry['bot']}" for entry in history])

# --- Two-Way Communication Function ---
def two_way_exchange(agent_name: str, raw_message: str, state: LLMState) -> str:
    # Head LLM provides a suggestion based on agent output and conversation context.
    context = chat_history_text(state.memory.get("chat_history", []))
    suggestion_prompt = (
        f"You are the head LLM. The {agent_name} produced the following output:\n"
        f"{raw_message}\n\n"
        f"Here is the conversation history:\n{context}\n\n"
        "Provide a concise suggestion to improve or clarify this output. "
        "Respond only with the suggestion."
    )
    suggestion = "".join(chunk.content for chunk in llm.stream(suggestion_prompt)).strip()
    
    # Agent revises its output based on the head LLM suggestion.
    update_prompt = (
        f"You are the {agent_name}. Your original output was:\n"
        f"{raw_message}\n\n"
        f"The head LLM suggests: {suggestion}\n\n"
        "Please update your output based on this suggestion. "
        "Respond only with the revised output."
    )
    revised_output = "".join(chunk.content for chunk in llm.stream(update_prompt)).strip()
    return revised_output

# --- Specialized Agents with Two-Way Communication ---

def data_loader_agent(state: LLMState):
    if state.memory.get("file_data") is None:
        return {"response": "No file uploaded. Please upload a file to process."}
    df = state.memory["file_data"]
    raw_output = (
        f"File contains {df.shape[0]} rows and {df.shape[1]} columns.\n"
        f"Columns: {', '.join(df.columns)}"
    )
    final_output = two_way_exchange("data_loader_agent", raw_output, state)
    return {"response": final_output}

def data_summarization_agent(state: LLMState):
    if state.memory.get("file_data") is None:
        return {"response": "No file uploaded. Please upload a file for summarization."}
    df = state.memory["file_data"]
    raw_output = "Summary Statistics:\n" + df.describe().to_string()
    final_output = two_way_exchange("data_summarization_agent", raw_output, state)
    return {"response": final_output}

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
    raw_output = "Heatmap generated."
    final_output = two_way_exchange("data_visualization_agent", raw_output, state)
    return {"response": final_output, "image": f"data:image/png;base64,{image_base64}"}

def python_executor_agent(state: LLMState):
    if state.memory.get("file_data") is None:
        return {"response": "No file loaded. Please upload a file first."}
    df = state.memory["file_data"]
    # The prompt instructs the agent to generate Pandas code that answers the user query.
    prompt = (
        f"Generate Pandas code to accomplish the following user request: {state.query}\n"
        f"Available DataFrame columns: {', '.join(df.columns)}\n"
        "Ensure that the code computes the answer (for example, for an age range, compute the min and max) and assign the result to a variable named 'result'.\n"
        "Return only the code."
    )
    raw_code = "".join(chunk.content for chunk in llm.stream(prompt)).strip()
    communication_input = f"Python Executor Agent generated code:\n{raw_code}\n"
    refined_code = two_way_exchange("python_executor_agent", communication_input, state)
    try:
        exec_locals = {"df": df}  # Provide the DataFrame as context.
        exec(refined_code, {}, exec_locals)
        output = exec_locals.get("result", "Code executed successfully.")
    except Exception as e:
        output = f"Error executing code: {str(e)}"
    full_output = f"Generated Code:\n{refined_code}\nExecution Output:\n{output}"
    final_output = two_way_exchange("python_executor_agent", full_output, state)
    return {"response": final_output}

def general_ai_agent(state: LLMState):
    chat_hist = chat_history_text(state.memory.get("chat_history", []))
    prompt = (
        f"You are a conversational AI that takes the conversation history into account.\n\n"
        f"Previous Chat History:\n{chat_hist}\n\n"
        f"User Query: \"{state.query.strip()}\"\n\n"
        "Respond appropriately, integrating prior context. Provide only your final answer."
    )
    raw_output = "".join(chunk.content for chunk in llm.stream(prompt))
    final_output = two_way_exchange("general_ai_agent", raw_output, state)
    return {"response": final_output}

# --- Supervisor: Two-Way Communication for Agent Selection ---
def determine_agent(state: LLMState) -> str:
    chat_hist = chat_history_text(state.memory.get("chat_history", []))
    file_status = "yes" if state.memory.get("file_data") is not None else "no"
    
    prompt = (
        "You are the head LLM selecting the best tool for the current query. Consider the following context:\n\n"
        f"Chat History:\n{chat_hist}\n\n"
        f"User Query: \"{state.query.strip()}\"\n"
        f"File Available: {file_status}\n\n"
        "Available Tools:\n"
        "- data_visualization_agent: For visual data representations (charts, heatmaps).\n"
        "- data_loader_agent: For file analysis (e.g., column names).\n"
        "- data_summarization_agent: For generating statistical insights.\n"
        "- python_executor_agent: For dynamically generating and executing Pandas code (ideal for column analysis, e.g. 'What is the age range of people taking term deposits?').\n"
        "- general_ai_agent: For general conversation.\n\n"
        "Select one tool that best suits the query. Respond only with the tool name."
    )
    decision = "".join(chunk.content for chunk in llm.stream(prompt)).strip().lower()
    valid_agents = {
        "data_visualization_agent",
        "data_loader_agent",
        "data_summarization_agent",
        "python_executor_agent",
        "general_ai_agent"
    }
    chosen_agent = decision if decision in valid_agents else "general_ai_agent"
    # Optionally refine the selection:
    selection_prompt = (
        f"You selected {chosen_agent} based on the query and chat history. "
        "If after further analysis you would revise your selection, what would be your final choice? "
        "Respond only with the final agent name."
    )
    final_decision = "".join(chunk.content for chunk in llm.stream(selection_prompt)).strip().lower()
    return final_decision if final_decision in valid_agents else chosen_agent

def supervisor(state: LLMState):
    state.agent = determine_agent(state)
    return {"agent": state.agent}

# --- Dispatcher: Calls the chosen agent ---
def dispatcher(state: LLMState):
    agent_func = globals()[state.agent]
    return agent_func(state)

# --- Build the Graph ---
graph = StateGraph(LLMState)
graph.add_node("supervisor", supervisor)
graph.add_node("dispatcher", dispatcher)
graph.set_entry_point("supervisor")
graph.add_edge("supervisor", "dispatcher")
executable = graph.compile()

# --- File Loader Function (Loads file once into memory) ---
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

# --- Streamlit UI ---
st.title("Dynamic AI-Powered Data Assistant with Two-Way Communication")

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
    st.write(f"**Chatbot:** {result['response']}")
    if "image" in result:
        st.image(result["image"], caption="Generated Heatmap", use_column_width=True)
    # Save conversation history
    st.session_state.memory["chat_history"].append({"user": user_query, "bot": result["response"]})

# Display Chat History
if st.session_state.memory["chat_history"]:
    st.write("### Chat History:")
    for chat in st.session_state.memory["chat_history"]:
        st.write(f"**You:** {chat['user']}")
        st.write(f"**AI:** {chat['bot']}")