import io
import base64
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
from typing import Optional, Any, Dict, List
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph
from pydantic import BaseModel
import warnings
import nbformat
from nbconvert import PythonExporter
import subprocess
import sys
import contextlib

# Suppress matplotlib non-interactive backend warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# ----------------------------------------------------------------------
# Initialize Head LLM (Supervisor) – used only for understanding human language
# and routing the query to the appropriate static agent.
# ----------------------------------------------------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key="AIzaSyDrmiz2LgB8lWR1T3OmJM9kp9VnrUFIr50",
    streaming=True,
)

# ----------------------------------------------------------------------
# Memory Log: Store conversation history and uploaded file data.
# ----------------------------------------------------------------------
if "memory" not in st.session_state:
    st.session_state.memory = {"chat_history": [], "file_data": None, "file_name": None}

# ----------------------------------------------------------------------
# Shared State Definition – holds the current query, file, and memory.
# ----------------------------------------------------------------------
class LLMState(BaseModel):
    query: str
    response: str = ""
    agent: str = "general_ai_agent"  # default fallback agent
    file: Optional[Any] = None
    memory: Dict[str, Any] = {}
    model_config = {"arbitrary_types_allowed": True}

# ----------------------------------------------------------------------
# Helper: Convert chat history to a single text block.
# ----------------------------------------------------------------------
def chat_history_text(history: List[Dict[str, str]]) -> str:
    if not history:
        return "No prior conversation."
    return "\n".join([f"User: {entry['user']}\nAI: {entry['bot']}" for entry in history])

# ----------------------------------------------------------------------
# Helper: Explain results in plain language.
# Uses the head LLM to translate technical output into human-friendly language.
# ----------------------------------------------------------------------
def explain_in_plain_language(raw_text: str, agent_name: str, context: str = "") -> str:
    explanation_prompt = (
        f"You are an expert data analyst. Analyze the following output from the {agent_name} "
        "and explain it in plain, human-friendly language for a non-technical audience. "
        "Focus on describing trends, patterns, and general insights without too many raw numbers. "
        "Here are the results:\n\n"
        f"{raw_text}\n\n"
        f"Context:\n{context}\n\n"
        "Provide only the explanation."
    )
    explanation = "".join(chunk.content for chunk in llm.stream(explanation_prompt)).strip()
    return explanation

# ----------------------------------------------------------------------
# Agent: Data Loader – loads file and extracts basic structure.
# Static code returns file name, shape, and column data.
# ----------------------------------------------------------------------
def data_loader_agent(state: LLMState):
    if state.memory.get("file_data") is None:
        return {"response": "No file uploaded. Please upload a CSV or Excel file."}
    df = state.memory["file_data"]
    raw_output = (
        f"File '{st.session_state.memory['file_name']}' loaded successfully.\n"
        f"It has {df.shape[0]} rows and {df.shape[1]} columns.\n"
        f"Column details:\n{df.dtypes.to_string()}"
    )
    plain_output = explain_in_plain_language(raw_output, "data_loader_agent", chat_history_text(state.memory.get("chat_history", [])))
    return {"response": plain_output}

# ----------------------------------------------------------------------
# Agent: Data Summarizer – computes descriptive statistics and (if possible) trend analysis.
# Static code: runs df.describe() and, if a datetime column exists, performs a 5-point rolling average trend.
# ----------------------------------------------------------------------
# def data_summarization_agent(state: LLMState):
#     if state.memory.get("file_data") is None:
#         return {"response": "No file uploaded. Please upload a file for summarization."}
#     df = state.memory["file_data"]
#     stats = df.describe().to_string()
#     trend_text = ""
#     figure = None

#     # Check for a datetime column for trend analysis.
#     time_columns = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
#     if time_columns:
#         time_col = time_columns[0]
#         df_sorted = df.sort_values(by=time_col)
#         numeric_cols = df_sorted.select_dtypes(include=["number"]).columns.tolist()
#         if numeric_cols:
#             trend_col = numeric_cols[0]
#             rolling_avg = df_sorted[trend_col].rolling(window=5).mean()
#             trend_text = (
#                 f"\nA trend analysis on '{trend_col}' (using a 5-point rolling average) indicates how values change over time."
#             )
#             # Create a simple line plot.
#             fig, ax = plt.subplots()
#             ax.plot(df_sorted[time_col], df_sorted[trend_col], label="Original Data")
#             ax.plot(df_sorted[time_col], rolling_avg, label="Rolling Average", linestyle='--')
#             ax.set_xlabel(time_col)
#             ax.set_ylabel(trend_col)
#             ax.set_title(f"Trend Analysis of {trend_col}")
#             ax.legend()
#             figure = fig

#     raw_output = f"Descriptive statistics:\n{stats}{trend_text}"
#     plain_output = explain_in_plain_language(raw_output, "data_summarization_agent", chat_history_text(state.memory.get("chat_history", [])))
#     return {"response": plain_output, "figure": figure}

# ----------------------------------------------------------------------
# Agent: Python Executor – executes a static piece of code based on known query keywords.
# Static code paths:
#   - If query contains 'describe': returns df.describe()
#   - If query contains 'visualize': returns a basic scatter plot using first two numeric columns.
#   - If query contains 'success rate': computes success/failure based on a column containing 'status'
# ----------------------------------------------------------------------
def execute_notebook(notebook_path: str):
    try:
        # Convert notebook to Python script
        with open(notebook_path, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)

        exporter = PythonExporter()
        script, _ = exporter.from_notebook_node(nb)

        # Execute the script
        exec_globals = {}
        exec(script, exec_globals)

        # Collect text-based outputs
        result_text = exec_globals.get("result", "Notebook executed, but no output captured.")

        # Collect Matplotlib figures if generated
        figure_list = []
        for fig_num in plt.get_fignums():
            fig = plt.figure(fig_num)
            figure_list.append(fig)

        return result_text, figure_list

    except Exception as e:
        return f"Error executing notebook: {str(e)}", None

def python_executor_agent(state: LLMState):
    if state.memory.get("file_data") is None:
        return {"response": "No file loaded. Please upload a file first."}

    df = state.memory["file_data"]
    query_lower = state.query.lower()

    # If user asks for "analyze the data", execute the uploaded notebook
    if "analyze the data" in query_lower:
        notebook_path = "Marketing.ipynb"
        result_text, figure_list = execute_notebook(notebook_path)

        plain_output = explain_in_plain_language(result_text, "python_executor_agent", chat_history_text(state.memory.get("chat_history", [])))
        return {"response": plain_output, "figure": figure_list if figure_list else None}

    # Otherwise, proceed with normal Pandas code execution
    # (Existing logic for Pandas query execution remains unchanged)
#def python_executor_agent(state: LLMState):
    else:
        if state.memory.get("file_data") is None:
            return {"response": "No file loaded. Please upload a file first."}
        df = state.memory["file_data"]
        query_lower = state.query.lower()
        raw_response = ""
        figure_list = None

        if "describe" in query_lower:
            raw_response = df.describe().to_string()
        elif "visualize" in query_lower:
            # Create a scatter plot from the first two numeric columns.
            numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
            if len(numeric_cols) >= 2:
                fig, ax = plt.subplots()
                ax.scatter(df[numeric_cols[0]], df[numeric_cols[1]])
                ax.set_xlabel(numeric_cols[0])
                ax.set_ylabel(numeric_cols[1])
                ax.set_title("Scatter Plot of {} vs. {}".format(numeric_cols[0], numeric_cols[1]))
                raw_response = f"A scatter plot was created using '{numeric_cols[0]}' and '{numeric_cols[1]}'."
                figure_list = [fig]
            else:
                raw_response = "Not enough numerical columns to generate a visualization."
        elif "success rate" in query_lower or "failure rate" in query_lower:
            status_column = next((col for col in df.columns if "status" in col.lower()), None)
            if status_column:
                rates = df[status_column].value_counts(normalize=True) * 100
                raw_response = f"Success/Failure Rate based on '{status_column}':\n{rates.to_string()}"
                fig, ax = plt.subplots()
                rates.plot(kind='bar', ax=ax)
                ax.set_ylabel("Percentage")
                ax.set_title(f"Success/Failure Rate for {status_column}")
                figure_list = [fig]
            else:
                raw_response = "No status column found to compute success/failure rate."
        else:
            raw_response = "Static execution: No predefined action for the given query. Try 'describe', 'visualize', or 'success rate'."
    
        plain_output = explain_in_plain_language(raw_response, "python_executor_agent", chat_history_text(state.memory.get("chat_history", [])))
        return {"response": plain_output, "figure": figure_list}

# ----------------------------------------------------------------------
# Agent: General AI – handles general conversation.
# Static code simply echoes a response.
# ----------------------------------------------------------------------
def general_ai_agent(state: LLMState):
    raw_output = f"You said: {state.query.strip()}. This is a static general response."
    plain_output = explain_in_plain_language(raw_output, "general_ai_agent", chat_history_text(state.memory.get("chat_history", [])))
    return {"response": plain_output}

# ----------------------------------------------------------------------
# Supervisor: Uses the head LLM only to interpret human language and select the correct static agent.
# ----------------------------------------------------------------------
def determine_agent(state: LLMState) -> str:
    chat_hist = chat_history_text(state.memory.get("chat_history", []))
    file_status = "yes" if state.memory.get("file_data") is not None else "no"
    prompt = (
        "You are an expert in natural language understanding. Based on the following context, determine which of the following static agents "
        "should handle the user's query. The available agents are:\n"
        "- data_loader_agent: For file inspection.\n"
        #"- data_summarization_agent: For descriptive statistics and trend analysis.\n"
        "- python_executor_agent: For executing static code such as analyzing data, describing data, visualizing, or computing success rates.\n"
        "- general_ai_agent: For general conversation.\n\n"
        f"Chat History:\n{chat_hist}\n\n"
        f"User Query: \"{state.query.strip()}\"\n"
        f"File Available: {file_status}\n\n"
        "Respond only with the name of the chosen agent."
    )
    decision = "".join(chunk.content for chunk in llm.stream(prompt)).strip().lower()
    valid_agents = {"data_loader_agent", "data_summarization_agent", "python_executor_agent", "general_ai_agent"}
    return decision if decision in valid_agents else "general_ai_agent"

def supervisor(state: LLMState):
    state.agent = determine_agent(state)
    return {"agent": state.agent}

# ----------------------------------------------------------------------
# Dispatcher: Routes the query to the selected static agent.
# ----------------------------------------------------------------------
def dispatcher(state: LLMState):
    agent_func = globals()[state.agent]
    return agent_func(state)

# ----------------------------------------------------------------------
# Build the Graph using StateGraph from langgraph.
# ----------------------------------------------------------------------
graph = StateGraph(LLMState)
graph.add_node("supervisor", supervisor)
graph.add_node("dispatcher", dispatcher)
graph.add_node("data_loader_agent", data_loader_agent)
#graph.add_node("data_summarization_agent", data_summarization_agent)
graph.add_node("python_executor_agent", python_executor_agent)
graph.add_node("general_ai_agent", general_ai_agent)

graph.set_entry_point("supervisor")
graph.add_edge("supervisor", "dispatcher")
executable = graph.compile()

# ----------------------------------------------------------------------
# File Loader: Loads the CSV or Excel file into memory.
# ----------------------------------------------------------------------
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

# ----------------------------------------------------------------------
# Streamlit UI
# ----------------------------------------------------------------------
st.title("Dynamic AI-Powered Data Assistant with Local Python Execution")
uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xls", "xlsx"])
user_query = st.text_input("Enter your query:")

# Store file in memory once.
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
    # Display any figures returned by the agent.
    if result.get("figure") is not None:
        figures = result["figure"]
        if not isinstance(figures, list):
            figures = [figures]
        for fig in figures:
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
    st.session_state.memory["chat_history"].append({"user": user_query, "bot": result["response"]})

if st.session_state.memory["chat_history"]:
    st.write("### Chat History:")
    for chat in st.session_state.memory["chat_history"]:
        st.write(f"**You:** {chat['user']}")
        st.write(f"**AI:** {chat['bot']}")