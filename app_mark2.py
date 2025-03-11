import io
import base64
from matplotlib import table
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
from typing import Optional, Any, Dict, List
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph
from pydantic import BaseModel

# ----------------------------------------------------------------------
# Initialize Head LLM (Supervisor) – the "Team Lead" that coordinates outputs.
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
# Shared State Definition – holds the current query, uploaded file, and memory.
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
# Two-Way Communication: Agents exchange messages with the head LLM
# to refine and improve their outputs.
# ----------------------------------------------------------------------
def two_way_exchange(agent_name: str, raw_message: str, state: LLMState) -> str:
    context = chat_history_text(state.memory.get("chat_history", []))
    suggestion_prompt = (
        f"You are the head LLM overseeing the Smart AI Data Science Application. The {agent_name} produced the following output:\n"
        f"{raw_message}\n\n"
        f"Conversation context:\n{context}\n\n"
        "Provide a concise suggestion to improve or clarify this output. Respond only with the suggestion."
    )
    suggestion = "".join(chunk.content for chunk in llm.stream(suggestion_prompt)).strip()
    
    update_prompt = (
        f"You are the {agent_name} in the Smart AI Data Science Application. Your original output was:\n"
        f"{raw_message}\n\n"
        f"The head LLM suggests: {suggestion}\n\n"
        "Update your output based on this suggestion. Respond only with the revised output."
    )
    revised_output = "".join(chunk.content for chunk in llm.stream(update_prompt)).strip()
    return revised_output

# ----------------------------------------------------------------------
# Agent: Data Loader – loads file and extracts basic structure.
# ----------------------------------------------------------------------
def data_loader_agent(state: LLMState):
    if state.memory.get("file_data") is None:
        return {"response": "No file uploaded. Please upload a CSV or Excel file."}
    df = state.memory["file_data"]
    raw_output = (
        f"Loaded file: {st.session_state.memory['file_name']}\n"
        f"Shape: {df.shape[0]} rows x {df.shape[1]} columns\n"
        f"Columns: {df.dtypes}\n"
    )
    final_output = two_way_exchange("data_loader_agent", raw_output, state)
    return {"response": final_output}

# ----------------------------------------------------------------------
# Agent: Data Summarizer – computes descriptive statistics.
# ----------------------------------------------------------------------
def data_summarization_agent(state: LLMState):
    if state.memory.get("file_data") is None:
        return {"response": "No file uploaded. Please upload a file for summarization."}
    df = state.memory["file_data"]
    sts = df.describe()
    raw_output = "Descriptive Statistics:\n" + st.table(sts.loc[["count", "mean", "std", "min", "25%", "50%", "75%", "max"]])
    final_output = two_way_exchange("data_summarization_agent", raw_output, state)
    return {"response": final_output}

# ----------------------------------------------------------------------
# Agent: Python Executor – dynamically generates and executes Pandas code
# to answer questions and returns just the execution result.
# ----------------------------------------------------------------------
def python_executor_agent(state: LLMState):
    if state.memory.get("file_data") is None:
        return {"response": "No file loaded. Please upload a file first."}
    df = state.memory["file_data"]
    prompt = (
    f"Generate Pandas code to answer the following question: {state.query}\n"
    f"DataFrame columns available: {', '.join(df.columns)}\n"
    "Do not return only import statements or placeholders. Use the provided DataFrame 'df' to perform a relevant computation and assign the final result to a variable named 'result'.\n"
    "For example, if the question asks for descriptive statistics, your code should include something like:\n"
    "result = df.describe()\n"
    "Return only the code without any markdown formatting."
    )
    raw_code = "".join(chunk.content for chunk in llm.stream(prompt)).strip()
    
    # Remove markdown formatting if present.
    if raw_code.startswith("```"):
        raw_code_lines = raw_code.splitlines()
        if raw_code_lines[0].startswith("```"):
            raw_code_lines = raw_code_lines[1:]
        if raw_code_lines and raw_code_lines[-1].startswith("```"):
            raw_code_lines = raw_code_lines[:-1]
        raw_code = "\n".join(raw_code_lines)
    
    refined_code = two_way_exchange("python_executor_agent", raw_code, state)
    
    if refined_code.startswith("```"):
        refined_code_lines = refined_code.splitlines()
        if refined_code_lines[0].startswith("```"):
            refined_code_lines = refined_code_lines[1:]
        if refined_code_lines and refined_code_lines[-1].startswith("```"):
            refined_code_lines = refined_code_lines[:-1]
        refined_code = "\n".join(refined_code_lines)
    
    # Debug: Log the refined code
    print("Refined Code:\n", refined_code)
    
    try:
        exec_locals = {"df": df}
        exec(refined_code, {}, exec_locals)
        if "result" not in exec_locals:
            return {"response": f"Execution completed, but no 'result' variable was set.\nGenerated Code:\n{refined_code}"}
        result_value = exec_locals.get("result")
        # After executing the code and obtaining 'result'
        if hasattr(result, "axes"):  # crude check if it's a Matplotlib figure
            st.pyplot(result)
        else:
            st.write(result)
    except Exception as e:
        result_value = f"Error executing code: {str(e)}\nCode:\n{refined_code}"
    return {"response": str(result_value)}

# ----------------------------------------------------------------------
# Agent: General AI – for general conversation questions.
# ----------------------------------------------------------------------
def general_ai_agent(state: LLMState):
    chat_hist = chat_history_text(state.memory.get("chat_history", []))
    prompt = (
        f"You are a conversational AI with the following history:\n{chat_hist}\n\n"
        f"User Query: \"{state.query.strip()}\"\n"
        "Provide a clear, concise answer that takes the conversation context into account."
    )
    raw_output = "".join(chunk.content for chunk in llm.stream(prompt))
    final_output = two_way_exchange("general_ai_agent", raw_output, state)
    return {"response": final_output}

# ----------------------------------------------------------------------
# Supervisor: Selects the best agent based on query, file status, and chat history.
# ----------------------------------------------------------------------
def determine_agent(state: LLMState) -> str:
    chat_hist = chat_history_text(state.memory.get("chat_history", []))
    file_status = "yes" if state.memory.get("file_data") is not None else "no"
    
    prompt = (
        "You are the head LLM tasked with choosing the most suitable agent for the current query. "
        "Consider the following context:\n\n"
        f"Chat History:\n{chat_hist}\n\n"
        f"User Query: \"{state.query.strip()}\"\n"
        f"File Available: {file_status}\n\n"
        "Available Agents:\n"
        "- data_loader_agent: For file structure analysis.\n"
        "- data_summarization_agent: For descriptive statistics.\n"
        "- python_executor_agent: For dynamically generating and executing Pandas code (e.g. column relationships).\n"
        "- general_ai_agent: For general conversation.\n\n"
        "Examples for guidance:\n"
        "1. Chat History: None, User Query: 'Please load my data file so I can inspect its structure.', File Available: yes → Expected: data_loader_agent\n"
        "2. Chat History: ['User uploaded file \"data.csv\".'], User Query: 'Can you provide descriptive statistics for the dataset?', File Available: yes → Expected: data_summarization_agent\n"
        "3. Chat History: None, User Query: 'I need to analyze the relationship between the columns \"age\" and \"income\" using code.', File Available: yes → Expected: python_executor_agent\n"
        "4. Chat History: None, User Query: 'What is the weather like today?', File Available: no → Expected: general_ai_agent\n\n"
        "Respond only with the name of the agent that best suits this query."
    )
    
    decision = "".join(chunk.content for chunk in llm.stream(prompt)).strip().lower()
    valid_agents = {"data_loader_agent", "data_summarization_agent", "python_executor_agent", "general_ai_agent"}
    chosen_agent = decision if decision in valid_agents else "general_ai_agent"
    
    selection_prompt = (
        f"You selected {chosen_agent} based on the context. "
        "After further analysis, confirm the final agent selection. Respond only with the final agent name."
    )
    
    final_decision = "".join(chunk.content for chunk in llm.stream(selection_prompt)).strip().lower()
    return final_decision if final_decision in valid_agents else chosen_agent

def supervisor(state: LLMState):
    state.agent = determine_agent(state)
    return {"agent": state.agent}

# ----------------------------------------------------------------------
# Dispatcher: Routes the query to the selected agent.
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
graph.add_node("data_summarization_agent", data_summarization_agent)
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

# Store file in memory once
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
    st.session_state.memory["chat_history"].append({"user": user_query, "bot": result["response"]})

if st.session_state.memory["chat_history"]:
    st.write("### Chat History:")
    for chat in st.session_state.memory["chat_history"]:
        st.write(f"**You:** {chat['user']}")
        st.write(f"**AI:** {chat['bot']}")

