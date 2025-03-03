from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph
from pydantic import BaseModel

# Initialize the model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key="AIzaSyDrmiz2LgB8lWR1T3OmJM9kp9VnrUFIr50",
    streaming=True,
)

# Define the state using Pydantic
class LLMState(BaseModel):
    query: str

# Define a function to query the LLM
def query_llm(state: LLMState):
    response = llm.stream(state.query)
    for chunk in response:
        print(chunk, end="", flush=True)
    return {}

# Define the graph
graph = StateGraph(LLMState)
graph.add_node("llm_query", query_llm)
graph.set_entry_point("llm_query")

# Compile the graph
executable = graph.compile()

# Run the graph with a query
executable.invoke(LLMState(query="Explain how AI works"))