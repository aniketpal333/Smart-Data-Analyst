from langchain_google_genai import ChatGoogleGenerativeAI

# Initialize the model with API key
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key="AIzaSyDrmiz2LgB8lWR1T3OmJM9kp9VnrUFIr50",
    streaming=True,
)

# Stream the response
for chunk in llm.stream("Explain how AI works"):
    print(chunk, end="", flush=True)  # Print each streamed chunk