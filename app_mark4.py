import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import stanza
from transformers import pipeline

# Download and initialize the English pipeline in Stanza (only needed once)
stanza.download('en', verbose=False)
nlp_stanza = stanza.Pipeline('en', processors='tokenize,pos,ner', verbose=False)

# Initialize zero-shot classifier for intent detection using a pre-trained model
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Create a dummy dataset for demonstration
np.random.seed(42)
df = pd.DataFrame({
    "age": np.random.randint(18, 60, 100),
    "salary": np.random.randint(30000, 120000, 100)
})

def show_head(data):
    """Return the first few rows of the dataset."""
    return data.head()

def plot_histogram(data, column):
    """Plot a histogram for the specified column."""
    if column not in data.columns:
        return f"Error: Column '{column}' not found in the dataset."
    plt.figure(figsize=(8, 5))
    sns.histplot(data[column], kde=True, color='skyblue')
    plt.title(f"Histogram of {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.show()
    return f"Displayed histogram for '{column}'."

def calculate_mean(data, column):
    """Calculate and return the mean of the specified column."""
    if column not in data.columns:
        return f"Error: Column '{column}' not found in the dataset."
    mean_val = data[column].mean()
    return f"The mean of '{column}' is {mean_val:.2f}."

def determine_intent(query):
    """
    Use the zero-shot classifier to determine the user's intent.
    Candidate labels represent the possible tasks.
    """
    candidate_labels = ["show head", "plot histogram", "calculate mean"]
    result = classifier(query, candidate_labels)
    return result["labels"][0]

def extract_column(query):
    """
    Use Stanza to tokenize and process the query,
    and extract the token immediately following the word 'of'.
    """
    doc = nlp_stanza(query)
    # Iterate through sentences and tokens in the first sentence (assuming single sentence)
    if doc.sentences:
        sentence = doc.sentences[0]
        for i, word in enumerate(sentence.words):
            if word.text.lower() == "of" and i < len(sentence.words) - 1:
                return sentence.words[i+1].text.lower()
    return None

def process_query(query, data):
    """
    Process the user's query:
      1. Determine the intent using the zero-shot classifier.
      2. Extract any needed parameters (e.g. column name) using Stanza.
      3. Call the appropriate data analysis function.
    """
    intent = determine_intent(query)
    if intent == "show head":
        return show_head(data)
    elif intent == "plot histogram":
        col = extract_column(query)
        if col:
            return plot_histogram(data, col)
        else:
            return ("Please specify the column for histogram plotting. "
                    "For example: 'plot histogram of age'")
    elif intent == "calculate mean":
        col = extract_column(query)
        if col:
            return calculate_mean(data, col)
        else:
            return ("Please specify the column for calculating the mean. "
                    "For example: 'what is the mean of salary'")
    else:
        return ("I'm sorry, I didn't understand your query. "
                "Try 'show head', 'plot histogram of <column>', or 'calculate mean of <column>'.")

def chatbot():
    """Interactive loop for the data analysis chatbot."""
    print("Welcome to the Robust Data Analysis Chatbot (using Stanza)!")
    print("You can ask: 'show head', 'plot histogram of <column>', or 'calculate mean of <column>'.")
    print("Type 'exit' to quit.\n")
    while True:
        user_query = input("You: ")
        if user_query.strip().lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        response = process_query(user_query, df)
        if isinstance(response, pd.DataFrame):
            print(response)
        else:
            print(response)

if __name__ == "__main__":
    chatbot()