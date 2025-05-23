import io
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
from typing import Optional, Any, Dict, List
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph
from pydantic import BaseModel
from sklearn.model_selection import train_test_split
import xgboost as xgb

# ----------------------------------------------------------------------
# Initialize Head LLM (Supervisor) – the "Team Lead" that coordinates outputs.
# ----------------------------------------------------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key="AIzaSyDrmiz2LgB8lWR1T3OmJM9kp9VnrUFIr50",
    streaming=True,
    temperature=0.2,
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
    return "\n".join(
        [f"User: {entry['user']}\nAI: {entry['bot']}" for entry in history]
    )


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
        "Remove all numerical datas and give a summarised output for the numerical datas."
    )
    suggestion = "".join(
        chunk.content for chunk in llm.stream(suggestion_prompt)
    ).strip()

    update_prompt = (
        f"You are the {agent_name} in the Smart AI Data Science Application. Your original output was:\n"
        f"{raw_message}\n\n"
        f"The head LLM suggests: {suggestion}\n\n"
        "Update your output based on this suggestion. Respond only with the revised output."
    )
    revised_output = "".join(
        chunk.content for chunk in llm.stream(update_prompt)
    ).strip()
    return revised_output

def data_analyser_agent(state:LLMState):
    if state.memory.get("file_data") is None:
        return {"response": "No file uploaded. Please upload a file for summarization."}

    df = state.memory["file_data"]

    # Load and analyze data
    df = load_and_prepare_data(df)
    
    # Calculate conversion ratios
    conversion_ratios = calculate_conversion_ratios(df)
    
    # Analyze feature importance and get XGBoost predictions
    feature_importance, xgb_model, accuracy, X_test, y_test, y_prob = analyze_feature_importance(df)
    
    # Analyze target audience
    target_analysis = analyze_target_audience(df, xgb_model, X_test, y_test, y_prob)
    
    # Perform customer segmentation
    segment_profiles = perform_customer_segmentation(df)
    
    # Generate visualizations
    plot_conversion_insights(conversion_ratios, feature_importance, segment_profiles)
    
    # Generate and print recommendations
    recommendations = generate_recommendations(conversion_ratios, feature_importance, target_analysis)
    print(recommendations)
    
    # Print model performance
    print(f"\nXGBoost Model Accuracy: {accuracy:.2%}")
    
    # Print target audience insights
    print("\nHigh-Value Target Audience Insights:")
    print(f"Number of high-value targets: {target_analysis['total_high_value']}")
    print(f"Conversion rate among high-value targets: {target_analysis['high_value_conversion']:.2%}")
    
    # Print segment insights
    print("\nCustomer Segment Profiles:")
    print(segment_profiles)

def load_and_prepare_data(df):
    # Load the marketing data
    #df = pd.read_csv('train.csv', sep=';')

    df = data_preprocessing(df)
        
    # Convert date-related features
    df['age_group'] = pd.qcut(df['age'], q=4, labels=['Young', 'Adult', 'Middle-aged', 'Senior'])
    
    # Create campaign intensity using fixed-width bins
    campaign_min = df['campaign'].min()
    campaign_max = df['campaign'].max()
    campaign_range = campaign_max - campaign_min
    campaign_step = campaign_range / 3
    
    campaign_bins = [campaign_min, 
                    campaign_min + campaign_step,
                    campaign_min + 2 * campaign_step,
                    campaign_max]
    
    df['campaign_intensity'] = pd.cut(df['campaign'], 
                                     bins=campaign_bins, 
                                     labels=['Low', 'Medium', 'High'],
                                     include_lowest=True)
    
    return df

def data_preprocessing(df):
    df.drop_duplicates(inplace = True)

    # Store target variable and poutcome
    y = df['y'].copy()
    poutcome = df['poutcome'].copy()

    # Store original categorical columns needed for groupby
    categorical_cols = ['job', 'education', 'marital', 'month']
    original_cats = {col: df[col].copy() for col in categorical_cols}

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna("unknown")
    
    # Exclude target variable and poutcome from encoding
    df_to_encode = df.drop(['y', 'poutcome'] + categorical_cols, axis=1)
    df_encoded = pd.get_dummies(df_to_encode, drop_first = True)

    # Drop unwanted columns
    df_encoded = df_encoded.drop(['contact_unknown'], axis=1)

    # Add back the original categorical columns
    for col in categorical_cols:
        df_encoded[col] = original_cats[col]

    # Add back the target variable and poutcome
    df_encoded['y'] = y
    df_encoded['poutcome'] = poutcome

    return df_encoded

def calculate_conversion_ratios(df):
    # Calculate overall conversion ratio
    overall_ratio = (df['y'] == 'yes').mean()
    
    # Calculate advanced conversion metrics
    conversion_by_age_group = df.groupby('age_group', observed=True)['y'].apply(lambda x: (x == 'yes').mean())
    conversion_by_campaign = df.groupby('campaign_intensity', observed=True)['y'].apply(lambda x: (x == 'yes').mean())
    conversion_by_poutcome = df.groupby('poutcome', observed=True)['y'].apply(lambda x: (x == 'yes').mean())
    
    # Calculate conversion ratios by different features
    conversion_by_job = df.groupby('job', observed=True)['y'].apply(lambda x: (x == 'yes').mean())
    conversion_by_education = df.groupby('education', observed=True)['y'].apply(lambda x: (x == 'yes').mean())
    conversion_by_marital = df.groupby('marital', observed=True)['y'].apply(lambda x: (x == 'yes').mean())
    conversion_by_month = df.groupby('month', observed=True)['y'].apply(lambda x: (x == 'yes').mean())
    
    return {
        'overall': overall_ratio,
        'by_job': conversion_by_job,
        'by_education': conversion_by_education,
        'by_marital': conversion_by_marital,
        'by_month': conversion_by_month,
        'by_age_group': conversion_by_age_group,
        'by_campaign': conversion_by_campaign,
        'by_poutcome': conversion_by_poutcome
    }

def analyze_feature_importance(df):
    # Prepare data for modeling
    X = pd.get_dummies(df.drop('y', axis=1))
    y = (df['y'] == 'yes').astype(int)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train XGBoost model
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    xgb_model.fit(X_train, y_train)
    
    # Get predictions and probabilities
    y_pred = xgb_model.predict(X_test)
    y_prob = xgb_model.predict_proba(X_test)[:, 1]
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Get feature importance
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': xgb_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    return feature_importance, xgb_model, accuracy, X_test, y_test, y_prob

def perform_customer_segmentation(df):
    # Prepare numerical features for clustering
    cluster_features = ['age', 'balance', 'campaign', 'previous']
    X_cluster = StandardScaler().fit_transform(df[cluster_features])
    
    # Find optimal number of clusters
    n_clusters = 4  # You can make this dynamic based on silhouette score
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['segment'] = kmeans.fit_predict(X_cluster)
    
    # Calculate segment characteristics
    segment_profiles = df.groupby('segment').agg({
        'age': 'mean',
        'balance': 'mean',
        'campaign': 'mean',
        'y': lambda x: (x == 'yes').mean()
    }).round(2)
    
    return segment_profiles

def plot_conversion_insights(ratios, feature_importance, segment_profiles):
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 25))
    gs = fig.add_gridspec(4, 2)
    
    # Plot conversion by job
    ax1 = fig.add_subplot(gs[0, 0])
    ratios['by_job'].sort_values().plot(kind='barh', ax=ax1)
    ax1.set_title('Conversion Ratio by Job', fontsize=12, pad=15)
    ax1.set_xlabel('Conversion Ratio')
    
    # Plot conversion by education
    ax2 = fig.add_subplot(gs[0, 1])
    ratios['by_education'].sort_values().plot(kind='barh', ax=ax2)
    ax2.set_title('Conversion Ratio by Education', fontsize=12, pad=15)
    ax2.set_xlabel('Conversion Ratio')
    
    # Plot conversion by month with enhanced styling
    ax3 = fig.add_subplot(gs[1, 0])
    month_plot = ratios['by_month'].sort_values()
    bars = month_plot.plot(kind='barh', ax=ax3)
    ax3.set_title('Conversion Ratio by Month', fontsize=12, pad=15)
    ax3.set_xlabel('Conversion Ratio')
    
    # Plot feature importance with enhanced styling
    ax4 = fig.add_subplot(gs[1, 1])
    top_features = feature_importance.head(10)
    sns.barplot(data=top_features, y='Feature', x='Importance', ax=ax4, palette='viridis')
    ax4.set_title('Top 10 Important Features', fontsize=12, pad=15)
    
    # Plot conversion by age group with enhanced styling
    ax5 = fig.add_subplot(gs[2, 0])
    ratios['by_age_group'].plot(kind='barh', ax=ax5, color='skyblue')
    ax5.set_title('Conversion Ratio by Age Group', fontsize=12, pad=15)
    ax5.set_xlabel('Conversion Ratio')
    
    # Plot segment profiles with enhanced styling
    ax6 = fig.add_subplot(gs[2, 1])
    segment_profiles['y'].plot(kind='bar', ax=ax6, color='lightgreen')
    ax6.set_title('Conversion Ratio by Customer Segment', fontsize=12, pad=15)
    ax6.set_xlabel('Segment')
    ax6.set_ylabel('Conversion Ratio')
    
    # Add correlation heatmap for numerical features
    ax7 = fig.add_subplot(gs[3, :])
    df = load_and_prepare_data()
    numerical_cols = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
    correlation_matrix = df[numerical_cols].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=ax7)
    ax7.set_title('Correlation Heatmap of Numerical Features', fontsize=12, pad=15)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('conversion_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def analyze_target_audience(df, xgb_model, X_test, y_test, y_prob):
    # Create target audience segments based on prediction probability
    prediction_df = pd.DataFrame({
        'Actual': y_test,
        'Probability': y_prob
    }, index=X_test.index)
    
    # Define high-value targets (probability > 0.7)
    high_value_mask = prediction_df['Probability'] > 0.7
    high_value_features = X_test[high_value_mask]
    high_value_predictions = prediction_df[high_value_mask]
    
    # Analyze characteristics of high-value targets
    high_value_profile = high_value_features.mean() if not high_value_features.empty else pd.Series()
    
    # Calculate conversion rate for high-value targets
    high_value_conversion = high_value_predictions['Actual'].mean() if not high_value_predictions.empty else 0
    
    return {
        'high_value_profile': high_value_profile,
        'high_value_conversion': high_value_conversion,
        'total_high_value': len(high_value_features)
    }

def generate_recommendations(ratios, feature_importance, target_analysis):
    recommendations = [
        "Key findings and recommendations:",
        f"1. Overall conversion rate: {ratios['overall']:.2%}",
        "\nTop performing segments:",
        f"2. Best performing job category: {ratios['by_job'].idxmax()} ({ratios['by_job'].max():.2%})",
        f"3. Best performing education level: {ratios['by_education'].idxmax()} ({ratios['by_education'].max():.2%})",
        f"4. Best performing month: {ratios['by_month'].idxmax()} ({ratios['by_month'].max():.2%})",
        "\nHigh-Value Target Audience Insights:",
        f"5. Identified {target_analysis['total_high_value']} high-potential customers",
        f"6. High-value segment conversion rate: {target_analysis['high_value_conversion']:.2%}",
        "\nRecommendations for maximizing conversion:",
        f"7. Focus on top 3 predictive features: {', '.join(feature_importance['Feature'].head(3))}",
        "8. Target marketing campaigns during high-performing months",
        "9. Prioritize high-value segments with personalized approaches",
        "10. Leverage XGBoost predictions for lead scoring and prioritization"
    ]
    return '\n'.join(recommendations)

    # prompt = f"""
    # Given the following dataset columns: {list(df.columns)},
    # identify the most probable target features for predictive modeling.
    # Return only the column names that are suitable as target variables.
    # """
    # response = llm.predict(prompt)
    # response = response.split(",")

    # probable_targets = response
    # selected_option = st.selectbox("Choose a target feature:", probable_targets)
    # st.write(f"You selected: {selected_option}")

#     x = df.drop("poutcome_success", axis = 1)
#     y = df["poutcome_success"]

#     x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

#     xgb_model = xgb.XGBClassifier(
#     n_estimators=100,       # Number of trees
#     learning_rate=0.1,      # Learning rate
#     max_depth=3,            # Maximum tree depth
#     random_state=42,
#     eval_metric='logloss'     # Evaluation metric
#     )

#     xgb_model.fit(x_train, y_train)

#     y_pred_prob_xgb = xgb_model.predict_proba(x_test)[:, 1]
#     x_test_prob_xgb = x_test.copy()
#     x_test_prob_xgb["pred_prob"] = y_pred_prob_xgb

#     x_test_prob_xgb.sort_values(by='pred_prob', ascending = False, inplace = True)

#     high_prob = x_test_prob_xgb[x_test_prob_xgb['pred_prob'] > 0.72].copy()

#     # List of target features.
#     features = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']

#     # Dictionary to store the common range (IQR) for each feature.
#     common_ranges = {}

#     # Remove outliers from each feature using the 1.5 * IQR rule.
#     for col in features:
#         Q1 = high_prob[col].quantile(0.25)
#         Q3 = high_prob[col].quantile(0.75)
#         IQR = Q3 - Q1
#         lower_bound = Q1 - 1.5 * IQR
#         upper_bound = Q3 + 1.5 * IQR
    
#         # Save the common range (the IQR)
#         common_ranges[col] = (round(Q1), round(Q3))
    
#         # Remove outliers for this feature.
#         high_prob = high_prob[(high_prob[col] >= lower_bound) & (high_prob[col] <= upper_bound)]

#     # Define the list of columns to consider
#     columns_list = [
#         'job_blue-collar', 'job_entrepreneur', 'job_housemaid', 
#         'job_management', 'job_retired', 'job_self-employed', 'job_services',
#         'job_student', 'job_technician', 'job_unemployed', 'marital_married',
#         'marital_single', 'education_secondary', 'education_tertiary',
#         'default_yes', 'housing_yes', 'loan_yes', 'contact_telephone',
#         'month_aug', 'month_dec', 'month_feb', 'month_jan', 'month_jul',
#         'month_jun', 'month_mar', 'month_may', 'month_nov', 'month_oct',
#         'month_sep', 'y_yes'
#     ]

#     # Filter the DataFrame for rows where 'pred_prob' > 0.72
#     filtered_df = x_test_prob_xgb[x_test_prob_xgb['pred_prob'] > 0.72]

#     # Create a dictionary to store counts for each column
#     counts = {}

#     # For each column in the list, count the rows where the column is True (or 1)
#     for col in columns_list:
#         if col in filtered_df.columns:
#             # If the column is numeric (0/1) or boolean, this will work.
#             # If the column is a string, you might need to map it to a boolean.
#             count_true = filtered_df[col].astype(bool).sum()
#             counts[col] = count_true

#     # Extract the maximum for each category
#     max_job, job_count = get_max_category(counts, 'job_')
#     max_marital, marital_count = get_max_category(counts, 'marital_')
#     max_education, education_count = get_max_category(counts, 'education_')
#     max_month, month_count = get_max_category(counts, 'month_')

#     category_counts = {
#         "max_job": {"category": max_job, "count": job_count},
#         "max_marital": {"category": max_marital, "count": marital_count},
#         "max_education": {"category": max_education, "count": education_count},
#         "max_month": {"category": max_month, "count": month_count},
#     }

#     # Convert category_counts and common_ranges into human-readable text
#     category_counts_str = "\n".join(
#         [f"- {key.replace('max_', '').capitalize()}: {value['category']}" #(Count: {value['count']})"
#         for key, value in category_counts.items()]
#     )

#     common_ranges_str = "\n".join(
#         [f"- {col}: Q1={iqr[0]}, Q3={iqr[1]}" for col, iqr in common_ranges.items()]
#     )

#     # Create the raw message to be sent to the two-way exchange function
#     raw_message = f"""
#     The analysis identified the most prominent categorical features among high-probability data points: {category_counts_str}
#     Additionally, we determined the common value ranges for numerical features: {common_ranges_str}
#     """

#     # Call the two-way exchange function to refine the response
#     human_readable_output = two_way_exchange("Analysis Agent", raw_message, state)

#     # Display the final result
#     return {"response": human_readable_output}


# def get_max_category(counts, prefix):
#     # Filter the dictionary for keys starting with the prefix
#     filtered = {k: v for k, v in counts.items() if k.startswith(prefix)}
#     if filtered:
#         # Find the key with the maximum count
#         max_key = max(filtered, key=filtered.get)
#         return max_key, filtered[max_key]
#     else:
#         return None, None
    
def data_preprocessing(df):
    df.drop_duplicates(inplace = True)

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna("unknown")
    
    df_encoded = pd.get_dummies(df, drop_first = True)

    df = df_encoded.drop(['job_unknown'], axis=1)
    df = df_encoded.drop(['education_unknown'], axis=1)
    df = df_encoded.drop(['contact_unknown'], axis=1)
    df = df_encoded.drop(['poutcome_unknown'], axis=1)
    df = df_encoded.drop(['poutcome_other'], axis=1)

    return df

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
# Agent: Data Summarizer – computes descriptive statistics and performs trend analysis.
# ----------------------------------------------------------------------
def data_summarization_agent(state: LLMState):
    if state.memory.get("file_data") is None:
        return {"response": "No file uploaded. Please upload a file for summarization."}

    df = state.memory["file_data"]
    summary_stats = df.describe().to_string()

    trend_analysis_text = ""
    figure = None
    # Check for a datetime column for trend analysis
    time_columns = [
        col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])
    ]
    if time_columns:
        time_col = time_columns[0]
        df_sorted = df.sort_values(by=time_col)
        # Choose first numerical column (excluding the datetime column)
        numeric_cols = df_sorted.select_dtypes(include=["number"]).columns.tolist()
        if numeric_cols:
            trend_col = numeric_cols[0]
            rolling_avg = df_sorted[trend_col].rolling(window=5).mean()
            trend_analysis_text = f"\nTrend Analysis on {trend_col} (Rolling Average, window=5):\n{rolling_avg.tail(5).to_string()}"

            # Create a matplotlib figure for visualization
            fig, ax = plt.subplots()
            ax.plot(df_sorted[time_col], df_sorted[trend_col], label="Original Data")
            ax.plot(
                df_sorted[time_col],
                rolling_avg,
                label="Rolling Average",
                linestyle="--",
            )
            ax.set_xlabel(time_col)
            ax.set_ylabel(trend_col)
            ax.set_title(f"Trend Analysis of {trend_col}")
            ax.legend()
            figure = fig

    raw_output = f"Descriptive Statistics:\n{summary_stats}{trend_analysis_text}"
    final_output = two_way_exchange("data_summarization_agent", raw_output, state)
    # Return the textual response and, if available, the figure for visualization.
    return {"response": final_output, "figure": figure}


# ----------------------------------------------------------------------
# Agent: Python Executor – dynamically generates and executes Pandas code
# to answer questions and returns just the execution result.
# ----------------------------------------------------------------------
def python_executor_agent(state: LLMState):
    if state.memory.get("file_data") is None:
        return {"response": "No file loaded. Please upload a file first."}
    df = state.memory["file_data"]
    query_lower = state.query.lower()

    # If query asks for success or failure rates, compute and visualize them.
    if "success rate" in query_lower or "failure rate" in query_lower:
        status_column = next((col for col in df.columns if "y" in col.lower()), None)
        if status_column:
            rates = df[status_column].value_counts(normalize=True) * 100
            response_text = f"Success/Failure Rate (%):\n{rates.to_string()}"

            # Create a bar chart using matplotlib
            fig, ax = plt.subplots()
            rates.plot(kind="bar", ax=ax)
            ax.set_ylabel("Percentage")
            ax.set_title(f"Success/Failure Rate for {status_column}")
            return {"response": response_text, "figure": fig}
        else:
            return {
                "response": "No status column found to compute success/failure rate."
            }

    # Otherwise, generate and execute Pandas code dynamically.
    prompt = (
        f"Generate Pandas code to answer the following question: {state.query}\n"
        f"DataFrame columns available: {', '.join(df.columns)}\n"
        "Do not return only import statements or placeholders—write the complete executable code.\n"
        "Use the provided DataFrame `df` to perform the computation and assign the final result to a variable named `result`.\n"
        "For example, if the question asks for descriptive statistics, your code should include something like:\n"
        "`result = df.describe()`\n"
        "Return only the raw Python code—do not include explanations, markdown formatting, or comments.\n"
        "Ensure that the final computed value is stored in a variable named `result`\n"
        "Do not generate comments and ensure perfect indentation.\n"
        "Do not define `df`, it will automatically be injected in the code.\n"
        "The code structure should be like this: 1. import statements\n2. run(df) function and 3.calling the run(df)"
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
    print(
        f"""# Executor agent used:\ndf: {df}\nraw_code: {raw_code}\nprompt: {prompt}\nrefined_code: {refined_code}"""
    )

    try:
        exec_vars = {"df": df}
        exec(refined_code, exec_vars, exec_vars)
        print(f"""# Executor agent finished:\nexec_locals: {exec_vars}""")
        if "result" not in exec_vars:
            return {
                "response": f"Execution completed, but no 'result' variable was set.\nGenerated Code:\n{refined_code}"
            }
        result_value = exec_vars.get("result")
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
        f'User Query: "{state.query.strip()}"\n'
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
        f'User Query: "{state.query.strip()}"\n'
        f"File Available: {file_status}\n\n"
        "Available Agents:\n"
        "- data_loader_agent: For file structure analysis.\n"
        "- data_summarization_agent: For descriptive statistics and trend analysis.\n"
        "- data_analyser_agent: For analysing and giving the output for the target audiences.\n"
        "- python_executor_agent: For dynamically generating and executing Pandas code (e.g. column relationships, success/failure rates).\n"
        "- general_ai_agent: For general conversation.\n\n"
        "Examples for guidance:\n"
        "1. Chat History: None, User Query: 'Please load my data file so I can inspect its structure.', File Available: yes → Expected: data_loader_agent\n"
        "2. Chat History: ['User uploaded file \"data.csv\".'], User Query: 'Can you provide descriptive statistics for the dataset?', File Available: yes → Expected: data_summarization_agent\n"
        "3. Chat History: None, User Query: 'I need to analyse the data provided.', File Available: yes → Expected: data_analyser_agent\n"
        "4. Chat History: None, User Query: 'I need to check the relationship between the columns \"age\" and \"income\" using code.', File Available: yes → Expected: python_executor_agent\n"
        "5. Chat History: None, User Query: 'What is the weather like today?', File Available: no → Expected: general_ai_agent\n\n"
        "Respond only with the name of the agent that best suits this query."
    )


    decision = "".join(chunk.content for chunk in llm.stream(prompt)).strip().lower()
    valid_agents = {
        "data_loader_agent",
        "data_summarization_agent",
        "data_analyser_agent",
        "python_executor_agent",
        "general_ai_agent",
    }
    chosen_agent = decision if decision in valid_agents else "general_ai_agent"

    selection_prompt = (
        f"You selected {chosen_agent} based on the context. "
        "After further analysis, confirm the final agent selection. Respond only with the final agent name."
    )

    final_decision = (
        "".join(chunk.content for chunk in llm.stream(selection_prompt)).strip().lower()
    )
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
graph.add_node("data_analyser_agent", data_analyser_agent)
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
            return pd.read_csv(
                io.StringIO(file.getvalue().decode("utf-8")), sep=None, engine="python"
            )
        elif file.name.endswith((".xls", ".xlsx")):
            return pd.read_excel(file)
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None


# ----------------------------------------------------------------------
# Streamlit UI
# ----------------------------------------------------------------------
st.title("Dynamic AI-Powered Data Assistant with Local Python Execution")
uploaded_file = st.file_uploader(
    "Upload a CSV or Excel file", type=["csv", "xls", "xlsx"]
)
user_query = st.text_input("Enter your query:")

# Store file in memory once
if (
    uploaded_file is not None
    and uploaded_file.name != st.session_state.memory["file_name"]
):
    st.session_state.memory["file_data"] = load_file(uploaded_file)
    st.session_state.memory["file_name"] = uploaded_file.name

if st.button("Submit"):
    state = LLMState(
        query=user_query, file=uploaded_file, memory=st.session_state.memory
    )
    result = executable.invoke(state)
    st.write(f"**Chatbot:** {result['response']}")
    # If a matplotlib figure is returned, display it.
    if result.get("figure") is not None:
        st.pyplot(result["figure"])
    st.session_state.memory["chat_history"].append(
        {"user": user_query, "bot": result["response"]}
    )

if st.session_state.memory["chat_history"]:
    st.write("### Chat History:")
    for chat in st.session_state.memory["chat_history"]:
        st.write(f"**You:** {chat['user']}")
        st.write(f"**AI:** {chat['bot']}")