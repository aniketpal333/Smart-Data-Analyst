import io
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import numpy as np
from typing import Optional, Any, Dict, List
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph
from pydantic import BaseModel
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

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
        #"Remove all numerical datas and give a summarised output for the numerical datas."
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

def analyze_dataset_structure(df):
    # First check if 'y' column exists and is binary
    if 'y' in df.columns and df['y'].nunique() <= 2:
        unique_vals = df['y'].unique()
        if 'yes' in unique_vals or 'Yes' in unique_vals:
            return 'y', 'yes' if 'yes' in unique_vals else 'Yes'
        elif 1 in unique_vals or True in unique_vals:
            return 'y', 1 if 1 in unique_vals else True
        else:
            return 'y', unique_vals[-1]
    
    # Look for other binary columns with common target names
    target_names = ['response', 'target', 'outcome', 'label', 'class']
    for col in df.columns:
        if df[col].nunique() <= 2 and any(name in col.lower() for name in target_names):
            unique_vals = df[col].unique()
            if 'yes' in unique_vals or 'Yes' in unique_vals:
                return col, 'yes' if 'yes' in unique_vals else 'Yes'
            elif 1 in unique_vals or True in unique_vals:
                return col, 1 if 1 in unique_vals else True
            else:
                return col, unique_vals[-1]
    
    # If no suitable binary column found, analyze using LLM
    df_info = df.info(buf=io.StringIO(), show_counts=True)
    df_sample = df.head().to_string()
    
    analysis_prompt = f"""Analyze this dataset and identify:
    1. The target/dependent variable (binary classification column)
    2. The positive class value in the target variable

    Some common target column names can be y, response, target, outcome, etc.
    So try to choose target column names like these from the given dataset...

    Dataset Info:
    {df_info}

    Sample Data:
    {df_sample}

    Respond in JSON format only:
    {{
        "target_column": "column_name",
        "positive_class": "value"
    }}
    """
    
    analysis = llm.invoke(analysis_prompt)
    try:
        result = eval(analysis)
        return result["target_column"], result["positive_class"]
    except:
        return 'y', 'yes'  # Default fallback values

def initialize_target_info(df):
    # Clear existing target_info to ensure fresh detection
    if "target_info" in st.session_state:
        del st.session_state.target_info
    
    # Analyze and set new target info
    target_column, positive_class = analyze_dataset_structure(df)
    st.session_state.target_info = {
        "target_column": target_column,
        "positive_class": positive_class
    }

def data_analyser_agent(state:LLMState):
    if state.memory.get("file_data") is None:
        return {"response": "No file uploaded. Please upload a file for summarization."}

    # Initialize target info before data preprocessing
    initialize_target_info(state.memory["file_data"])
    
    # Load and prepare data
    df = load_and_prepare_data()
    
    # Get or set target column and positive class
    if "target_info" not in st.session_state:
        target_column, positive_class = analyze_dataset_structure(state.memory["file_data"], state)
        st.session_state.target_info = {
            "target_column": target_column,
            "positive_class": positive_class
        }
    
    target_column = st.session_state.target_info["target_column"]
    positive_class = st.session_state.target_info["positive_class"]
    
    # Check for derived features
    derived_features = {}
    if 'age_group' in df.columns:
        derived_features['age_group'] = 'age_group'
    if 'campaign_intensity' in df.columns:
        derived_features['campaign_intensity'] = 'campaign_intensity'
    
    # Calculate conversion ratios with dynamic parameters
    conversion_ratios = calculate_conversion_ratios(df, target_column, positive_class, derived_features)
    
    # Analyze feature importance with dynamic parameters
    feature_importance, xgb_model, accuracy, X_test, y_test, y_prob = analyze_feature_importance(df, target_column, positive_class)
    
    # Analyze target audience with dynamic parameters
    target_analysis = analyze_target_audience(df, xgb_model, X_test, y_test, y_prob, probability_threshold=0.7)
    
    # Identify numerical columns for segmentation
    numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
    numerical_cols = [col for col in numerical_cols if col != target_column and col != 'segment']
    
    # Perform customer segmentation with dynamic parameters
    segment_profiles = perform_customer_segmentation(df, target_column, positive_class, n_clusters=4, features=numerical_cols[:4] if len(numerical_cols) > 4 else numerical_cols)
    
    # Generate visualizations with dynamic parameters
    plot_conversion_insights(conversion_ratios, feature_importance, segment_profiles, target_column, numerical_cols)
    
    # Generate recommendations with dynamic parameters
    recommendations = generate_recommendations(conversion_ratios, feature_importance, target_analysis, target_column)
    
    # Format the response
    response = f"""
    \n======== Conversion Insights ========
    {recommendations}

    \n======== Model Performance ========
    \nXGBoost Accuracy: {accuracy:.2%}

    \n======== High-Value Targets ========
    \nTotal High-Value: {target_analysis['total_high_value']}
    \nConversion Rate: {target_analysis['high_value_conversion']:.2%}

    \n======== Customer Segments ========
    \n{segment_profiles.rename(columns={'index':'Segment'}).to_string(float_format='%.2f', header=True, index=False, col_space=18, justify='center', formatters={col: lambda x: f'{x:.2f}' for col in segment_profiles.columns}) if not segment_profiles.empty else 'No segment profiles available'}

    \n======== Target Column Information ========
    \nTarget Column: {target_column}
    """
    
    return {"response": response}

def load_and_prepare_data():
    # Load the marketing data
    df = state.memory["file_data"]

    df = data_preprocessing(df)
        
    # Convert date-related features
    # df['age_group'] = pd.qcut(df['age'], q=4, labels=['Young', 'Adult', 'Middle-aged', 'Senior'])
    
    # # Create campaign intensity using fixed-width bins
    # campaign_min = df['campaign'].min()
    # campaign_max = df['campaign'].max()
    # campaign_range = campaign_max - campaign_min
    # campaign_step = campaign_range / 3
    
    # campaign_bins = [campaign_min, 
    #                 campaign_min + campaign_step,
    #                 campaign_min + 2 * campaign_step,
    #                 campaign_max]
    
    # df['campaign_intensity'] = pd.cut(df['campaign'], 
    #                                  bins=campaign_bins, 
    #                                  labels=['Low', 'Medium', 'High'],
    #                                  include_lowest=True)
    
    return df

def data_preprocessing(df):
    df = df.copy()
    df.drop_duplicates(inplace=True)

    # Initialize target info if not already set
    if "target_info" not in st.session_state:
        target_column, positive_class = analyze_dataset_structure(df)
        st.session_state.target_info = {
            "target_column": target_column,
            "positive_class": positive_class
        }
    
    target_column = st.session_state.target_info["target_column"]
    
    # Ensure target column exists in the DataFrame
    if target_column not in df.columns:
        # Look for common binary target columns
        binary_cols = [col for col in df.columns if df[col].nunique() <= 2]
        if binary_cols:
            target_column = binary_cols[0]  # Use the first binary column found
            st.session_state.target_info["target_column"] = target_column
            # Update positive class based on the new target column
            unique_vals = df[target_column].unique()
            if 1 in unique_vals or True in unique_vals:
                st.session_state.target_info["positive_class"] = 1 if 1 in unique_vals else True
            elif 'yes' in unique_vals or 'Yes' in unique_vals:
                st.session_state.target_info["positive_class"] = 'yes' if 'yes' in unique_vals else 'Yes'
            else:
                st.session_state.target_info["positive_class"] = unique_vals[-1]
    
    # Store target variable
    y = df[target_column].copy()

    # Handle missing values
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna("unknown")

    # Exclude target variable from encoding
    df_to_encode = df.drop(target_column, axis=1)
    df_encoded = pd.get_dummies(df_to_encode, drop_first=True)

    # Automatically drop columns with '_unknown' suffix
    unknown_cols = [col for col in df_encoded.columns if col.endswith('_unknown')]
    df_encoded = df_encoded.drop(unknown_cols, axis=1)

    # Add back the target variable
    df_encoded[target_column] = y

    return df_encoded

def calculate_conversion_ratios(df, target_column=None, positive_class=None, derived_features=None):
    """Calculate conversion ratios for different features in the dataset.
    
    Args:
        df: Processed DataFrame
        target_column: Name of the target column
        positive_class: Value in target column that represents a positive outcome
        derived_features: Dictionary of derived feature names (e.g., {'age_group': 'age_group'})
    
    Returns:
        Dictionary of conversion ratios by different features
    """
    # Get target info from session state
    if target_column is None or positive_class is None:
        target_column = st.session_state.target_info["target_column"]
        positive_class = st.session_state.target_info["positive_class"]
    # Load original data for groupby operations
    df_original = state.memory["file_data"]
    df_original.drop_duplicates(inplace=True)
    
    # Calculate overall conversion ratio
    overall_ratio = (df[target_column] == positive_class).mean()
    
    # Initialize results dictionary
    results = {'overall': overall_ratio}
    
    # Handle derived features if they exist
    if derived_features is not None and isinstance(derived_features, dict):
        for feature_name, column_name in derived_features.items():
            if column_name in df.columns:
                try:
                    results[f'by_{feature_name}'] = df.groupby(column_name, observed=True)[target_column].apply(
                        lambda x: (x == positive_class).mean()
                    )
                except Exception:
                    # Skip if groupby fails
                    pass
    
    # Identify categorical columns in original data for groupby operations
    categorical_cols = df_original.select_dtypes(include=['object', 'category']).columns
    
    # Calculate conversion ratios for each categorical column
    for col in categorical_cols:
        if col != target_column and col in df_original.columns:
            try:
                results[f'by_{col}'] = df_original.groupby(col)[target_column].apply(
                    lambda x: (x == positive_class).mean()
                )
            except Exception:
                # Skip if groupby fails
                pass
    
    return results

def analyze_feature_importance(df, target_column=None, positive_class=None, model_params=None):
    """
    Analyze feature importance using XGBoost model.
    
    Args:
        df: Processed DataFrame
        target_column: Name of the target column
        positive_class: Value in target column that represents a positive outcome
        model_params: Dictionary of XGBoost model parameters
        
    Returns:
        Tuple of (feature_importance, xgb_model, accuracy, X_test, y_test, y_prob)
    """
    # Get target info from session state
    if target_column is None or positive_class is None:
        target_column = st.session_state.target_info["target_column"]
        positive_class = st.session_state.target_info["positive_class"]
    # Prepare data for modeling
    X = pd.get_dummies(df.drop(target_column, axis=1))
    y = (df[target_column] == positive_class).astype(int)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Set default model parameters
    default_params = {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 5,
        'random_state': 42
    }
    
    # Update with custom parameters if provided
    if model_params and isinstance(model_params, dict):
        default_params.update(model_params)
    
    # Train XGBoost model
    xgb_model = xgb.XGBClassifier(**default_params)
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

def perform_customer_segmentation(df, target_column=None, positive_class=None, n_clusters=4, features=None):
    """
    Perform customer segmentation using KMeans clustering.
    
    Args:
        df: Processed DataFrame
        target_column: Name of the target column
        positive_class: Value in target column that represents a positive outcome
        n_clusters: Number of clusters to create
        features: List of numerical features to use for clustering
        
    Returns:
        DataFrame with segment profiles
    """
    # Get target info from session state
    if target_column is None or positive_class is None:
        target_column = st.session_state.target_info["target_column"]
        positive_class = st.session_state.target_info["positive_class"]
    # Dynamically identify numerical features if not provided
    if features is None:
        # Get numerical columns, excluding the target
        numerical_cols = df.select_dtypes(include=['number']).columns
        features = [col for col in numerical_cols if col != target_column and col != 'segment']
        
        # If we have too many features, select top ones by variance
        if len(features) > 5:
            # Select top 4 features by variance
            variances = df[features].var()
            features = variances.nlargest(4).index.tolist()
    
    # Ensure we have valid features
    valid_features = [f for f in features if f in df.columns]
    if not valid_features:
        # Fallback to basic numerical features if available
        for basic_feature in ['age', 'balance', 'amount', 'duration', 'income']:
            if basic_feature in df.columns and pd.api.types.is_numeric_dtype(df[basic_feature]):
                valid_features.append(basic_feature)
                if len(valid_features) >= 3:
                    break
    
    # If still no valid features, return empty DataFrame
    if not valid_features:
        return pd.DataFrame()
    
    # Prepare features for clustering
    X_cluster = StandardScaler().fit_transform(df[valid_features])
    
    # Find optimal number of clusters
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df = df.copy()  # Create a copy to avoid modifying the original
    df['segment'] = kmeans.fit_predict(X_cluster)
    
    # Calculate segment characteristics
    agg_dict = {feature: 'mean' for feature in valid_features}
    agg_dict[target_column] = lambda x: (x == positive_class).mean()
    
    segment_profiles = df.groupby('segment').agg(agg_dict).round(2)
    
    return segment_profiles

def plot_conversion_insights(ratios, feature_importance, segment_profiles, target_column=None, numerical_cols=None):
    """
    Plot conversion insights using Streamlit.
    
    Args:
        ratios: Dictionary of conversion ratios
        feature_importance: DataFrame with feature importance
        segment_profiles: DataFrame with segment profiles
        target_column: Name of the target column
        numerical_cols: List of numerical columns for correlation analysis
    """
    # Get target info from session state
    if target_column is None:
        target_column = st.session_state.target_info["target_column"]
    st.header("Conversion Analysis Dashboard")
    
    # Create two columns for the first row
    col1, col2 = st.columns(2)
    
    # Dynamically select top categorical features to plot
    categorical_plots = []
    for key, value in ratios.items():
        if key.startswith('by_') and not value.empty and len(value) > 1:
            # Limit to top 10 categories by conversion ratio
            value = value.sort_values(ascending=False).head(10)
            categorical_plots.append((key, value))
    
    # Ensure we have at least some plots
    if not categorical_plots:
        st.write("No categorical features available for plotting")
        return
    
    # Plot first categorical feature
    if len(categorical_plots) > 0:
        with col1:
            key, value = categorical_plots[0]
            category_name = key[3:].replace('_', ' ').title()
            fig = plt.figure(figsize=(10, 6))
            value.sort_values().plot(kind='barh')
            plt.title(f'Conversion Ratio by {category_name} (Top 10)', fontsize=12, pad=15)
            plt.xlabel('Conversion Ratio')
            st.pyplot(fig)
            plt.close()
    
    # Plot second categorical feature if available
    if len(categorical_plots) > 1:
        with col2:
            key, value = categorical_plots[1]
            category_name = key[3:].replace('_', ' ').title()
            fig = plt.figure(figsize=(10, 6))
            value.sort_values().plot(kind='barh')
            plt.title(f'Conversion Ratio by {category_name}', fontsize=12, pad=15)
            plt.xlabel('Conversion Ratio')
            st.pyplot(fig)
            plt.close()
    
    # Create two columns for the second row
    col3, col4 = st.columns(2)
    
    # Plot third categorical feature if available
    if len(categorical_plots) > 2:
        with col3:
            key, value = categorical_plots[2]
            category_name = key[3:].replace('_', ' ').title()
            fig = plt.figure(figsize=(10, 6))
            value.sort_values().plot(kind='barh')
            plt.title(f'Conversion Ratio by {category_name}', fontsize=12, pad=15)
            plt.xlabel('Conversion Ratio')
            st.pyplot(fig)
            plt.close()
    
    # Plot feature importance
    with col4 if len(categorical_plots) > 2 else col3:
        if not feature_importance.empty and 'Feature' in feature_importance.columns:
            fig = plt.figure(figsize=(10, 6))
            top_features = feature_importance.head(10)
            sns.barplot(data=top_features, y='Feature', x='Importance', palette='viridis')
            plt.title('Top 10 Important Features', fontsize=12, pad=15)
            st.pyplot(fig)
            plt.close()
        else:
            st.write("Feature importance data not available")
    
    # Create two columns for the third row if we have segment data
    if not segment_profiles.empty and target_column in segment_profiles.columns:
        col5, col6 = st.columns(2)
        
        # Plot derived features if available
        derived_feature_plotted = False
        for key in ratios.keys():
            if key.startswith('by_') and key not in [plot[0] for plot in categorical_plots[:3]]:
                with col5:
                    if not ratios[key].empty:
                        fig = plt.figure(figsize=(10, 6))
                        ratios[key].plot(kind='barh', color='skyblue')
                        category_name = key[3:].replace('_', ' ').title()
                        plt.title(f'Conversion Ratio by {category_name}', fontsize=12, pad=15)
                        plt.xlabel('Conversion Ratio')
                        st.pyplot(fig)
                        plt.close()
                        derived_feature_plotted = True
                        break
        
        # Plot segment profiles
        with col6 if derived_feature_plotted else col5:
            fig = plt.figure(figsize=(10, 6))
            segment_profiles[target_column].plot(kind='bar', color='lightgreen')
            plt.title('Conversion Ratio by Customer Segment', fontsize=12, pad=15)
            plt.xlabel('Segment')
            plt.ylabel('Conversion Ratio')
            st.pyplot(fig)
            plt.close()
    
    # Create correlation heatmap in full width
    st.subheader("Correlation Analysis")
    fig_corr = plt.figure(figsize=(12, 8))
    
    # Get the data
    df = load_and_prepare_data()
    
    # Dynamically identify numerical columns if not provided
    if numerical_cols is None:
        numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
        # Exclude target if it's numerical and any segment column
        numerical_cols = [col for col in numerical_cols if col != target_column and col != 'segment']
        # Limit to top 7 columns to avoid overcrowding
        if len(numerical_cols) > 7:
            numerical_cols = numerical_cols[:7]
    
    # Ensure we have valid numerical columns
    valid_num_cols = [col for col in numerical_cols if col in df.columns]
    
    if valid_num_cols:
        correlation_matrix = df[valid_num_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Correlation Heatmap of Numerical Features', fontsize=12, pad=15)
        st.pyplot(fig_corr)
    else:
        st.write("No numerical columns available for correlation analysis")
    
    plt.close()

def analyze_target_audience(df, xgb_model, X_test, y_test, y_prob, probability_threshold=0.7):
    """
    Analyze high-value target audience based on prediction probabilities.
    
    Args:
        df: Processed DataFrame
        xgb_model: Trained XGBoost model
        X_test: Test features
        y_test: Test target values
        y_prob: Prediction probabilities
        probability_threshold: Threshold to define high-value targets
        
    Returns:
        Dictionary with high-value target analysis
    """
    # Get target column and positive class from session state
    target_column = st.session_state.target_info["target_column"]
    positive_class = st.session_state.target_info["positive_class"]
    
    # Create target audience segments based on prediction probability
    prediction_df = pd.DataFrame({
        'Actual': y_test,
        'Probability': y_prob
    }, index=X_test.index)
    
    # Define high-value targets based on threshold
    high_value_mask = prediction_df['Probability'] > probability_threshold
    high_value_features = X_test[high_value_mask]
    high_value_predictions = prediction_df[high_value_mask]
    
    # Analyze characteristics of high-value targets
    high_value_profile = high_value_features.mean() if not high_value_features.empty else pd.Series()
    
    # Calculate conversion rate for high-value targets
    high_value_conversion = high_value_predictions['Actual'].mean() if not high_value_predictions.empty else 0
    
    # Calculate additional metrics if possible
    metrics = {
        'high_value_profile': high_value_profile,
        'high_value_conversion': high_value_conversion,
        'total_high_value': len(high_value_features),
        'percentage_high_value': len(high_value_features) / len(X_test) if len(X_test) > 0 else 0
    }
    
    return metrics

def generate_recommendations(ratios, feature_importance, target_analysis, target_column):
    """
    Generate recommendations based on analysis results.
    
    Args:
        ratios: Dictionary of conversion ratios
        feature_importance: DataFrame with feature importance
        target_analysis: Dictionary with target audience analysis
        target_column: Name of the target column
        
    Returns:
        String with recommendations
    """
    # Get target column and positive class from session state
    target_column = st.session_state.target_info["target_column"]
    positive_class = st.session_state.target_info["positive_class"]
    
    recommendations = [
        "\nKey findings and recommendations:",
        f"1. Overall {target_column} rate: {ratios['overall']:.2%}"
    ]
    
    # Add top performing segments if available
    segment_insights = []
    segment_count = 2  # Start counter for recommendation numbering
    
    # Dynamically add insights for each categorical feature
    for key, value in ratios.items():
        if key.startswith('by_') and not value.empty:
            try:
                category_name = key[3:]  # Remove 'by_' prefix
                best_category = value.idxmax()
                best_rate = value.max()
                segment_insights.append(
                    f"\n{segment_count}. Best performing {category_name}: {best_category} ({best_rate:.2%})"
                )
                segment_count += 1
                
                # Limit to top 3 segment insights
                if len(segment_insights) >= 3:
                    break
            except Exception:
                # Skip if there's an error getting max value
                continue
    
    if segment_insights:
        recommendations.append("\nTop performing segments:")
        recommendations.extend(segment_insights)
    
    # Add high-value audience insights if available
    if target_analysis and 'total_high_value' in target_analysis:
        recommendations.append("\nHigh-Value Target Audience Insights:")
        recommendations.append(f"\n{segment_count}. Identified {target_analysis['total_high_value']} high-potential customers")
        segment_count += 1
        
        if 'high_value_conversion' in target_analysis:
            recommendations.append(f"\n{segment_count}. High-value segment conversion rate: {target_analysis['high_value_conversion']:.2%}")
            segment_count += 1
    
    # Add feature recommendations if available
    recommendations.append("\nRecommendations for maximizing conversion:")
    
    if not feature_importance.empty and 'Feature' in feature_importance.columns:
        top_features = feature_importance['Feature'].head(3).tolist()
        if top_features:
            recommendations.append(f"\n{segment_count}. Focus on top {len(top_features)} predictive features: {', '.join(top_features)}")
            segment_count += 1
    
    # Add general recommendations
    recommendations.extend([
        f"\n{segment_count}. Target marketing campaigns during high-performing periods",
        f"\n{segment_count + 1}. Prioritize high-value segments with personalized approaches",
        f"\n{segment_count + 2}. Leverage predictive models for lead scoring and prioritization"
    ])
    
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
    
# def data_preprocessing(df):
#     df.drop_duplicates(inplace = True)

#     # Store target variable before preprocessing
#     y = df['y'].copy()

#     # Handle missing values
#     for col in df.columns:
#         if pd.api.types.is_numeric_dtype(df[col]):
#             df[col] = df[col].fillna(df[col].median())
#         else:
#             if pd.api.types.is_categorical_dtype(df[col]):
#                 df[col] = df[col].cat.add_categories(['unknown']).fillna('unknown')
#             else:
#                 df[col] = df[col].fillna("unknown")
    
#     # Encode all columns except target variable
#     df_to_encode = df.drop('y', axis=1)
#     df_encoded = pd.get_dummies(df_to_encode)

#     # Add back the target variable
#     df_encoded['y'] = y

#     # Ensure target variable is preserved
#     if 'y' not in df_encoded.columns:
#         raise ValueError("Target variable 'y' was lost during preprocessing")

#     return df_encoded

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