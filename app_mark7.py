import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import xgboost as xgb
from sklearn.model_selection import train_test_split

def load_and_prepare_data():
    # Load the marketing data
    df = pd.read_csv('train.csv', sep=';')

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

    # Store target variable
    y = df['y'].copy()

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna("unknown")
    
    # Exclude target variable from encoding
    df_to_encode = df.drop('y', axis=1)
    df_encoded = pd.get_dummies(df_to_encode, drop_first = True)

    # Drop unwanted columns
    df_encoded = df_encoded.drop(['job_unknown'], axis=1)
    df_encoded = df_encoded.drop(['education_unknown'], axis=1)
    df_encoded = df_encoded.drop(['contact_unknown'], axis=1)
    df_encoded = df_encoded.drop(['poutcome_unknown'], axis=1)
    df_encoded = df_encoded.drop(['poutcome_other'], axis=1)

    # Add back the target variable
    df_encoded['y'] = y

    return df_encoded

def calculate_conversion_ratios(df):
    # Load original data for groupby operations
    df_original = pd.read_csv('train.csv', sep=';')
    df_original.drop_duplicates(inplace=True)
    
    # Calculate overall conversion ratio
    overall_ratio = (df['y'] == 'yes').mean()
    
    # Calculate advanced conversion metrics using original categorical columns
    conversion_by_age_group = df.groupby('age_group', observed=True)['y'].apply(lambda x: (x == 'yes').mean())
    conversion_by_campaign = df.groupby('campaign_intensity', observed=True)['y'].apply(lambda x: (x == 'yes').mean())
    conversion_by_poutcome = df_original.groupby('poutcome')['y'].apply(lambda x: (x == 'yes').mean())
    
    # Calculate conversion ratios by different features using original categorical columns
    conversion_by_job = df_original.groupby('job')['y'].apply(lambda x: (x == 'yes').mean())
    conversion_by_education = df_original.groupby('education')['y'].apply(lambda x: (x == 'yes').mean())
    conversion_by_marital = df_original.groupby('marital')['y'].apply(lambda x: (x == 'yes').mean())
    conversion_by_month = df_original.groupby('month')['y'].apply(lambda x: (x == 'yes').mean())
    
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

def main():
    # Load and analyze data
    df = load_and_prepare_data()
    
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

if __name__ == "__main__":
    main()