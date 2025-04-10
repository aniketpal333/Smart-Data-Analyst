import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def load_and_prepare_data():
    # Load the marketing data
    df = pd.read_csv('train.csv', sep=';')
    
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
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train logistic regression
    model = LogisticRegression(max_iter=9000, solver='saga')
    model.fit(X_scaled, y)
    
    # Get feature importance
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': abs(model.coef_[0])
    }).sort_values('Importance', ascending=False)
    
    return feature_importance

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

def generate_recommendations(ratios, feature_importance):
    recommendations = [
        "Key findings and recommendations:",
        f"1. Overall conversion rate: {ratios['overall']:.2%}",
        "\nTop performing segments:",
        f"2. Best performing job category: {ratios['by_job'].idxmax()} ({ratios['by_job'].max():.2%})",
        f"3. Best performing education level: {ratios['by_education'].idxmax()} ({ratios['by_education'].max():.2%})",
        f"4. Best performing month: {ratios['by_month'].idxmax()} ({ratios['by_month'].max():.2%})",
        "\nRecommendations for maximizing conversion:",
        f"5. Focus on top 3 influential features: {', '.join(feature_importance['Feature'].head(3))}",
        "6. Target marketing campaigns during high-performing months",
        "7. Customize approach based on education and job segments"
    ]
    return '\n'.join(recommendations)

def main():
    # Load and analyze data
    df = load_and_prepare_data()
    
    # Calculate conversion ratios
    conversion_ratios = calculate_conversion_ratios(df)
    
    # Analyze feature importance
    feature_importance = analyze_feature_importance(df)
    
    # Perform customer segmentation
    segment_profiles = perform_customer_segmentation(df)
    
    # Generate visualizations
    plot_conversion_insights(conversion_ratios, feature_importance, segment_profiles)
    
    # Generate and print recommendations
    recommendations = generate_recommendations(conversion_ratios, feature_importance)
    print(recommendations)
    
    # Print segment insights
    print("\nCustomer Segment Profiles:")
    print(segment_profiles)

if __name__ == "__main__":
    main()