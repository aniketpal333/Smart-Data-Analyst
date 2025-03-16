import pandas as pd
import numpy as np
import re

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay

from sklearn.preprocessing import StandardScaler

import xgboost as xgb
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

df_train = pd.read_csv("train.csv", sep = ";")
df_test = pd.read_csv("test.csv", sep = ";")

df_legend = pd.concat([df_train, df_test], ignore_index = True)

df_train.head()

df_test.head()

df_train.drop_duplicates(inplace = True)
df_test.drop_duplicates(inplace = True)

for col in df_train.columns:
    if pd.api.types.is_numeric_dtype(df_train[col]):
        df_train[col] = df_train[col].fillna(df_train[col].median())
    else:
        df_train[col] = df_train[col].fillna("unknown")

for col in df_test.columns:
    if pd.api.types.is_numeric_dtype(df_test[col]):
        df_test[col] = df_test[col].fillna(df_test[col].median())
    else:
        df_test[col] = df_test[col].fillna("unknown")

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z09\s]", "", text)
    return text

df_train_encoded = pd.get_dummies(df_train, drop_first = True)
df_test_encoded = pd.get_dummies(df_test, drop_first = True)

df_train_encoded.head()

df_test_encoded.head()

df_train_encoded.isna().sum()

df2 = pd.concat([df_train_encoded, df_test_encoded], ignore_index = True)
df2.head()

df2.describe()

df2.columns

df2 = df2.drop(['job_unknown'], axis=1)
df2 = df2.drop(['education_unknown'], axis=1)
df2 = df2.drop(['contact_unknown'], axis=1)
df2 = df2.drop(['poutcome_unknown'], axis=1)
df2 = df2.drop(['poutcome_other'], axis=1)

df2.columns

target = "poutcome_success"

x = df2.drop(target, axis = 1)
y = df2[target]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

model = LogisticRegression(max_iter = 9000, solver = "saga")
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nModel Accuracy : {accuracy : .2f}")

x_train.head()

x_test.head()

y_train.head()

y_test.head()

y_pred_prob = model.predict_proba(x_test)[:, 1]
x_test_prob = x_test.copy()
x_test_prob["pred_prob"] = y_pred_prob

x_test_prob.head()

df2.columns

x_test_prob.columns

df_legend.columns

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

model_std = LogisticRegression(max_iter = 9000, solver = "saga")
model_std.fit(x_train_scaled, y_train)

coefs = model_std.coef_[0]
feature_names = x_test.columns

coef_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefs})
coef_df['AbsCoefficient'] = coef_df['Coefficient'].abs()
coef_df.sort_values(by='AbsCoefficient', ascending = False, inplace = True)

print("Feature importance based on logistic regression coefficients:")
print(coef_df)

plt.figure(figsize = (12, 6))
plt.bar(coef_df['Feature'], coef_df['AbsCoefficient'], color = 'skyblue')
plt.xlabel('Features')
plt.ylabel('Absolute Coefficient')
plt.title('Feature Importance Based on Absolute Coefficients')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

xgb_model = xgb.XGBClassifier(
    n_estimators=100,       # Number of trees
    learning_rate=0.1,      # Learning rate
    max_depth=3,            # Maximum tree depth
    random_state=42,
    eval_metric='logloss'     # Evaluation metric
)

xgb_model.fit(x_train, y_train)

y_pred_xgb = xgb_model.predict(x_test)

accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
print("XGBoost Model Accuracy: {:.2f}".format(accuracy_xgb))

y_pred_prob_xgb = xgb_model.predict_proba(x_test)[:, 1]
x_test_prob_xgb = x_test.copy()
x_test_prob_xgb["pred_prob"] = y_pred_prob_xgb

xgb.plot_importance(xgb_model, max_num_features=10, importance_type='gain')
plt.title("XGBoost Feature Importance")
plt.tight_layout()
plt.show()

x_test_prob_xgb.head()

plt.figure(figsize=(10, 6))
plt.hist(y_pred_prob, bins=20, alpha=0.5, label='Logistic Regression')
plt.hist(y_pred_prob_xgb, bins=20, alpha=0.5, label='XGBoost')
plt.xlabel('Predicted Probability')
plt.ylabel('Frequency')
plt.title('Histogram of Predicted Probabilities: Logistic Regression vs XGBoost')
plt.legend()
plt.show()

x_test_prob_xgb.sort_values(by='pred_prob', ascending = False, inplace = True)

x_test_prob_xgb.head()

x_test_prob_xgb.shape

x_test_prob_xgb.columns

# Suppose df is your full DataFrame with a 'pred_prob' column.
# First, filter for high-probability customers.
high_prob = x_test_prob_xgb[x_test_prob_xgb['pred_prob'] > 0.72].copy()

# List of target features.
features = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']

# Dictionary to store the common range (IQR) for each feature.
common_ranges = {}

# Remove outliers from each feature using the 1.5 * IQR rule.
for col in features:
    Q1 = high_prob[col].quantile(0.25)
    Q3 = high_prob[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Save the common range (the IQR)
    common_ranges[col] = (Q1, Q3)
    
    # Remove outliers for this feature.
    high_prob = high_prob[(high_prob[col] >= lower_bound) & (high_prob[col] <= upper_bound)]

# Display the most common range for each feature.
print("Most common ranges (IQR) for each feature:")
for col, rng in common_ranges.items():
    print(f"{col}: {rng}")

# Define the list of columns to consider
columns_list = [
    'job_blue-collar', 'job_entrepreneur', 'job_housemaid', 
    'job_management', 'job_retired', 'job_self-employed', 'job_services',
    'job_student', 'job_technician', 'job_unemployed', 'marital_married',
    'marital_single', 'education_secondary', 'education_tertiary',
    'default_yes', 'housing_yes', 'loan_yes', 'contact_telephone',
    'month_aug', 'month_dec', 'month_feb', 'month_jan', 'month_jul',
    'month_jun', 'month_mar', 'month_may', 'month_nov', 'month_oct',
    'month_sep', 'y_yes'
]

# Filter the DataFrame for rows where 'pred_prob' > 0.72
filtered_df = x_test_prob_xgb[x_test_prob_xgb['pred_prob'] > 0.72]

# Create a dictionary to store counts for each column
counts = {}

# For each column in the list, count the rows where the column is True (or 1)
for col in columns_list:
    if col in filtered_df.columns:
        # If the column is numeric (0/1) or boolean, this will work.
        # If the column is a string, you might need to map it to a boolean.
        count_true = filtered_df[col].astype(bool).sum()
        counts[col] = count_true

# Print the counts
print("Counts for each column where pred_prob > 0.72 and the column is True:")
for col, count in counts.items():
    print(f"{col}: {count}")

# Define the counts dictionary based on your extracted data
counts = {
    'job_blue-collar': 4,
    'job_entrepreneur': 2,
    'job_housemaid': 1,
    'job_management': 5,
    'job_retired': 13,
    'job_self-employed': 0,
    'job_services': 2,
    'job_student': 2,
    'job_technician': 8,
    'job_unemployed': 3,
    'marital_married': 29,
    'marital_single': 11,
    'education_secondary': 17,
    'education_tertiary': 18,
    'default_yes': 0,
    'housing_yes': 0,
    'loan_yes': 0,
    'contact_telephone': 7,
    'month_aug': 7,
    'month_dec': 0,
    'month_feb': 4,
    'month_jan': 1,
    'month_jul': 3,
    'month_jun': 3,
    'month_mar': 2,
    'month_may': 5,
    'month_nov': 0,
    'month_oct': 7,
    'month_sep': 9
}

def get_max_category(counts, prefix):
    # Filter the dictionary for keys starting with the prefix
    filtered = {k: v for k, v in counts.items() if k.startswith(prefix)}
    if filtered:
        # Find the key with the maximum count
        max_key = max(filtered, key=filtered.get)
        return max_key, filtered[max_key]
    else:
        return None, None

# Extract the maximum for each category
max_job, job_count = get_max_category(counts, 'job_')
max_marital, marital_count = get_max_category(counts, 'marital_')
max_education, education_count = get_max_category(counts, 'education_')
max_month, month_count = get_max_category(counts, 'month_')

print("Most common job:", max_job, "with count:", job_count)
print("Most common marital status:", max_marital, "with count:", marital_count)
print("Most common education:", max_education, "with count:", education_count)
print("Most common month:", max_month, "with count:", month_count)

