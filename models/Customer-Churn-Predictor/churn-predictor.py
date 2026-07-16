import os
import tempfile
import warnings
import numpy as np
import pandas as pd

import seaborn as sns
import missingno as msno
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from kaggle.api.kaggle_api_extended import KaggleApi
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, 
    confusion_matrix, roc_curve, roc_auc_score
)

warnings.filterwarnings('ignore')

# 1. LOAD DATASET

try:
    api = KaggleApi()
    api.authenticate()
    with tempfile.TemporaryDirectory() as tmp_dir:
        print("Fetching dataset...")
        api.dataset_download_files("blastchar/telco-customer-churn", path=tmp_dir, unzip=False)
        downloaded_files = os.listdir(tmp_dir)
        target_file = downloaded_files[0]       
        file_path = os.path.join(tmp_dir, target_file)
        dataset = pd.read_csv(file_path, compression='infer', dtype_backend="numpy_nullable")
    
    df = dataset.copy() 

except Exception as e:
    print(f"An error occurred while fetching the dataset: {e}")

# 2. EXPLORATORY DATA ANALYSIS & PREPROCESSING

print("--- Initial Data Info ---")
print(df.head())
print("Shape:", df.shape)
print(df.info())
print("Columns:", df.columns.values)
print("Data Types:\n", df.dtypes)
msno.matrix(df)
plt.show()

# Data Cleaning
df = df.drop(['customerID'], axis=1)
df['TotalCharges'] = pd.to_numeric(df.TotalCharges, errors='coerce')

print("\n--- Missing Values ---")
print(df.isnull().sum())

# Fill missing TotalCharges with the column mean
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].mean())

df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})

numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
print("\n--- Numerical Columns Summary ---")
print(df[numerical_cols].describe())

# 3. DATA VISUALIZATION

g_labels = ['Male', 'Female']
c_labels = ['No', 'Yes']

fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]])
fig.add_trace(go.Pie(labels=g_labels, values=df['gender'].value_counts(), name="Gender"), 1, 1)
fig.add_trace(go.Pie(labels=c_labels, values=df['Churn'].value_counts(), name="Churn"), 1, 2)
fig.update_traces(hole=.4, hoverinfo="label+percent+name", textfont_size=16)
fig.update_layout(
    title_text="Gender and Churn Distributions",
    annotations=[
        dict(text='Gender', x=0.16, y=0.5, font_size=20, showarrow=False),
        dict(text='Churn', x=0.84, y=0.5, font_size=20, showarrow=False)
    ]
)
fig.show()

# * 26.6 % of customers switched to another firm.
# * Customers are 49.5 % female and 50.5 % male.

plt.figure(figsize=(6, 6))
labels = ["Churn: Yes","Churn: No"]
values = [1869, 5163]
labels_gender = ["F","M","F","M"]
sizes_gender = [939, 930, 2544, 2619]
colors = ['#ff6666', '#66b3ff']
colors_gender = ['#c2c2f0','#ffb3e6', '#c2c2f0','#ffb3e6']
explode = (0.3, 0.3) 
explode_gender = (0.1, 0.1, 0.1, 0.1)
textprops = {"fontsize": 15}

plt.pie(values, labels=labels, autopct='%1.1f%%', pctdistance=1.08, labeldistance=0.8, colors=colors, startangle=90, frame=True, explode=explode, radius=10, textprops=textprops, counterclock=True)
plt.pie(sizes_gender, labels=labels_gender, colors=colors_gender, startangle=90, explode=explode_gender, radius=7, textprops=textprops, counterclock=True)
centre_circle = plt.Circle((0,0), 5, color='black', fc='white', linewidth=0)
fig_pie = plt.gcf()
fig_pie.gca().add_artist(centre_circle)

plt.title('Churn Distribution w.r.t Gender: Male(M), Female(F)', fontsize=15, y=1.1)
plt.axis('equal')
plt.tight_layout()
plt.savefig('Churn Distribution w.r.t Gender.png', dpi=300)
plt.show()

# * There is negligible difference in customer percentage/ count who changed the service provider. Both genders behaved in similar fashion when it comes to migrating to another service provider/firm.

fig = px.histogram(df, x="Churn", color="Contract", barmode="group", title="<b>Customer contract distribution<b>")
fig.update_layout(width=700, height=500, bargap=0.1)
fig.show()

# * About 75% of customer with Month-to-Month Contract opted to move out as compared to 13% of customrs with One Year Contract and 3% with Two Year Contract

fig = px.histogram(df, x="Churn", color="PaymentMethod", title="<b>Customer Payment Method distribution w.r.t. Churn</b>")
fig.update_layout(width=700, height=500, bargap=0.1)
fig.show()

# * Major customers who moved out were having Electronic Check as Payment Method.
# * Customers who opted for Credit-Card automatic transfer or Bank Automatic Transfer and Mailed Check as Payment Method were less likely to move out.  

fig = go.Figure()
fig.add_trace(go.Bar(x=[['Churn:No', 'Churn:No', 'Churn:Yes', 'Churn:Yes'], ["Female", "Male", "Female", "Male"]], y=[965, 992, 219, 240], name='DSL'))
fig.add_trace(go.Bar(x=[['Churn:No', 'Churn:No', 'Churn:Yes', 'Churn:Yes'], ["Female", "Male", "Female", "Male"]], y=[889, 910, 664, 633], name='Fiber optic'))
fig.add_trace(go.Bar(x=[['Churn:No', 'Churn:No', 'Churn:Yes', 'Churn:Yes'], ["Female", "Male", "Female", "Male"]], y=[690, 717, 56, 57], name='No Internet'))
fig.update_layout(title_text="<b>Churn Distribution w.r.t. Internet Service and Gender</b>")
fig.show()

# * A lot of customers choose the Fiber optic service and it's also evident that the customers who use Fiber optic have high churn rate, this might suggest a dissatisfaction with this type of internet service.
# * Customers having DSL service are majority in number and have less churn rate compared to Fibre optic service.

# Standardized Histogram Plotting
hist_configs = [
    ("Dependents", {"Yes": "#FF97FF", "No": "#AB63FA"}, "group"),
    ("Partner", {"Yes": '#FFA15A', "No": '#00CC96'}, "group"),
    ("SeniorCitizen", {"Yes": '#00CC96', "No": '#B6E880'}, None),
    ("OnlineSecurity", {"Yes": "#FF97FF", "No": "#AB63FA"}, "group"),
    ("PaperlessBilling", {"Yes": '#FFA15A', "No": '#00CC96'}, None),
    ("TechSupport", None, "group"),
    ("PhoneService", {"Yes": '#00CC96', "No": '#B6E880'}, None)
]

for col, cmap, bmode in hist_configs:
    fig = px.histogram(df, x="Churn", color=col, barmode=bmode, title=f"<b>Churn distribution w.r.t. {col}</b>", color_discrete_map=cmap)
    fig.update_layout(width=700, height=500, bargap=0.1)
    fig.show()

# * Customers without dependents are more likely to churn
# * Customers that doesn't have partners are more likely to churn
# * It can be observed that the fraction of senior citizen is very less. Most of the senior citizens churn.
# * Most customers churn in the absence of online security, 
# * Customers with Paperless Billing are most likely to churn.
# * Customers with no TechSupport are most likely to migrate to another service provider.
# * Very small fraction of customers don't have a phone service and out of that, 1/3rd Customers are more likely to churn.

# KDE Plots
sns.set_context("paper", font_scale=1.1)
plt.figure()
ax = sns.kdeplot(df.MonthlyCharges[(df["Churn"] == 'No')], color="Red", fill=True)
ax = sns.kdeplot(df.MonthlyCharges[(df["Churn"] == 'Yes')], ax=ax, color="Blue", fill=True)
ax.legend(["Not Churn","Churn"], loc='upper right')
ax.set(ylabel='Density', xlabel='Monthly Charges', title='Distribution of monthly charges by churn')
plt.savefig('Monthly Charges distribution.png', dpi=300)
plt.show()

# * Customers with higher Monthly Charges are also more likely to churn

plt.figure()
ax = sns.kdeplot(df.TotalCharges[(df["Churn"] == 'No')], color="Gold", fill=True)
ax = sns.kdeplot(df.TotalCharges[(df["Churn"] == 'Yes')], ax=ax, color="Green", fill=True)
ax.legend(["Not Churn","Churn"], loc='upper right')
ax.set(ylabel='Density', xlabel='Total Charges', title='Distribution of total charges by churn')
plt.savefig('Total Charges distribution.png', dpi=300)
plt.show()

# * Customers with lower Total Charges are also more likely to churn

fig = px.box(df, x='Churn', y='tenure')
fig.update_layout(autosize=True, width=750, height=600, title_font=dict(size=25, family='Courier'), title='<b>Tenure vs Churn</b>')
fig.update_yaxes(title_text='Tenure (Months)')
fig.update_xaxes(title_text='Churn')
fig.show()

# * New customers are more likely to churn

plt.figure(figsize=(25, 10))
corr = df.apply(lambda x: pd.factorize(x)[0]).corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, xticklabels=corr.columns, yticklabels=corr.columns, annot=True, linewidths=.2, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Heatmap', fontsize=20)
plt.savefig('Correlation Heatmap.png', dpi=300)
plt.show()

# 4. DATA PREPARATION FOR MODELING 

df = df.drop(columns=[
    "gender",
    "PhoneService"
])

X = df.drop(columns=['Churn'])

# fix: Convert target to numeric binary labels and ensure NumPy indexing works with sklearn
y = df['Churn'].map({'No': 0, 'Yes': 1}).to_numpy()

num_cols = ["tenure", 'MonthlyCharges', 'TotalCharges']

def displot(feature, frame, color='r'):
    plt.figure(figsize=(8,3))
    sns.histplot(frame[feature], color=color, kde=True) 
    plt.title(f"Distribution for {feature}")
    plt.xlabel(feature)
    plt.ylabel('Count')
    plt.savefig(f'Distribution for_{feature}.png', dpi=300)
    plt.show()

for feat in num_cols: 
    displot(feat, df)

feature_names = X.columns
X_values = X.to_numpy()

X_train_arr, X_test_arr, y_train, y_test = train_test_split(X_values, y, test_size=0.30, random_state=42, stratify=y)

X_train = pd.DataFrame(X_train_arr, columns=feature_names)
X_test = pd.DataFrame(X_test_arr, columns=feature_names)

for col in num_cols:
    X_train[col] = pd.to_numeric(X_train[col])
    X_test[col] = pd.to_numeric(X_test[col])

three_level_cols = ['MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                    'TechSupport', 'StreamingTV', 'StreamingMovies']
cat_cols_ohe = ['PaymentMethod', 'Contract', 'InternetService'] + three_level_cols
cat_cols_le = list(set(X_train.columns) - set(num_cols) - set(cat_cols_ohe))

# Label-encode binary/near-binary columns
for col in cat_cols_le:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col])
    X_test[col] = le.transform(X_test[col])

# One-hot encode the multi-category columns
ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
train_ohe = pd.DataFrame(
    ohe.fit_transform(X_train[cat_cols_ohe]),
    columns=ohe.get_feature_names_out(cat_cols_ohe),
    index=X_train.index,
)
test_ohe = pd.DataFrame(
    ohe.transform(X_test[cat_cols_ohe]),
    columns=ohe.get_feature_names_out(cat_cols_ohe),
    index=X_test.index,
)

X_train = pd.concat([X_train.drop(columns=cat_cols_ohe), train_ohe], axis=1)
X_test = pd.concat([X_test.drop(columns=cat_cols_ohe), test_ohe], axis=1)

ratio = (y_train == 0).sum() / (y_train == 1).sum()

# 5. MODEL EVALUATION FUNC

def evaluate_model(model, model_name, X_test, y_test):
    print(f"--- {model_name} ---")
    pred_probs = model.predict_proba(X_test)[:, 1]
    predictions = (pred_probs >= 0.56).astype(int)
    
    print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
    print(f"AUC Score: {roc_auc_score(y_test, pred_probs):.4f}")
    print("\nClassification Report:\n", classification_report(y_test, predictions))
    
    # Confusion Matrix
    plt.figure(figsize=(4,3))
    sns.heatmap(confusion_matrix(y_test, predictions), annot=True, fmt="d", linecolor="k", linewidths=3)
    plt.title(f"{model_name.upper()} CONFUSION MATRIX", fontsize=14)
    plt.savefig(f'{model_name}_Confusion_Matrix.png', dpi=300)
    plt.show()
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, pred_probs)
    plt.figure()
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr, label=model_name, color="r")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} ROC Curve', fontsize=16)
    plt.savefig(f'{model_name}_ROC_Curve.png', dpi=300)
    plt.show()

# Model Training and Evaluation

# Random Forest
model_rf = RandomForestClassifier(n_estimators=500, oob_score=True, n_jobs=6, random_state=42, max_features="sqrt", max_leaf_nodes=30, class_weight='balanced')
model_rf.fit(X_train, y_train)
evaluate_model(model_rf, "Random_Forest", X_test, y_test)

# Logistic Regression
lr_model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
lr_model.fit(X_train, y_train)
evaluate_model(lr_model, "Logistic_Regression", X_test, y_test)

# Gradient Boosting Classifier
gb_model = GradientBoostingClassifier(learning_rate=0.01, subsample=0.8, min_weight_fraction_leaf=0.1, n_estimators=500, max_depth=3, random_state=42)
gb_model.fit(X_train, y_train)
evaluate_model(gb_model, "Gradient_Boosting", X_test, y_test)

# CatBoost
catboost_model = CatBoostClassifier(iterations=1000, l2_leaf_reg=5, learning_rate=0.01, depth=4, eval_metric='AUC', random_seed=42, logging_level='Silent', auto_class_weights='Balanced')
catboost_model.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=True)
evaluate_model(catboost_model, "CatBoost_Classifier", X_test, y_test)

# XGBoost
xgb_model = XGBClassifier(n_estimators=500, scale_pos_weight=ratio * 0.9, learning_rate=0.01, max_depth=3, subsample=0.7, colsample_bytree=0.7, random_state=42)
xgb_model.fit(X_train, y_train)
evaluate_model(xgb_model, "XGBoost_Classifier", X_test, y_test)

# Voting Classifier
VotingClassifier = VotingClassifier(estimators=[('cat', catboost_model), ('xgb', xgb_model), ('lr', lr_model)], voting='soft')
VotingClassifier.fit(X_train, y_train)
evaluate_model(VotingClassifier, "Voting_Classifier", X_test, y_test)
