import os
import tempfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from kaggle.api.kaggle_api_extended import KaggleApi
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

def main():

    # LOAD DATASET
    
    api = KaggleApi()
    api.authenticate()

    with tempfile.TemporaryDirectory() as tmp_dir:
        print("Fetching dataset...")
        api.dataset_download_files("ealaxi/paysim1", path=tmp_dir, unzip=False)
        downloaded_files = os.listdir(tmp_dir)
        target_file = downloaded_files[0]
        file_path = os.path.join(tmp_dir, target_file)
        
        df = pd.read_csv(file_path, compression='infer')
        fresh_df = df.copy()  # Keep a fresh copy for later use 
        
    print(df.head())

    df.info()
    print(df.columns)

    print(df["isFraud"].value_counts())
    print(df["isFlaggedFraud"].value_counts())

    print("Total missing values:", df.isnull().sum().sum())
    print("Number of rows:", df.shape[0])

    print(round((df["isFraud"].value_counts()[1] / df.shape[0]) * 100, 2))

    # Exploratory Data Analysis

    df["type"].value_counts().plot(kind="bar", title="transaction types", color="skyblue")
    plt.xlabel("transaction types")
    plt.ylabel("count")
    plt.savefig("transaction_types.png")
    plt.show()

    fraud_by_type = df.groupby("type")["isFraud"].mean().sort_values(ascending=False)
    print(fraud_by_type)
    fraud_by_type.plot(kind="bar", title="Fraud by transaction type", color="salmon")
    plt.ylabel("fraud rate")
    plt.savefig("fraud_by_transaction_type.png")
    plt.show()

    print(df["amount"].describe().astype(int))

    sns.histplot(np.log1p(df["amount"]), bins=100, kde=True, color="green")
    plt.title("transaction amount distribution(log scaled)")
    plt.xlabel("log(amount + 1)")
    plt.savefig("transaction_amount_distribution.png")
    plt.show()

    sns.boxplot(data=df[df["amount"] < 50000], x="isFraud", y="amount")
    plt.title("amount vs isfraud(filtered under 50k)")
    plt.savefig("amount_vs_isfraud.png")
    plt.show()

    df["balanceDiffOrig"] = df["oldbalanceOrg"] - df["newbalanceOrig"]
    df["balanceDiffDest"] = df["newbalanceDest"] - df["oldbalanceDest"]

    print("Negative balanceDiffOrig count:", (df["balanceDiffOrig"] < 0).sum())
    print("Negative balanceDiffDest count:", (df["balanceDiffDest"] < 0).sum())

    print(df.head(2))

    frauds_per_step = df[df["isFraud"] == 1]["step"].value_counts().sort_index()
    plt.plot(frauds_per_step.index, frauds_per_step.values, label="frauds per step")
    plt.xlabel("step(time)")
    plt.ylabel("number of frauds")
    plt.title("frauds over time")
    plt.grid(True)
    plt.savefig("frauds_over_time.png")
    plt.show()

    df.drop(columns="step", inplace=True)
    print(df.head())

    top_senders = df["nameOrig"].value_counts().head(10)
    print(top_senders)

    top_receivers = df["nameDest"].value_counts().head(10)
    print(top_receivers)

    fraud_users = df[df["isFraud"] == 1]["nameOrig"].value_counts().head(10)
    print(fraud_users)

    fraud_types = df[df["type"].isin(["TRANSFER", "CASH_OUT"])]
    print(fraud_types["type"].value_counts())

    sns.countplot(data=fraud_types, x="type", hue="isFraud")
    plt.title("fraud distribution in transfer & cash_out")
    plt.savefig("fraud_distribution_transfer_cashout.png")
    plt.show()

    corr = df[["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest", "isFraud"]].corr()
    print(corr)

    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("correlation matrix")
    plt.savefig("correlation_matrix.png")
    plt.show()

    zero_after_transfer = df[
        (df["oldbalanceOrg"] > 0) &
        (df["newbalanceOrig"] == 0) &
        (df["type"].isin(["TRANSFER", "CASH_OUT"]))
    ]
    print("Zero-after-transfer rows:", len(zero_after_transfer))
    print(zero_after_transfer.head()) 

    # Data Preprocessing and Model Training

    df = fresh_df
    print(df.head())

    df_model = df.drop(["nameOrig", "nameDest", "isFlaggedFraud"], axis=1)
    cols_to_transform = ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]
    df_model[cols_to_transform] = np.log1p(df_model[cols_to_transform])
    df_model["step_hour"] = df["step"] % 24
    df_model = df_model[df_model["type"].isin(["TRANSFER", "CASH_OUT"])]
    print(df_model.head())

    categorial = ["type"]
    numeric = ["step_hour", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]

    x = df_model.drop(columns=["isFraud", "step"])
    y = df_model["isFraud"]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, stratify=y, random_state=42)

    # Compute the actual class imbalance ratio 
    neg, pos = y_train.value_counts()[0], y_train.value_counts()[1]
    ratio = neg / pos 
    print(f"scale_pos_weight (neg/pos): {ratio:.2f}")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(drop="first"), categorial),
        ],
        remainder="drop",
    )

    pipeline = Pipeline([
        ("prep", preprocessor),
        ("clf", XGBClassifier(
            scale_pos_weight=ratio * 0.85,
            min_child_weight=2,
            max_depth=6,
            learning_rate=0.1,
            n_estimators=500,
            random_state=42,
            verbosity=1,
            eval_metric="logloss"
        )),
    ])

    pipeline.fit(x_train, y_train)
    y_pred = pipeline.predict_proba(x_test)[:, 1] >= 0.947

    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    print(pipeline.score(x_test, y_test) * 100)

    # Save the trained model to a file for later use

    joblib.dump(pipeline, "fraud_detection_pipeline.pkl")
    print("Model saved to fraud_detection_pipeline.pkl")

    
if __name__ == "__main__":
    main()
