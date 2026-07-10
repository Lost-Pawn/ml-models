
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib

warnings.filterwarnings("ignore")
sns.set(style="whitegrid")

DATA_PATH = "fraud_data.csv"


def main():
    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    df = pd.read_csv(DATA_PATH)
    print(df.head())

    df.info()

    print(df.columns)

    print(df["isFraud"].value_counts())

    print(df["isFlaggedFraud"].value_counts())

    print("Total missing values:", df.isnull().sum().sum())

    print("Number of rows:", df.shape[0])

    print(round((df["isFraud"].value_counts()[1] / df.shape[1]) * 100, 2))

    # ------------------------------------------------------------------
    # 2. Exploratory analysis
    # ------------------------------------------------------------------
    df["type"].value_counts().plot(kind="bar", title="transaction types", color="skyblue")
    plt.xlabel("transaction types")
    plt.ylabel("count")
    plt.show()

    fraud_by_type = df.groupby("type")["isFraud"].mean().sort_values(ascending=False)
    print(fraud_by_type)
    fraud_by_type.plot(kind="bar", title="Fraud by transaction type", color="salmon")
    plt.ylabel("fraud rate")
    plt.show()

    print(df["amount"].describe().astype(int))

    sns.histplot(np.log1p(df["amount"]), bins=100, kde=True, color="green")
    plt.title("transaction amount distribution(log scaled)")
    plt.xlabel("log(amount + 1)")
    plt.show()

    sns.boxplot(data=df[df["amount"] < 50000], x="isFraud", y="amount")
    plt.title("amount vs isfraud(filtered under 50k)")
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
    plt.show()

    corr = df[["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest",
               "newbalanceDest", "isFraud"]].corr()
    print(corr)

    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("correlation matrix")
    plt.show()

    zero_after_transfer = df[
        (df["oldbalanceOrg"] > 0) &
        (df["newbalanceOrig"] == 0) &
        (df["type"].isin(["TRANSFER", "CASH_OUT"]))
    ]
    print("Zero-after-transfer rows:", len(zero_after_transfer))
    print(zero_after_transfer.head())

    # ------------------------------------------------------------------
    # 3. Model training
    # ------------------------------------------------------------------
    df = pd.read_csv(DATA_PATH)
    print(df.head())

    df_model = df.drop(["nameOrig", "nameDest", "isFlaggedFraud"], axis=1)
    print(df_model.head())

    categorial = ["type"]
    numeric = ["amount", "oldbalanceOrg", "newbalanceOrig", "newbalanceDest"]

    y = df_model["isFraud"]
    x = df_model.drop("isFraud", axis=1)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, stratify=y
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(drop="first"), categorial),
        ],
        remainder="drop",
    )

    pipeline = Pipeline([
        ("prep", preprocessor),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000)),
    ])

    pipeline.fit(x_train, y_train)

    y_pred = pipeline.predict(x_test)

    print(classification_report(y_test, y_pred))

    print(confusion_matrix(y_test, y_pred))

    print(pipeline.score(x_test, y_test) * 100)

    # ------------------------------------------------------------------
    # 4. Save trained model
    # ------------------------------------------------------------------
    joblib.dump(pipeline, "fraud_detection_pipeline.pkl")
    print("Model saved to fraud_detection_pipeline.pkl")


if __name__ == "__main__":
    main()
