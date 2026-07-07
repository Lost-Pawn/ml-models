import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error

# LOAD DATASET

housing = fetch_california_housing()

df = pd.DataFrame(
    housing.data,
    columns=housing.feature_names
)

df["MedHouseVal"] = housing.target

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nStatistical Summary:")
print(df.describe())


# FEATURE CORRELATION

plt.figure(figsize=(10,8))
sns.heatmap(
    df.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("correlation_heatmap.png")  # Save the heatmap as an image
plt.show()


# HISTOGRAMS

df.hist(
    figsize=(15,10),
    bins=30
)

plt.suptitle("Feature Distributions")
plt.tight_layout()
plt.savefig("feature_distributions.png")  # Save the histograms as an image
plt.show()


# BOXPLOTS FOR OUTLIERS

plt.figure(figsize=(14,8))

sns.boxplot(data=df)

plt.xticks(rotation=45)
plt.title("Boxplots for Outlier Detection")

plt.tight_layout()
plt.savefig("boxplots_outliers.png")  # Save the boxplots as an image
plt.show()


# IMPORTANT FEATURES VS HOUSE PRICE

important_features = [
    "MedInc",
    "AveRooms",
    "HouseAge",
    "Latitude",
    "Longitude"
]

for feature in important_features:
    plt.figure(figsize=(7,5))

    sns.scatterplot(
        x=df[feature],
        y=df["MedHouseVal"],
        alpha=0.3
    )

    plt.title(f"{feature} vs House Price")
    plt.xlabel(feature)
    plt.ylabel("Median House Value")

    plt.tight_layout()
    plt.savefig(f"{feature}_vs_house_price.png")  # Save the scatter plot as an image
    plt.show()


# TRAIN BASE MODEL

X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LinearRegression()

model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

baseline_rmse = root_mean_squared_error(
    y_test,
    y_pred
)

print("\nBaseline RMSE:", baseline_rmse)

# REMOVE EXTREME OUTLIERS

clean_df = df[
    (df["AveOccup"] < 5) &
    (df["AveRooms"] < 8) &
    (df["AveBedrms"] < 4) &
    (df["HouseAge"] < 40) &
    (df["Population"] < 8000) &
    (df["MedHouseVal"] < 3)
]

print("\nOriginal dataset size:", len(df))
print("Dataset size after cleaning:", len(clean_df))



# RETRAIN AFTER CLEANING

X = clean_df.drop("MedHouseVal", axis=1)
y = clean_df["MedHouseVal"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LinearRegression()

model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

improved_rmse = root_mean_squared_error(
    y_test,
    y_pred
)

print("\nRMSE After Outlier Removal:", improved_rmse)

print("\nImprovement:", baseline_rmse - improved_rmse)
