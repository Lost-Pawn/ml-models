import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor     
from sklearn.metrics import root_mean_squared_error, r2_score

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
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("correlation_heatmap.png")  
plt.show()


# HISTOGRAMS

df.hist(figsize=(15,10), bins=50, color='skyblue', edgecolor='black')
plt.suptitle("Feature Distributions")
plt.tight_layout()
plt.savefig("feature_distributions.png")  
plt.show()


# BOXPLOTS FOR OUTLIERS

plt.figure(figsize=(14,8))
sns.boxplot(data=df)
plt.xticks(rotation=45)
plt.title("Boxplots for Outlier Detection")
plt.tight_layout()
plt.savefig("boxplots_outliers.png")  
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

    sns.scatterplot(x=df[feature], y=df["MedHouseVal"], alpha=0.3)
    plt.title(f"{feature} vs House Price")
    plt.xlabel(feature)
    plt.ylabel("Median House Value")

    plt.tight_layout()
    plt.savefig(f"{feature}_vs_house_price.png")  
    plt.show()


# TRAIN BASE MODEL

X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

baseline_rmse = root_mean_squared_error(y_test, y_pred)
print("\nBaseline RMSE:", baseline_rmse)


# REMOVE EXTREME OUTLIERS

df["Population"] = np.log1p(df["Population"])
df["AveOccup"] = np.log1p(df["AveOccup"])
df["AveRooms"] = np.log1p(df["AveRooms"])
df["AveBedrms"] = np.log1p(df["AveBedrms"])

clean_df = df[df["MedHouseVal"] < 5.0].reset_index(drop=True)

print("\nOriginal dataset size:", len(df))
print("Dataset size after cleaning:", len(clean_df))


# RETRAIN AFTER CLEANING

X = clean_df.drop("MedHouseVal", axis=1)
y = clean_df["MedHouseVal"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(bootstrap=False, max_features=3, n_estimators=300, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

improved_rmse = root_mean_squared_error(y_test, y_pred)

print("R2 Score:", r2_score(y_test, y_pred))
print("\nRMSE After Outlier Removal:", improved_rmse)
print("\nImprovement:", baseline_rmse - improved_rmse)

plt.figure(figsize=(7,5))
sns.lineplot(x=y_test, y=y_pred, alpha=0.3)
plt.title("Predicted vs Actual House Prices")
plt.xlabel("Actual House Prices")
plt.ylabel("Predicted House Prices")
plt.tight_layout()
plt.savefig("predicted_vs_actual.png") 
plt.show()
