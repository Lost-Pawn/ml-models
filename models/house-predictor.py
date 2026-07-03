import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error

housing = fetch_california_housing()

X = housing.data
y = housing.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LinearRegression()

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Root Mean Squared Error:", root_mean_squared_error(y_test, y_pred))

df = pd.DataFrame(
    housing.data, 
    columns=housing.feature_names
)

df['MedHouseVal'] = housing.target

plt.figure(figsize=(10, 8))
sns.heatmap(df.corr()[['MedHouseVal']], cmap='coolwarm', annot=True, fmt=".2f")
plt.tight_layout()

plt.savefig("graph.png", dpi=150)
plt.show()


