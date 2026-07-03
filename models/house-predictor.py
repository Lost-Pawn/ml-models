from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing(name="housing", as_frame=True)

X = housing.data
y = housing.target