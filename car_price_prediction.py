import pandas as pd
df= pd.read_csv("car data (1).csv")

print(df.head())
print(df.shape)
print(df.columns)
print(df.info())
print(df.isnull().sum())
print("Duplicate rows:",df.duplicated().sum())
print(df.describe())

import matplotlib.pyplot as plt
import seaborn as sns


# Create histogram for numerical data
plt.figure(figsize=(16, 8))

# 1. Year
plt.subplot(2, 4, 1)

plt.hist(df["Year"], bins=10, edgecolor="black", color="blue")

plt.xlabel("Year")
plt.ylabel("Frequency")
plt.title("Distribution of Car Year")


# 2. Selling Price
plt.subplot(2, 4, 2)

plt.hist(df["Selling_Price"], bins=10, edgecolor="black", color="blue")

plt.xlabel("Selling Price")
plt.ylabel("Frequency")
plt.title("Distribution of Selling Price")


# 3. Present Price
plt.subplot(2, 4, 3)

plt.hist(df["Present_Price"], bins=10, edgecolor="black", color="blue")

plt.xlabel("Present Price")
plt.ylabel("Frequency")
plt.title("Distribution of Present Price")


# 4. Kilometers Driven
plt.subplot(2, 4, 4)

plt.hist(df["Driven_kms"], bins=10, edgecolor="black", color="blue")

plt.xlabel(" Driven kilometers")
plt.ylabel("Frequency")
plt.title("Distribution of Kilometers Driven")


# 5. Owner
plt.subplot(2, 4, 5)

plt.hist(df["Owner"], bins=5, edgecolor="black", color="blue")

plt.xlabel("Number of Previous Owners")
plt.ylabel("Frequency")
plt.title("Distribution of Previous Owners")


plt.tight_layout()
plt.show()
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
sns.countplot(x=df["Fuel_Type"])
plt.xlabel("Fuel Type")
plt.ylabel("Number of Cars")
plt.title("Distribution of Fuel Type")

plt.subplot(1, 3, 2)
sns.countplot(x=df["Selling_type"])
plt.xlabel("Selling Type")
plt.ylabel("Number of Cars")
plt.title("Distribution of Selling Type")

plt.subplot(1, 3, 3)
sns.countplot(x=df["Transmission"])
plt.xlabel("Transmission")
plt.ylabel("Number of Cars")
plt.title("Distribution of Transmission")

plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))

plt.scatter(df["Present_Price"], df["Selling_Price"])

plt.xlabel("Present Price")
plt.ylabel("Selling Price")
plt.title("Present Price vs Selling Price")

plt.show()

#for heatmap
plt.figure(figsize=(8, 6))

correlation = df.corr(numeric_only=True)

sns.heatmap(correlation, annot=True, cmap="coolwarm")

plt.title("Correlation Heatmap")
plt.show()



df["Kms_Per_Price"] = df["Driven_kms"] / df["Present_Price"]

print(df[["Driven_kms", "Present_Price", "Kms_Per_Price"]].head())


X = df.drop("Selling_Price", axis=1)
y = df["Selling_Price"]

print("Features:")
print(X.head())

print("\nTarget:")
print(y.head())

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)

#update the processing
categorical_columns = [
    "Car_Name",
    "Fuel_Type",
    "Selling_type",
    "Transmission"
]
numerical_columns = [
    "Present_Price",
    "Driven_kms",
    "Owner",
    "Kms_Per_Price"
]
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_columns)
    ],
    remainder="passthrough"
)

X_train_processed = preprocessor.fit_transform(X_train)

X_test_processed = preprocessor.transform(X_test)

print("Training data after preprocessing:", X_train_processed.shape)
print("Testing data after preprocessing:", X_test_processed.shape)

#train the regression model
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train_processed, y_train)
y_pred = model.predict(X_test_processed)
print("Actual Prices:")
print(y_test.head())

print("\nPredicted Prices:")
print(y_pred[:5])

#now evaluate the next regression model
#MAE,MSE ,RMSE,R^2
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("R² Score:", r2)

#train the random forest model
from sklearn.ensemble import RandomForestRegressor
random_forest_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

random_forest_model.fit(X_train_processed, y_train)

random_forest_predictions = random_forest_model.predict(X_test_processed)

print("Actual Prices:")
print(y_test.head())

print("\nRandom Forest Predicted Prices:")
print(random_forest_predictions[:5])

rf_mae = mean_absolute_error(y_test, random_forest_predictions)

rf_mse = mean_squared_error(y_test, random_forest_predictions)

rf_rmse = np.sqrt(rf_mse)

rf_r2 = r2_score(y_test, random_forest_predictions)

print("Random Forest MAE:", rf_mae)
print("Random Forest MSE:", rf_mse)
print("Random Forest RMSE:", rf_rmse)
print("Random Forest R² Score:", rf_r2)
