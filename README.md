# Car Price Prediction

## Project Overview

This project aims to predict the selling price of a used car using machine learning regression models. It uses car-related features such as present price, manufacturing year, kilometers driven, fuel type, transmission type, and previous owners to estimate the selling price.

This project was completed as part of the CodeAlpha Internship.

## Objectives

* Analyze the dataset to understand the factors affecting car prices.
* Perform Exploratory Data Analysis (EDA) using data visualization.
* Clean and prepare the data for machine learning.
* Apply feature engineering and encode categorical variables.
* Train and evaluate regression models.
* Compare model performance using evaluation metrics.

## Dataset

The project uses the dataset `car data (1).csv`.

The dataset contains information about used cars, including the following features:

* `Car_Name`: Name of the car.
* `Year`: Manufacturing year of the car.
* `Selling_Price`: Selling price of the car; this is the target variable.
* `Present_Price`: Current or listed market price of the car.
* `Driven_kms`: Total kilometers driven.
* `Fuel_Type`: Fuel type of the car.
* `Selling_type`: Whether the seller is a dealer or an individual.
* `Transmission`: Manual or automatic transmission.
* `Owner`: Number of previous owners.

## Technologies and Libraries Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn

## Project Workflow

### 1. Data Loading and Exploration

The dataset is loaded using Pandas. Basic exploration is performed to understand its structure, dimensions, column names, missing values, duplicate records, and statistical summaries.

### 2. Exploratory Data Analysis (EDA)

The following visualizations are created:

* Histograms to examine numerical features.
* Count plots to analyze categorical features.
* A scatter plot to examine the relationship between present price and selling price.
* A correlation heatmap to understand relationships between numerical variables.

### 3. Feature Engineering

A new feature, `Kms_Per_Price`, is created by dividing the kilometers driven by the present price.

This feature is used to explore whether the relationship between car usage and present price helps the model predict the selling price.

### 4. Data Preprocessing

* The target variable, `Selling_Price`, is separated from the input features.
* The dataset is divided into training and testing sets using an 80:20 split.
* Categorical variables are converted into numerical representations using `OneHotEncoder`.
* `ColumnTransformer` is used to apply the encoding while passing the remaining columns through.

### 5. Model Training

Two regression models are trained:

**Linear Regression:** Used as a baseline model to predict car selling prices.

**Random Forest Regressor:** Used to capture more complex relationships between the input features and selling prices.

### 6. Model Evaluation

Both models are evaluated using the following metrics:

* **Mean Absolute Error (MAE):** Measures the average absolute difference between actual and predicted prices.
* **Mean Squared Error (MSE):** Measures the average squared prediction error.
* **Root Mean Squared Error (RMSE):** Measures prediction error in the same units as the target variable.
* **R² Score:** Measures how well the model explains the variation in selling prices on the test dataset.

## Results

In the evaluated run, the models produced the following results:

| Evaluation Metric | Linear Regression | Random Forest Regressor |
| ----------------- | ----------------: | ----------------------: |
| MAE               |            1.0625 |                  0.6426 |
| MSE               |            2.6242 |                  1.1282 |
| RMSE              |            1.6199 |                  1.0622 |
| R² Score          |            0.8861 |                  0.9510 |

The Random Forest Regressor performed better than Linear Regression on this test split, with lower prediction errors and a higher R² score.

*Note: Replace these values if your final code prints different results. The scores describe performance on this test split and do not guarantee the same performance on new data.*

## Conclusion

This project demonstrates how machine learning can be used to predict used car selling prices.

The workflow includes data exploration, visualization, feature engineering, categorical encoding, model training, and evaluation.

Among the two models tested, the Random Forest Regressor achieved better results on the test dataset.

The project provided practical experience in data analysis and regression modelling using Python and Scikit-learn.

## How to Run the Project

1. Install Python on your computer.

2. Install the required libraries:

   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```

3. Place `car data (1).csv` and `car_price_prediction.py` in the same project folder.

4. Open the project folder in VS Code.

5. Run the Python file:

   ```bash
   python car_price_prediction.py
   ```

6. Review the visualizations, predicted prices, and model evaluation metrics.

## Project Structure

```text
codealpha_car_price_prediction/
│
├── car_price_prediction.py
├── car data (1).csv
└── README.md
```

## Internship

**Organization:** CodeAlpha
**Project:** Car Price Prediction
**Task:** Task 3 — Car Price Prediction
