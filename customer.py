import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler


# Sample data structure with additional features
data = {
    'Gender': ['Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male'],
    'TransactionFrequency': [2, 3, 2, 1, 3, 2, 3],  # Already numerical
    'AverageTransactionAmount': [150.00, 200.00, 120.00, 300.00, 250.00, 180.00, 190.00],
    'TotalTransactionAmount': [6000.00, 8000.00, 5200.00, 12000.00, 10000.00, 7200.00, 8200.00],
    'LoanAmountRequested': [10000, 15000, 12000, 18000, 20000, 13000, 15000],
    'LoanAmountApproved': [9000, 14000, 11000, 17000, 18000, 12000, 13000],
    'AccountBalance': [5000, 7000, 6000, 12000, 8000, 9000, 9000],
    'AccountAge': [2, 3, 1, 5, 4, 3, 2],  # New feature: Age of account in years
    'MonthlyIncome': [3000, 4000, 2500, 5000, 4500, 3500, 3800],  # New feature: Monthly income
}

df = pd.DataFrame(data)

# Convert Gender to numerical
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})

# Features and target variable
X = df[['Gender', 'TransactionFrequency', 'AverageTransactionAmount', 'AccountBalance', 'AccountAge', 'MonthlyIncome']]
y = df['LoanAmountApproved']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create a pipeline with a Random Forest Regressor
model = RandomForestRegressor()

# Hyperparameter tuning using Grid Search
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', None]
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# Best model
best_model = grid_search.best_estimator_

# Predict and evaluate
predictions = best_model.predict(X_test)
mse = mean_squared_error(y_test, predictions)

print(f'Mean Squared Error: {mse}')
print('Predictions:', predictions)

# Analyze trends based on gender
gender_analysis = df.groupby('Gender').agg({
    'AverageTransactionAmount': 'mean',
    'TotalTransactionAmount': 'sum',
    'LoanAmountRequested': 'mean',
    'LoanAmountApproved': 'mean',
    'AccountBalance': 'mean',
    'AccountAge': 'mean',
    'MonthlyIncome': 'mean'
}).reset_index()

# Adjust pandas display settings to show all columns
pd.set_option('display.max_columns', None)

# Display the gender analysis with formatted outputs
for index, row in gender_analysis.iterrows():
    gender = 'Female' if row['Gender'] == 0 else 'Male'
    print(f"Gender: {row['Gender']} represents {gender}")
    print(f"AverageTransactionAmount:\nFor {gender.lower()}s (Gender = {row['Gender']}), the average transaction amount is ${row['AverageTransactionAmount']:.2f}.")
    print(f"TotalTransactionAmount:\n{gender}s made a total of ${row['TotalTransactionAmount']:.2f} in transactions.")
    print(f"LoanAmountRequested:\nOn average, {gender.lower()}s requested ${row['LoanAmountRequested']:.2f} in loans.")
    print(f"LoanAmountApproved:\nOn average, {gender.lower()}s were approved for ${row['LoanAmountApproved']:.2f} in loans.")
    print(f"AccountBalance:\nThe average account balance for {gender.lower()}s is ${row['AccountBalance']:.2f}.")
    print(f"AccountAge:\nThe average age of accounts for {gender.lower()}s is {row['AccountAge']:.2f} years.")
    print(f"MonthlyIncome:\nThe average monthly income for {gender.lower()}s is ${row['MonthlyIncome']:.2f}.\n")
