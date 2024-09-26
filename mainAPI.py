from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Global ARIMA model variable
arima_model = None

@app.route('/forecast', methods=['POST'])
def forecast():
    global arima_model  # Use the global ARIMA model

    # Get the data from the request
    input_data = request.json

    # Convert input data to DataFrame
    df = pd.DataFrame(input_data)

    # Ensure the 'Date' column is in datetime format and set it as index
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    # Set the frequency of the date index to weekly
    df = df.asfreq('W')

    # Calculate total cash inflow and outflow
    df['Inflow'] = df['TransactionFees'] + df['AccountMaintenanceFees'] + df['LoanRepayments']
    df['Outflow'] = df['LoanDisbursements']
    df['NetCashFlow'] = df['Inflow'] - df['Outflow']

    # Prepare data for training the ARIMA model
    train_data = df['NetCashFlow'][:-1]  # Use all but the last 2 weeks for training

    # Always fit the ARIMA model on the latest data
    arima_model = ARIMA(train_data, order=(1, 1, 1)).fit()

    # Forecast future net cash flows
    forecast_steps = 1  # Forecast for the next week
    forecast = arima_model.forecast(steps=forecast_steps)

    # Create DataFrame for forecasted results
    forecast_dates = pd.date_range(start=df.index[-1] + pd.DateOffset(weeks=1), periods=forecast_steps, freq='W')
    forecast_df = pd.DataFrame({'Date': forecast_dates, 'ForecastedNetCashFlow': forecast})

    # Convert forecast dates to strings
    forecast_df['Date'] = forecast_df['Date'].dt.strftime('%Y-%m-%d')

    # Prepare the response with only historical data input
    historical_data_output = {str(key): value for key, value in df['NetCashFlow'][:-forecast_steps].to_dict().items()}  # Only include historical data

    result = {
        "HistoricalData": historical_data_output,  # Historical data input
        "ForecastData": forecast_df.set_index('Date').rename_axis(None).to_dict(orient='records'),  # Forecast
    }

    return jsonify(result)

@app.route('/customer', methods=['POST'])
def customer():
    # Get the data from the request
    input_data = request.json

    # Convert input data to DataFrame
    df = pd.DataFrame(input_data)

    # Convert Gender to numerical
    df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})

    # Features and target variable
    X = df[['Gender', 'TransactionFrequency', 'AverageTransactionAmount', 'AccountBalance']]
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
        'max_depth': [10, 20],
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

    # Analyze trends based on gender
    gender_analysis = df.groupby('Gender').agg({
        'AverageTransactionAmount': 'mean',
        'TotalTransactionAmount': 'sum',
        'LoanAmountRequested': 'mean',
        'LoanAmountApproved': 'mean',
        'AccountBalance': 'mean'
    }).reset_index()

    # Prepare JSON response
    response = {
        'GenderAnalysis': []
    }

    # Append gender analysis to response
    for index, row in gender_analysis.iterrows():
        gender = 'Female' if row['Gender'] == 0 else 'Male'
        gender_info = {
            'Gender': gender,
            'AverageTransactionAmount': row['AverageTransactionAmount'],
            'TotalTransactionAmount': row['TotalTransactionAmount'],
            'LoanAmountRequested': row['LoanAmountRequested'],
            'LoanAmountApproved': row['LoanAmountApproved'],
            'AccountBalance': row['AccountBalance']
        }
        response['GenderAnalysis'].append(gender_info)

    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True)
