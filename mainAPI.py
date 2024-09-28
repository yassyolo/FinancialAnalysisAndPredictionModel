from flask import Flask, request, jsonify
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

arima_model = None

@app.route('/forecast', methods=['POST'])
def forecast():
    global arima_model

    input_data = request.json

    df = pd.DataFrame(input_data)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df = df.asfreq('W')

    df['Inflow'] = df['TransactionFees'] + df['AccountMaintenanceFees'] + df['LoanRepayments']
    df['Outflow'] = df['LoanDisbursements']
    df['NetCashFlow'] = df['Inflow'] - df['Outflow']

    train_data = df['NetCashFlow'][:-1]

    arima_model = ARIMA(train_data, order=(1, 1, 1)).fit()

    forecast_steps = 1
    forecast = arima_model.forecast(steps=forecast_steps)
    forecast_dates = pd.date_range(start=df.index[-1] + pd.DateOffset(weeks=1), periods=forecast_steps, freq='W')
    forecast_df = pd.DataFrame({'Date': forecast_dates, 'ForecastedNetCashFlow': forecast})
    forecast_df['Date'] = forecast_df['Date'].dt.strftime('%Y-%m-%d')

    historical_data_output = {str(key): value for key, value in df['NetCashFlow'][:-forecast_steps].to_dict().items()}  # Only include historical data

    result = {
        "HistoricalData": historical_data_output,
        "ForecastData": forecast_df.set_index('Date').rename_axis(None).to_dict(orient='records'),
    }

    return jsonify(result)

@app.route('/customer', methods=['POST'])
def customer():
    input_data = request.json

    df = pd.DataFrame(input_data)

    df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})

    X = df[['Gender', 'TransactionFrequency', 'AverageTransactionAmount', 'AccountBalance']]
    y = df['LoanAmountApproved']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = RandomForestRegressor()

    param_grid = {
        'n_estimators': [10, 50],
        'max_depth': [5, 10],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt', None]
    }

    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    predictions = best_model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)

    gender_analysis = df.groupby('Gender').agg({
        'AverageTransactionAmount': 'mean',
        'TotalTransactionAmount': 'sum',
        'LoanAmountRequested': 'mean',
        'LoanAmountApproved': 'mean',
        'AccountBalance': 'mean'
    }).reset_index()

    response = {
        'GenderAnalysis': []
    }

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
