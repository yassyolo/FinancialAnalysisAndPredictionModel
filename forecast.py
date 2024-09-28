import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

data = {
    'Date': pd.date_range(start='2023-01-01', periods=6, freq='W'),
    'TransactionFees': [5000, 3000, 4000, 2000, 1500, 2500],
    'AccountMaintenanceFees': [1000, 2000, 2000, 1500, 6000, 4000] ,
    'LoanDisbursements': [400000, 80000, 0, 0, 2000, 0],
    'LoanRepayments': [500000, 2000, 3000, 1000, 0, 5000]
}

df = pd.DataFrame(data)
df.set_index('Date', inplace=True)

df['Inflow'] = df['TransactionFees'] + df['AccountMaintenanceFees'] + df['LoanRepayments']
df['Outflow'] = df['LoanDisbursements']
df['NetCashFlow'] = df['Inflow'] - df['Outflow']

train_data = df['NetCashFlow'][:-2]
model = ARIMA(train_data, order=(1, 1, 1))
arima_model = model.fit()

forecast_steps = 1
forecast = arima_model.forecast(steps=forecast_steps)

forecast_dates = pd.date_range(start=df.index[-1] + pd.DateOffset(weeks=1), periods=forecast_steps, freq='W')
forecast_df = pd.DataFrame({'Date': forecast_dates, 'ForecastedNetCashFlow': forecast})

combined_df = pd.concat([df['NetCashFlow'], forecast_df.set_index('Date')])
import joblib

joblib.dump(arima_model, 'arima_model.pkl')

result = {
    "HistoricalData": df['NetCashFlow'].to_dict(),
    "ForecastData": forecast_df.set_index('Date').to_dict(orient='records'),
}

print(result)