Flask application that contains models for predicting cashflow and analysis based on gender for a bank system. It is deployed on Azure as app service and used by my project InfiniaBankSystem.

/forecast => predicts cashflow for the next week, based on cashflow from 6 weeks back. Cashflow is accumulated from loan disburements, loan repayments, collection of transaction fees and monthly account fees.

/customer => analysis of financial metrics based on the gender of the customers in the system. Data, collected for each gender: transaction frequency, average transaction amount, total transaction amount, requetsed and approved loan amount and account balance
