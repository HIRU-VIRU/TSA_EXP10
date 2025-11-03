# Exp. No: 10 IMPLEMENTATION OF SARIMA MODEL  
**Date:** 18/10/2025  

---

## **AIM:**  
To implement the SARIMA (Seasonal AutoRegressive Integrated Moving Average) model using Python for time series forecasting.

---

## **ALGORITHM:**
1. Explore the dataset.  
2. Check for stationarity of the time series using the Augmented Dickey-Fuller test.  
3. Determine SARIMA model parameters (p, d, q) and seasonal parameters (P, D, Q, s) using ACF and PACF plots.  
4. Fit the SARIMA model to the training data.  
5. Make time series predictions and perform model auto-fitting.  
6. Evaluate model predictions using RMSE.  
7. Visualize actual vs predicted values.

---

## **PROGRAM:**

```python
#Name:Hiruthik Sudhakar
#Reg No: 212223240054
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

# Load TSLA dataset
data = pd.read_csv('/content/TSLA.csv')

# Convert Date to datetime and set as index
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Plot Close price
plt.plot(data.index, data['Close'])
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title('TSLA Close Price Time Series')
plt.show()

# Function to check stationarity
def check_stationarity(timeseries):
    result = adfuller(timeseries)
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'\t{key}: {value}')

# Check stationarity of Close price
check_stationarity(data['Close'])

# Plot ACF and PACF
plot_acf(data['Close'])
plt.show()

plot_pacf(data['Close'])
plt.show()

# Train-test split
train_size = int(len(data) * 0.8)
train, test = data['Close'][:train_size], data['Close'][train_size:]

# SARIMA model
sarima_model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
sarima_result = sarima_model.fit()

# Forecast
predictions = sarima_result.predict(start=len(train), end=len(train) + len(test) - 1, dynamic=False)

# RMSE
mse = mean_squared_error(test, predictions)
rmse = np.sqrt(mse)
print('RMSE:', rmse)

# Plot actual vs predicted
plt.plot(test.index, test, label='Actual')
plt.plot(test.index, predictions, label='Predicted', color='red')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title('SARIMA Model Predictions on TSLA Close Price')
plt.legend()
plt.show()
```

## Output:
<img width="850" height="470" alt="download" src="https://github.com/user-attachments/assets/6afeebe0-8091-4a7f-9054-51728c6284ae" />
<img width="568" height="435" alt="download" src="https://github.com/user-attachments/assets/40bf8c66-2707-411e-9056-1a3c92102076" />
<img width="568" height="435" alt="download" src="https://github.com/user-attachments/assets/6e7dd266-6b00-4861-a7e0-05a02ec60a9b" />
<img width="870" height="470" alt="download" src="https://github.com/user-attachments/assets/ab633fdc-840c-4e16-ac8d-f6fdf58b011d" />

## Result

Thus, the SARIMA model was successfully implemented using the AirPassengers dataset.
