import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
data = pd.read_excel("AirQualityUCI.xlsx")
x = data["RH"]
y = data["T"]
#preparing data
x,y = np.array(x).reshape(-1,1), np.array(y)
train_x = x[:8000]
train_y = y[:8000]

test_x = x[8000:]
test_y = y[8000:]
#linear regression
model = LinearRegression().fit(train_x,train_y)
r_sq_train = model.score(train_x,train_y)
r_sq_test = model.score(test_x,test_y)
y_predict = model.predict(test_x)
plt.scatter(test_x,test_y)
plt.plot(test_x, y_predict)
plt.show()
print("score for train", r_sq_train)
print("Score for test", r_sq_test)
