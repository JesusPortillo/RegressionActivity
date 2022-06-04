import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

data = pd.read_excel("AirQualityUCI.xlsx")
x = data["RH"]
y = data["T"]
#preparing data

train_x = x[:8000]
train_y = y[:8000]

test_x = x[8000:]
test_y = y[8000:]


#polinomial regression
mymodel = np.poly1d(np.polyfit(train_x, train_y, 3))
myline = np.linspace(100, 1000, 8000)
plt.scatter(train_x,train_y)
plt.plot(train_x, mymodel(myline))
plt.show()
r2 = r2_score(train_y, mymodel(train_x))

print(r2)