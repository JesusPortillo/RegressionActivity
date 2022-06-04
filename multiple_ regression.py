import numpy as np
import pandas as pd
from sklearn import linear_model

data = pd.read_csv("cars.csv")
condition_list = [
    (data["Car"] == "Toyoty"),
    (data["Car"] == "Mitsubishi"),
    (data["Car"] == "Skoda"),
    (data["Car"] == "Fiat"),
    (data["Car"] == "Mini"),
    (data["Car"] == "VW"),
    (data["Car"] == "Mercedes"),
    (data["Car"] == "Ford"),
    (data["Car"] == "Audi"),
    (data["Car"] == "Hyundai"),
    (data["Car"] == "Suzuki"),
    (data["Car"] == "Honda"),
    (data["Car"] == "Hundai"),
    (data["Car"] == "Opel"),
    (data["Car"] == "BMW"),
    (data["Car"] == "Mazda"),
    (data["Car"] == "Volvo")
]
choice_list = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
data['brand_car_normalized'] = np.select(condition_list, choice_list, default="not_specified")
new_df = pd.DataFrame()
new_df["brand"] = data["Car"].drop_duplicates()
new_df["brand_car_normalized"] = choice_list
x = data[['Volume', 'Weight', 'CO2']]
y = data["brand_car_normalized"]
x,y = np.array(x), np.array(y)
regr = linear_model.LinearRegression().fit(x,y)
print(regr.coef_)
y_predicted = regr.predict([[1000, 790, 99]])
car_brand=int(np.round(y_predicted,decimals = 0))
name = new_df[new_df["brand_car_normalized"].isin([car_brand])]
print(data)
print("Es probable que la marca sea: ",name["brand"].values[0])