import json
import matplotlib.pyplot as plt
import requests
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from config import api_key
from pprint import pprint
url = "http://api.openweathermap.org/data/2.5/weather"

def extract_data(data):
    return {
        "name": data["name"],
        "lat": data["coord"]["lat"],
        "lon": data["coord"]["lon"],
        "temp_c": data["main"]["temp"]
    }
 
cities = ["Paris", "London", "Oslo", "Beijing", "Mumbai", "Manila", "New York", "Seattle", "Dallas", "Taipei"]

dataset = []

for city in cities:
    data = extract_data(requests.get(url, params={
        "q": city,
        "units": "metric",
        "appid": api_key
    }).json())

    dataset.append(data) 

df = pd.DataFrame(dataset)
pprint(df)

X = df["lat"].to_frame().values
pprint(X)

y = df["temp_c"].values
pprint(y)

model = LinearRegression()
model.fit(X, y)
intercept = model.intercept_
pprint(intercept)
slope = model.coef_[0]
pprint(slope)
r_squared = model.score(X, y)
pprint(r_squared)

formula = f"y = {round(intercept, 3)}+{round(slope, 3)}x"
pprint(formula)

r_squared_string = f"r^2 = {round(r_squared, 3)}"
pprint(r_squared_string)


plt.scatter(df["lat"], df["temp_c"])
predictions = model.predict(X)

plt.title("Effect of Lat on Temp (C)")
plt.xlabel("Lat")
plt.ylabel("Temp_C")
plt.plot(X, predictions, color ="red")
plt.text(20, 5, formula, color="red")
plt.text(20, 3, r_squared_string, color="red")
plt.show()


