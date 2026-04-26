import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# sample dataset
data = {
    "area": [1000, 1500, 2000, 2500],
    "price": [50, 75, 100, 130]
}

df = pd.DataFrame(data)

X = df[["area"]]
y = df["price"]

model = LinearRegression()
model.fit(X, y)

# save model
pickle.dump(model, open("model.pkl", "wb"))