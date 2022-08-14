from cProfile import label
from turtle import color
from sklearn import linear_model
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression


# Lade Datensatz aus Datei
# https://archive.ics.uci.edu/ml/datasets/Real+estate+valuation+data+set
file_location = os.path.dirname(os.path.realpath(
    __file__)) + "/data/Real_estate_valuation_data_set.xlsx"
df = pd.read_excel(file_location)

# Erstelle 1D Arrays für die einzelnen Parameter
date = df.drop(["No",
                "X2 house age",
                "X3 distance to the nearest MRT station",
                "X4 number of convenience stores",
                "X5 latitude",
                "X6 longitude",
                "Y house price of unit area"], axis=1)

age = df.drop(["No",
               "X1 transaction date",
               "X3 distance to the nearest MRT station",
               "X4 number of convenience stores",
               "X5 latitude",
               "X6 longitude",
               "Y house price of unit area"], axis=1)

dist = df.drop(["No",
                "X1 transaction date",
                "X2 house age",
                "X4 number of convenience stores",
                "X5 latitude",
                "X6 longitude",
                "Y house price of unit area"], axis=1)

num_conv = df.drop(["No",
                    "X1 transaction date",
                    "X2 house age",
                    "X3 distance to the nearest MRT station",
                    "X5 latitude",
                    "X6 longitude",
                    "Y house price of unit area"], axis=1)
all = df.drop(["No", "Y house price of unit area"], axis=1)

# Target
price = df.drop(["No", "X1 transaction date", "X2 house age", "X3 distance to the nearest MRT station",
                "X4 number of convenience stores", "X5 latitude", "X6 longitude"], axis=1)

# Wähle input/output and label
# ...
a = 4

match a:
    case 1:
        X = dist.values
        label = "Distance values"
    case 2:
        X = date.values
        label = "Date values"
    case 3:
        X = age.values
        label = "Age values"
    case 4:
        X = num_conv.values
        label = "Date values"
    case 5:
        X = all.values
        label = "All values"

Y = price

X_plot = np.array([[np.min(X)], [np.max(X)]])

# Erstelle Plot
fig, ax = plt.subplots()

# Plotte synthetischen Datensatz
plt.scatter(X, Y, color="blue", marker=".", label=label)
# plt.xlim(X_plot)

# Führe lineare Regression durch und berechne die Fehlerquadratsumme
reg = LinearRegression().fit(X, Y)
mse = mean_squared_error(Y, reg.predict(X))
mse_mean = mean_squared_error(Y, np.ones(Y.size)*float(Y.mean()))
print(mse_mean)

# plot & legend updated
plt.plot(X, reg.predict(X), color="black",
         label=f'Linear Regression\n mean squared error : {mse:.2f}\n score: {reg.score(X, Y):.4f}')

# Plotte Legende
legend = plt.legend(loc="upper center")

# legend updated
# legend.get_texts()[1].set_text("mse: " + str(int(mse)))

plt.show()
