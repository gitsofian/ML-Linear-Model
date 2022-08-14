import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

plt.style.use("seaborn-whitegrid")


# Generiere synthetische Daten
X, Y, coef = datasets.make_regression(
    n_samples=1000, n_features=1, n_informative=2, noise=10, coef=True, random_state=0, bias=100
)
Y = Y[:, np.newaxis]

# Array mit Stützstellen zum Plotten der Gerade
linX = np.array([np.min(X), np.max(X)])

# Erstelle Plot
fig, ax = plt.subplots()

# Plotte synthetischen Datensatz
plt.scatter(X, Y, color="cornflowerblue", marker=".")

# Plotte Gerade und speicher Objekt für spätere Änderungen
straight, = plt.plot(linX, 0*linX, color="red", label='Gerade\nErr=')
plt.xlim(linX)

# Aufgabe d) & e) -->

# Führe lineare Regression durch und berechne die Fehlerquadratsumme
reg = LinearRegression().fit(X, Y)
mse = mean_squared_error(Y, reg.predict(X))
print(f'score: {reg.score(X, Y)}')

# Plotte lineare Regression
plt.plot(X, reg.predict(X), color="blue")
# <--

# Plotte Legende
legend = plt.legend(loc="lower right")

# Erstelle Slider für Parameter m und c der Gerade
plt.subplots_adjust(bottom=0.25)
axm = plt.axes([0.19, 0.1, 0.65, 0.03])
axc = plt.axes([0.19, 0.05, 0.65, 0.03])

# Aufgabe c)

slider_m = Slider(axm, 'm', -200.0, 200.0, valinit=0, valstep=0.1)
slider_c = Slider(axc, 'c', -50.0, 200.0, valinit=100, valstep=0.1)

# Definiere Update Funktion für Slider


def update(val):
    # Lese aktuelle Werte aus den Slidern und aktualisiere die Gerade im Plot
    m = slider_m.val
    c = slider_c.val
    straight.set_ydata(m*linX + c)

    # Berechne Fehlerquadratsumme
    err = 0
    for i in range(X.size):
        err = err + (Y[i]-(m*X[i]+c))**2    # Methode der kleinsten Quadrate
    err = float(err/X.size)

    # Aufgabe b) -->
    # <--

    legend.get_texts()[0].set_text(
        f'Gerade\nErr gerechnet: {err:.3f}\n Mean Squared Error (mse): {mse:.3f}\n score: {reg.score(X, Y):.3f}')


# Rufe update Funktion auf, wenn sich der Sliderwert ändert
slider_m.on_changed(update)
slider_c.on_changed(update)
update(None)

# Zeige Plot
plt.show()
