import numpy as np
import matplotlib.pyplot as plt

from data import get_data, inspect_data, split_data

data = get_data()
inspect_data(data)

train_data, test_data = split_data(data)

# Simple Linear Regression
# predict MPG (y, dependent variable) using Weight (x, independent variable) using closed-form solution
# y = theta_0 + theta_1 * x - we want to find theta_0 and theta_1 parameters that minimize the prediction error

# We can calculate the error using MSE metric:
# MSE = SUM (from i=1 to n) (actual_output - predicted_output) ** 2

# get the columns
# podział danych na treningowe i testowe
y_train = train_data['MPG'].to_numpy()
x_train = train_data['Weight'].to_numpy()

y_test = test_data['MPG'].to_numpy()
x_test = test_data['Weight'].to_numpy()

# TODO: calculate closed-form solution
theta_best = [0, 0]


def dodajKolumneJedynek(x_kol):
    # do wektora (1 kolumna) x_train dodać kolumnę jedynek]
    # zamienianie x_train wierszowego na kolumnowy
    x_train_col = x_kol.reshape(-1, 1)
    # wymiar macierzy
    jedynki = np.ones((x_train_col.shape[0], 1))
    # połączenie jedynek i wektora
    X = np.hstack((jedynki, x_train_col))
    return X


X = dodajKolumneJedynek(x_train)

# do wzoru 1.13
# @ - mnożenie macierzowe
# linalg.inv <- ^(-1)
theta_best = np.linalg.inv(X.T@X)@X.T@y_train


# TODO: calculate error
# funkcja kosztu - różnica między rzeczywistymi wartosciami a wyliczonymi (nasz blad)
# MSE (theta) = 1/m * suma(od i=1 do m)[predykcja modelu i-tego wektora cech - rzeczywista wartosc cechy wyjsciowej i-tego wektora cech]

def MSE(X_MSE, y, theta):
    suma = 0
    # ilosc wierszy X_MSE .shape[0]
    for i in range(X_MSE.shape[0]):
        # dot zwraca skalar
        # X[i, :] zwracanie i-tego wiersza
        suma += (np.dot(theta, X_MSE[i, :]) - y[i])**2

    return suma/X_MSE.shape[0]


print("FUNKCJA KOSZTU")
print(MSE(X, y_train, theta_best))
print(MSE(dodajKolumneJedynek(x_test), y_test, theta_best))

# plot the regression line
x = np.linspace(min(x_test), max(x_test), 100)
y = float(theta_best[0]) + float(theta_best[1]) * x
plt.plot(x, y)
plt.scatter(x_test, y_test)
plt.xlabel('Weight')
plt.ylabel('MPG')
plt.show()

# TODO: standardization
# wzorek 1.15
zx_train = (x_train-x_train.mean())/x_train.std()
zy_train = (y_train-y_train.mean())/y_train.std()
zx_test = (x_test-x_train.mean())/x_train.std()
zy_test = (y_test-y_train.mean())/y_train.std()

print()
print("STANDARYZACJA")
# print(zx_train)

# TODO: calculate theta using Batch Gradient
zx_train_jeden = dodajKolumneJedynek(zx_train)

theta_best = [0, 0]
before = MSE(zx_train_jeden, zy_train, theta_best)
learn_rate = 0.1
theta_new = [0, 0]


# wzorek 1.7
def gradient(X_t, y, theta):
    wynik = X_t.T@(X_t@theta - y) * 2
    return wynik/X_t.shape[0]


for i in range(1000):
    theta_new = theta_best-learn_rate * gradient(zx_train_jeden, zy_train, theta_best)
    current = MSE(zx_train_jeden, zy_train, theta_new)
    if before > current:
        before = current
        theta_best = theta_new
    else:
        break

print("GRADIENT PROSTY")
print(theta_best)

# TODO: calculate error
print("FUNKCJA KOSZTU v2")
print(MSE(dodajKolumneJedynek(zx_test), zy_test, theta_best))

# plot the regression line
x = np.linspace(min(zx_test), max(zx_test), 100)
y = float(theta_best[0]) + float(theta_best[1]) * x
plt.plot(x, y)
plt.scatter(zx_test, zy_test)
plt.xlabel('Weight')
plt.ylabel('MPG')
plt.show()
