""" Regresion lineal para predecir el costo de un accidente de transito """
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression

# creamos un array de numeros aleatorios enteros
x = 2 * np.random.rand(100, 1)
y = 4 + 3 * x + np.random.randn(100, 1)

print("Longitud del conjunto de datos: ", len(x))


# visualizamos los datos
plt.plot(x, y, "b.")
plt.xlabel("Equipos afectados (u/1000)")
plt.ylabel("Costo del accidente (u/10000)")
plt.show()

# modificamos el conjunto de datos a medidas reales
data = {"n_equipos_afectados": x.flatten(), "costo_accidente": y.flatten()}
df = pd.DataFrame(data)
print(df.head(10))

# escalamos el conjunto de datos
df["n_equipos_afectados"] = df["n_equipos_afectados"]*1000
df["costo_accidente"] = df["costo_accidente"]*10000
df["n_equipos_afectados"] = df["n_equipos_afectados"].astype(int)
df["costo_accidente"] = df["costo_accidente"].astype(int)
print(df.head(10))

# visualizamos los datos
plt.plot(df["n_equipos_afectados"], df["costo_accidente"], "b.")
plt.xlabel("Equipos afectados")
plt.ylabel("Costo del accidente")
plt.show()

# construimos el modelo
lin_reg = LinearRegression()
lin_reg.fit(df["n_equipos_afectados"].values.reshape(-1, 1),
            df["costo_accidente"].values)

# parametros del modelo
print("Pendiente: ", lin_reg.coef_)
print("Ordenada al origen: ", lin_reg.intercept_)

# prediccion para el valor min y max del conjunto de datos de entrada para visualizar la recta
x_min_max = np.array([[df["n_equipos_afectados"].min()],
                     [df["n_equipos_afectados"].max()]])
y_train_pred = lin_reg.predict(x_min_max)

# visualizamos los datos de la prediccion
plt.plot(x_min_max, y_train_pred, "r-")
plt.plot(df["n_equipos_afectados"], df["costo_accidente"], "b.")
plt.xlabel("Equipos afectados")
plt.ylabel("Costo del accidente")
plt.show()

# prediccion para un valor de entrada
x_new = np.array([[500]])
y_new = lin_reg.predict(x_new)
print("Prediccion: ", y_new)

plt.plot(x_new, y_new, "rx")
plt.plot(df["n_equipos_afectados"], df["costo_accidente"], "b.")
plt.plot(x_min_max, y_train_pred, "g-")
plt.xlabel("Equipos afectados")
plt.ylabel("Costo del accidente")
plt.show()
