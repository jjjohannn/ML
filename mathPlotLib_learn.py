""" aprendiendo a usar matplotlib"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Data for plotting, example 1
# lista = [1, 2, 3, 4, 5, 6, 7, 8, 9]
# lista2 = [1, 4, 9, 16, 25, 36, 49, 64, 81]
# plt.plot(lista2, lista)
# plt.show()

# Data for plotting, example 2
# x = np.linspace(-2, 2, 100)
# y = x**2
# plt.plot(x, y)
# plt.title("Funcion cuadratica")
# plt.xlabel("x")
# plt.ylabel("y")
# plt.grid(True)
# plt.show()


# Data for plotting, example 3
# x = np.linspace(-2, 2, 100)
# y = x**2
# y2 = x**3
# plt.plot(x, y, x, y2)
# plt.show()

# Data for plotting, example 4
# x = np.linspace(-2, 2, 100)
# y = x**2
# y3 = x+1
# plt.grid()
# plt.plot(x, y, 'b--', label='x**2')
# plt.plot(x, y3, 'g', label='x+1')
# plt.legend(loc='best')
# plt.show()


# Data for plotting, example 5
# numeros primos hasta 100
# def es_primo(n):
#     """Función que determina si un número es primo"""
#     if n < 2:
#         return False
#     for i in range(2, int(np.sqrt(n)) + 1):
#         if n % i == 0:
#             return False
#     return True


# Generar números hasta 100
#

# Filtrar los números primos
# primos = list(filter(es_primo, numeros))
# pares = np.arange(0, 101, 2)

# y = np.arange(len(primos))
# y2 = np.arange(len(pares))
# Graficar como puntos
# plt.subplot(1, 2, 1)
# plt.scatter(primos, y, color='red', label='Primos')
# plt.legend(loc='best')
# plt.subplot(1, 2, 2)
# plt.scatter(pares, y2, color='blue', label='Pares')
# plt.legend(loc='best')
# plt.show()

# Data for plotting, example 6
# fibonacci = [0, 1]
# for i in range(2, 10):
#     fibonacci.append(fibonacci[i-1] + fibonacci[i-2])
# print(fibonacci)
# x = np.arange(len(fibonacci))
# plt.scatter(x, fibonacci, color='green')
# plt.show()
