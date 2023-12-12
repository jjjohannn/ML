""" aprendiendo a usar numpy"""
import numpy as np


# array de una dimencion
a = np.array([1, 2, 3])
print("Array de una dimension: ", a)
print("Tipo de dato: ", a.dtype)
print("Dimension: ", a.ndim)
print("Forma: ", a.shape)
print("Tamaño: ", a.size)

# array de dos dimenciones
b = np.array([[1, 2, 3], [5, 4, 6]])
print("Array de dos dimensiones: ",  b)
print("Tipo de dato: ", b.dtype)
print("Dimension: ", b.ndim)
print("Forma: ", b.shape)
print("Tamaño: ", b.size)

# array de tres dimenciones
c = np.array([[[1, 2, 3], [5, 4, 6]], [[1, 2, 3], [5, 4, 6]]])
print("Array de tres dimensiones: ",  c)
print("Tipo de dato: ", c.dtype)
print("Dimension: ", c.ndim)
print("Forma: ", c.shape)
print("Tamaño: ", c.size)

# copiamos el array b en d y editamos el array d
d = b
d[0, 0] = 0
print("Array d: ", d)
print("Array b: ", b)

# copiamos el array b en e y editamos el array e sin que cambie b
e = b.copy()
e[0, 0] = 1
print("Array e: ", e)
print("Array b: ", b)

# creamos un array de ceros
f = np.zeros((3, 4))
print("Array de ceros: ", f)

# creamos un array de unos
g = np.ones((3, 4))
print("Array de unos: ", g)

# creamos un array de numeros aleatorios
h = np.random.random((3, 4))
print("Array de numeros aleatorios: ", h)

# creamos un array de numeros aleatorios enteros
i = np.random.randint(1, 10, (3, 4))
print("Array de numeros aleatorios enteros: ", i)

# cambiamos la forma de un array g de 3x4 a 4x3
j = i.reshape((4, 3))
print("Array j: ", j)
