""" aprendiendo a usar pandas"""
import pandas as pd

# creamos un diccionario con nombres y edades
diccionarioS = {"Ana": 20, "Jose": 21, "Maria": 19, "Pedro": 18, "Luis": 20}
serie = pd.Series(diccionarioS)
print("Serie:\n", serie)

# accedemos a los elementos 2 y 4 de la serie mediante la posicion
print("Elemento 2: ", serie.iloc[1])
print("Elemento 4: ", serie.iloc[3])

# accedemos a los elementos 2 y 4 de la serie mediante el indice
print("Elemento 2: ", serie.loc["Jose"])
print("Elemento 4: ", serie.loc["Pedro"])

# accedemos a los elementos del 2 al 4 de la serie mediante la posicion
print("Elementos del 2 al 4:\n ", serie.iloc[1:4])

# copiamos la serie en serie2 y editamos la serie2
serie2 = serie.copy()
serie2 = pd.Series(diccionarioS, index=["Ana", "Jose"])
print("Serie2:\n", serie2)

# creamos un dataframe inicializando un diccionario de objetos tipo series
diccionario = {"Edad": pd.Series([20, 21, 19, 18, 20], ["Ana", "Jose", "Maria", "Pedro", "Luis"]),
               "Peso": pd.Series({"Ana": 50, "Jose": 60, "Maria": 55, "Pedro": 70, "Luis": 65}),
               "Hijos": pd.Series([1, 2], ["Ana", "Jose"])}
dataframe = pd.DataFrame(diccionario)
print("Dataframe:\n", dataframe)

# creamos un dataframe solo con las columnas Edad y Hijos
df = pd.DataFrame(diccionario, columns=[
    "Edad", "Hijos"], index=["Ana", "Jose"])
print("Dataframe:\n", df)

# creamos una lista de valores
valores = [[20, 50, 1],
           [21, 60, 2],
           [19, 55, 0],
           [18, 70, 0],
           [20, 65, 0]]

# creamos un dataframe con los valores de la lista
df1 = pd.DataFrame(valores, columns=["Edad", "Peso", "Hijos"], index=[
    "Ana", "Jose", "Maria", "Pedro", "Luis"])
print("Dataframe:\n", df1)

# creamos un dataframe con otro orden de columnas
valores2 = [[20, 21, 19, 18, 20],
            [50, 60, 55, 70, 65],
            [1, 2]]

df2 = pd.DataFrame(valores2, columns=[
    "Ana", "Jose", "Maria", "Pedro", "Luis"], index=["Edad", "Peso", "Hijos"])
print("Dataframe:\n", df2)

# accedemos a los elementos de la columna Edad
print("PESO: \n", df1["Peso"])

# accedemos a los elementos de la columna Edad y Hijos
print("\n", df1[["Edad", "Hijos"]])

# accedemos a los elementos de la fila Ana
print("\n", df1.loc["Ana"])

# accedemos a los elementos de la fila Ana y Jose
print("\n", df1.loc[["Ana", "Jose"]])

# accedemos a los elementos que tienen Peso mayor o igual a 60
print("\n", df1[df1["Peso"] >= 60])

# accedemos a los elementos que tienen Peso mayor o igual a 60 y Hijos mayor o igual a 1
print("\n", df1[(df1["Peso"] >= 60) & (df1["Hijos"] >= 1)])

# accedemos a las filas de la 1 a la 3
print(df1.iloc[1:3], "\n")

# consultamos el dataframe mediante el metodo query
print(df1.query("Peso >= 60 & Hijos >= 1"))

# agregamos una columna al dataframe
df1["Nacimiento"] = [1999, 1998, 1999, 1999, 1998]
print(df1, "\n")

# asignamos al dataframe df1 una columna mascotas
df1_mascotas = df1.assign(Mascotas=[1, 2, 0, 5, 0])
print(df1_mascotas, "\n")

# agregamos una columna de estatura al dataframe df1
df1["Estatura"] = [1.70, 1.80, 1.60, 1.75, 1.90]
print(df1, "\n")

# agregamos una columna calculada al dataframe df1 llamada IMC
df1["IMC"] = df1["Peso"] / df1["Estatura"] ** 2
print(df1, "\n")

# creamos una funcion que sume 2 a un valor del dataframe


def func(x):
    """
    This function adds 2 to the input value.
    """
    return x + 2


df1["Peso"] = df1["Peso"].apply(func)
print(df1, "\n")

# creamos una variable para evular si la persona tiene mas de2 hijos
TIENE_HIJOS = 2
print(df1.eval("Hijos >= @TIENE_HIJOS"))
