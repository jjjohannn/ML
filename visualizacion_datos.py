"""NSL-KDD Dataset conjunto de datos para clasificar trafico de red como anomalo o normal """
import pandas as pd
import arff
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from pandas.plotting import scatter_matrix

# vemos los datos del archivo KDDTrain+.txt
df = pd.read_csv("../ML/datasets/NSL-KDD/KDDTrain+.txt", header=None)

# vemos los datos del archivo KDDTrain+.arff
# with open("../ML/datasets/NSL-KDD/KDDTrain+.arff", "r", encoding="utf-8") as f:
#     data = arff.load(f)

# atributes = [i[0] for i in data['attributes']]
# print(pd.DataFrame(data['data'], columns=atributes))

# creamos una funcion para leer los datos de los archivos


def load_kdd_data(path):
    """lectura del conjunto de datos NSL-KDD'"""
    with open(path, "r", encoding="utf-8") as f:
        data = arff.load(f)
    atributes = [i[0] for i in data['attributes']]
    return pd.DataFrame(data['data'], columns=atributes)


df_original = load_kdd_data("../ML/datasets/NSL-KDD/KDDTrain+.arff")
df = df_original.copy()

# mostramos algunas caracteristicas de los datos
# print(df.shape)
# print(df.describe())
# print(df['class'].value_counts())
# print(df['protocol_type'].value_counts())

# creamos un histograma para ver la distribucion de los datos
# df.hist(bins=50, figsize=(20, 15))
# plt.show()

# buscamos correlaciones entre los datos
# primero convertimos los datos categoricos a numericos class y protocol_type
LabelEncoder = LabelEncoder()
columns = ['protocol_type', 'class', 'service', 'flag']
for column in columns:
    df[column] = LabelEncoder.fit_transform(df[column])


# luego buscamos las correlaciones de la caracteristica de salida (class)
corr_matrix = df.corr()
# corr_matrix['class'].sort_values(ascending=False)
# print(corr_matrix['class'].sort_values(ascending=False))

# Mostramos una matris de correlacion
# corr = df.corr()
# fig, ax = plt.subplots(figsize=(8, 8))
# ax.matshow(corr)
# plt.xticks(range(len(corr.columns)), corr.columns)
# plt.yticks(range(len(corr.columns)), corr.columns)
# print(plt.show())

# mostramos una matriz de dispersion solo para algunas caracteristicas (los sospechosos de mayor correlacion)
attributes = ['same_srv_rate', 'dst_host_srv_count',
              'dst_host_same_srv_rate', 'class']

scatter_matrix(df[attributes], figsize=(12, 8))
plt.show()
