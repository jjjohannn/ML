"""preparacion del conjunto de datos como estandarizacion y normalizacion sobre el conjunto de datos KDD"""
import pandas as pd
# import numpy as np
import arff
from sklearn.model_selection import train_test_split
# from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import OrdinalEncoder
# from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import RobustScaler

# creamos funciones auxiliares


def load_kdd_data(data_path):
    """leemos los datos"""
    with open(data_path, 'r', encoding='utf-8') as f:
        data = arff.load(f)
    attributes = [attr[0] for attr in data['attributes']]
    return pd.DataFrame(data['data'], columns=attributes)


def train_vsl_test_split(df, rstate=42, shuffle=True, stratify=None):
    """separamos el conjunto de datos en entrenamiento, prueba y validacion"""
    strat = df[stratify] if stratify else None
    train, test = train_test_split(
        df, test_size=0.4, random_state=rstate, shuffle=shuffle, stratify=strat)
    strat = test[stratify] if stratify else None
    val, test = train_test_split(
        test, test_size=0.5, random_state=rstate, shuffle=shuffle, stratify=strat)
    return train, test, val


# leemos los datos
dataframe = load_kdd_data('../ML/datasets/NSL-KDD/KDDTrain+.arff')
# separamos los datos en entrenamiento, prueba y validacion
train_set, test_set, val_set = train_vsl_test_split(
    dataframe, stratify='protocol_type')

# copiamos las caracteristica de salida
X_train = train_set.drop('class', axis=1)
y_train = train_set['class'].copy()


"""Nos pondremos en el caso de que falten valores asi que procederemos a modificar el data frame"""
# X_train.loc[(X_train['src_bytes'] > 400 & (
#     X_train['src_bytes'] < 800), 'src_bytes')] = np.nan
# X_train.loc[(X_train['dst_bytes'] > 500 & (
#     X_train['dst_bytes'] < 2000), 'dst_bytes')] = np.nan

# comprobamos si hay valores nulos
# print(X_train.isna().any())


# seleccionamos las filas que tienen valores nulos
filas_nulas = X_train[X_train.isnull().any(axis=1)]


"""ahora ejemplificaremos 3 opciones para tratar los valores nulos,
ya que los modelos no pueden trabajar con ellos"""

# opcion 1: eliminar las filas que contienen valores nulos
# X_train_copy = X_train.copy()
# X_train_copy.dropna(subset=["src_bytes", "dst_bytes"], inplace=True)


# opcion 2: eliminar la columna que contiene valores nulos
# X_train_copy = X_train.copy()
# X_train_copy.drop(["src_bytes", "dst_bytes"], axis=1, inplace=True)
# print("contamos el numero de valores nulos: ",
#       len(list(X_train)) - len(list(X_train_copy)))

# opcion 3: rellenar los valores nulos con la media
# X_train_copy = X_train.copy()
# media_srcbytes = X_train_copy["src_bytes"].mean()
# media_dstbytes = X_train_copy["dst_bytes"].mean()
# X_train_copy["src_bytes"].fillna(media_srcbytes, inplace=True)
# X_train_copy["dst_bytes"].fillna(media_dstbytes, inplace=True)

# tambien podemos rellenar con la mediana
# X_train_copy = X_train.copy()
# mediana_srcbytes = X_train_copy["src_bytes"].median()
# mediana_dstbytes = X_train_copy["dst_bytes"].median()
# X_train_copy["src_bytes"].fillna(mediana_srcbytes, inplace=True)
# X_train_copy["dst_bytes"].fillna(mediana_dstbytes, inplace=True)

# alternativas sklearn
# X_train_copy = X_train.copy()
# imputer = SimpleImputer(strategy="median")
# X_train_copy_num = X_train_copy.select_dtypes(exclude="object")
# imputer.fit(X_train_copy_num)
# X_train_copy_num_noan = imputer.transform(X_train_copy_num)
# X_train_copy = pd.DataFrame(X_train_copy_num_noan,
#                             columns=X_train_copy_num.columns)
# print(X_train_copy.head(10))

# transformando atributos categoricos a numericos
# print(X_train.info())
# protocol_type = X_train["protocol_type"]
# protocol_type_encoded, protocol_type_categories = protocol_type.factorize()
# # mostramos como se ha codificado
# for i in range(10):
#     print(protocol_type.iloc[i], " = ", protocol_type_encoded[i])

# print(protocol_type_categories)

# tranformaciones avanzadas ordinal encoder
# (algunos problemas con esto0 se pueden ver en clastering ya que se basan en distancia )

# protocol_type = X_train[["protocol_type"]]
# OrdinalEncoder = OrdinalEncoder()
# protocol_type_encoded = OrdinalEncoder.fit_transform(protocol_type)
# print(protocol_type_encoded[:10])
# print(OrdinalEncoder.categories_)

# one hot encoder
# protocol_type = X_train[["protocol_type"]]
# oh_encoder = OneHotEncoder()
# protocol_type_oh = oh_encoder.fit_transform(protocol_type)
# protocol_type_oh.toarray()
# for i in range(10):
#     print(protocol_type["protocol_type"].iloc[i],
#           " = ", protocol_type_oh.toarray()[i])

# get_dummies
# print(pd.get_dummies(X_train["protocol_type"]))

# escalando conjunto de datos (normalizacion y estandarizacion)
# no aplican a la caracteristicas de salida
scaler_attributes = X_train[['src_bytes', 'dst_bytes']]
robus_scaler = RobustScaler()
X_train_scaled = robus_scaler.fit_transform(scaler_attributes)
X_train_scaled = pd.DataFrame(X_train_scaled, columns=[
                              'src_bytes', 'dst_bytes'])
print(X_train_scaled.head(10))
print(X_train.head(10))
