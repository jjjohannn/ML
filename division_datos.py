"""division del conjunto de datos en entrenamiento y prueba. KDDTrain"""

import pandas as pd
import arff
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def load_kdd_data(path):
    """lectura del conjunto de datos NSL-KDD'"""
    with open(path, "r", encoding="utf-8") as f:
        data = arff.load(f)
    atributes = [i[0] for i in data['attributes']]
    return pd.DataFrame(data['data'], columns=atributes)


dataframe = load_kdd_data("../ML/datasets/NSL-KDD/KDDTrain+.arff")
# print(df.info())

# dividimos el conjunto de datos en entrenamiento y prueba.
# ademas agregamos stratify samplin para evitar sampling bias
# train_set, test_set = train_test_split(
#     df, test_size=0.4, random_state=42, stratify=df['protocol_type'])

# dividimos el conjunto test en validacion y prueba
# val_set, test_set = train_test_split(test_set, test_size=0.5, random_state=42)

# vemos la longitud de los conjuntos
# print("Longitud del conjunto de entrenamiento: ", len(train_set))
# print("Longitud del conjunto de validacion: ", len(val_set))
# print("Longitud del conjunto de prueba: ", len(test_set))

# creamos una funcion que nos realice la division de los datos en tres conjuntos


def train_val_test_split(df, rstate=42, shuffle=True, stratify=None):
    """
    Split the dataset into training, validation, and test sets.

    Args:
        df (pandas.DataFrame): The input dataset.
        rstate (int, optional): Random state for reproducibility. Defaults to 42.
        shuffle (bool, optional): Whether to shuffle the data before splitting. Defaults to True.
        stratify (str, optional): Column name for stratified sampling. Defaults to None.

    Returns:
        tuple: A tuple containing the training, validation, and test sets.
    """
    strat = df[stratify] if stratify else None
    train_set, test_set = train_test_split(
        df, test_size=0.4, random_state=rstate, shuffle=shuffle, stratify=strat)
    strat = test_set[stratify] if stratify else None
    val_set, test_set = train_test_split(
        test_set, test_size=0.5, random_state=rstate, shuffle=shuffle, stratify=strat)
    return train_set, val_set, test_set


print("longitud del conjunto inicial: ", len(dataframe))
# dividimos el conjunto de datos en entrenamiento, validacion y prueba
train_set, val_set, test_set = train_val_test_split(
    dataframe, stratify='protocol_type')

print("Longitud del conjunto de entrenamiento: ", len(train_set))
print("Longitud del conjunto de validacion: ", len(val_set))
print("Longitud del conjunto de prueba: ", len(test_set))

# comprobamos que stratify mantenga la proporcion de los datos
dataframe["protocol_type"].hist()
train_set["protocol_type"].hist()
val_set["protocol_type"].hist()
test_set["protocol_type"].hist()
plt.show()
