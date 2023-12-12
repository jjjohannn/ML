"""Creacion de transformadores de datos para el pipeline de entrenamiento"""
import arff
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer


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


df = load_kdd_data('../ML/datasets/NSL-KDD/KDDTrain+.arff')
train_set, test_set, val_set = train_vsl_test_split(
    df, stratify='protocol_type')

X_train = train_set.drop('class', axis=1)
y_train = train_set['class'].copy()


"""Nos pondremos en el caso de que falten valores asi que procederemos a modificar el data frame"""

X_train.loc[(X_train['src_bytes'] > 400) & (
    X_train['src_bytes'] < 800), 'src_bytes'] = np.nan
X_train.loc[(X_train['dst_bytes'] > 500) & (
    X_train['dst_bytes'] < 2000), 'dst_bytes'] = np.nan

# transformador para eliminar las filas con valores nulos


class DeleteNanRows(BaseEstimator, TransformerMixin):
    """Transformador para eliminar las filas con valores nulos"""

    def __init__(self):
        pass

    def fit(self, X, y=None):
        """Ajusta el transformador a los datos de entrada.

        Args:
            X (array-like): Datos de entrada.
            y (array-like, optional): Etiquetas de salida. Default es None.

        Returns:
            self: El objeto transformador ajustado.
        """
        return self

    def transform(self, X, y=None):
        """Transforma los datos de entrada.

        Args:
            X (array-like): Datos de entrada.
            y (array-like, optional): Etiquetas de salida. Default es None.

        Returns:
            array-like: Datos transformados.
        """
        return X.dropna()

# transformador para escalar los atributos


class CustomScaler(BaseEstimator, TransformerMixin):
    """Transformador para eliminar las columnas seleccionadas y escalar los atributos"""

    def __init__(self, attributes):
        self.attributes = attributes

    def fit(self, X, y=None):
        """Ajusta el transformador a los datos de entrada.

        Args:
            X (array-like): Datos de entrada.
            y (array-like, optional): Etiquetas de salida. Default es None.

        Returns:
            self: El objeto transformador ajustado.
        """
        return self

    def transform(self, X, y=None):
        """Transforma los datos de entrada.

        Args:
            X (array-like): Datos de entrada.
            y (array-like, optional): Etiquetas de salida. Default es None.

        Returns:
            array-like: Datos transformados.
        """
        X_copy = X.copy()
        scaler_attr = X_copy[self.attributes]
        robust_scaler = RobustScaler()
        X_scaled = robust_scaler.fit_transform(scaler_attr)
        X_scaled = pd.DataFrame(
            X_scaled, columns=self.attributes, index=X_copy.index)
        for attr in self.attributes:
            X_copy[attr] = X_scaled[attr]
        return X_copy

# Transformador para codificar las caracteristicas categoricas y devolver un dataframe


class CustomOneHotEncoder():
    """Transformador para codificar características categóricas y devolver un dataframe"""

    def __init__(self):
        self.encoder = OneHotEncoder(sparse=False)
        self.columns = None
        self._oh = OneHotEncoder()

    def fit(self, x, y=None):
        """Ajusta el transformador a los datos de entrada.

        Args:
            x (array-like): Datos de entrada.
            y (array-like, optional): Etiquetas de salida. Default es None.

        Returns:
            self: El objeto transformador ajustado.
        """
        x_cat = x.select_dtypes(include=['object'])
        self._columns = pd.get_dummies(x_cat).columns
        self._oh.fit(x_cat)
        return self

    def transform(self, x, y=None):
        """Transforma los datos de entrada.

        Args:
            x (array-like): Datos de entrada.
            y (array-like, optional): Etiquetas de salida. Default es None.

        Returns:
            array-like: Datos transformados.
        """
        x_copy = x.copy()
        x_cat = x.copy().select_dtypes(include=['object'])
        x_num = x_copy.select_dtypes(exclude=['object'])
        x_cat_oh = self._oh.transform(x_cat)
        x_cat_oh = pd.DataFrame(
            x_cat_oh, columns=self._columns, index=x_copy.index)
        x_copy.drop(list(x_cat), axis=1, inplace=True)
        return x_copy.join(x_cat_oh)


# delete_nan_rows = DeleteNanRows()
# X_train_prep = delete_nan_rows.fit_transform(X_train)

# custon_scaler = CustomScaler(['src_bytes', 'dst_bytes'])
# X_train_prep = custon_scaler.fit_transform(X_train_prep)


# ----------------------------------PIPELINE----------------------------------#

num_pipeline = Pipeline(
    [('imputer', SimpleImputer(strategy='median')), ('scaler', RobustScaler())])


# atributos numericos
num_attribs = list(X_train.select_dtypes(exclude=['object']))
# atributos categoricos
cat_attribs = list(X_train.select_dtypes(include=['object']))

full_pipeline = ColumnTransformer(
    [('num', num_pipeline, num_attribs), ('cat', OneHotEncoder(), cat_attribs),])

X_train_prep = full_pipeline.fit_transform(X_train)
X_train_prep = pd.DataFrame(X_train_prep, columns=list(
    pd.get_dummies(X_train)), index=X_train.index)

print(X_train_prep.head(10))
