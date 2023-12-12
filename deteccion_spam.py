from html.parser import HTMLParser
import email
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import nltk
# import pandas as pd

# descargamos los datos necesarios para el procesamiento de los emails
nltk.download('stopwords')

# creamos una clase para facilitar el procesamiebto de los archivos html


class MyHTMLParser(HTMLParser):
    """Clase para procesar archivos HTML y extraer texto"""

    def __init__(self):
        super().__init__()  # Call the base class constructor
        self.reset()
        self.strict = False
        self.convert_charrefs = True
        self.fed = []

    def handle_data(self, data):
        self.fed.append(data)

    def get_data(self):
        """Devuelve los datos acumulados como una cadena de texto."""
        return ''.join(self.fed)


def procesar_html(html):
    """ procesa el html y devuelve el texto """
    parser = MyHTMLParser()
    parser.feed(html)
    return parser.get_data()


# creamos un ejemplo para probar la eliminacion de tags html
# HTML = "<html><head><title>Test</title></head><body><h1>Parse me!</h1></body></html>"
# print(procesar_html(HTML))

# creamos una clase parser para eliminar campos no relevantes de los emails (stemming)

class Parser:
    """Clase para analizar y procesar emails"""

    def __init__(self):
        self.stemmer = nltk.PorterStemmer()
        self.stopwords = set(nltk.corpus.stopwords.words('english'))
        self.punctuation = list(string.punctuation)

    def parse(self, email_path):
        """ parsea el email  """
        path1 = "../ML/datasets/trec07p/"+email_path
        with open(path1, encoding='utf-8', errors='ignore') as f:
            msg = email.message_from_file(f)
        return None if not msg else self.get_email_content(msg)

    def get_email_content(self, msg):
        """ devuelve el contenido del email """
        subject = self.tokenize(msg['subject']) if msg['subject'] else []
        body = self.get_email_body(msg.get_payload(), msg.get_content_type())
        content_type = msg.get_content_type()
        # retornamos el contenido del email
        return {'subject': subject, 'body': body, 'content_type': content_type}

    def get_email_body(self, payload, content_type):
        """
        Extracts the body of the email.
        Args:
            payload (str or list): The payload of the email.
            content_type (str): The content type of the email.
        Returns:
            list: The tokenized body of the email.
        """
        # Extract the body of the email
        body = []
        if isinstance(payload, str) and content_type == 'text/plain':
            return self.tokenize(payload)
        if isinstance(payload, str) and content_type == 'text/html':
            return self.tokenize(procesar_html(payload))
        if isinstance(payload, list):
            for p in payload:
                body += self.get_email_body(p.get_payload(),
                                            p.get_content_type())
        return body

    def tokenize(self, text):
        """tranforma un una cadena de texto en tokens.
        realiza dos acciones principales: stemming y eliminacion de puntacion"""
        for c in self.punctuation:
            text = text.replace(c, '')
        text = text.replace('\n', ' ')
        text = text.replace('\t', ' ')
        tokens = list(filter(None, text.split(' ')))
        # stemming del texto
        return [self.stemmer.stem(w) for w in tokens if w not in self.stopwords]

# creamos un ejemplo para probar el parser
# inemail = open("inmail.1").read()
# print(inemail)


# probamos con un email con formato wav
# p = Parser()
# print(p.parse("../ML/datasets/trec07p/data/inmail.1"))

# abrimos el archivo index para ver la etiqueta de cada email
# with open("../ML/datasets/trec07p/full/index", encoding='utf-8') as file:
#      index = file.readlines()
# print(index)

def parse_index(path_to_index, n_elements):
    """
    Parses the index file and returns a list of dictionaries containing the label and email path.
    Args:
        path_to_index (str): The path to the index file.
        n_elements (int): The number of elements to parse.
    Returns:
        list: A list of dictionaries containing the label and email path.
    """
    ret_indexes = []
    with open(path_to_index, encoding='utf-8') as file:
        index_lines = file.readlines()
    for i in range(n_elements):
        mail_line = index_lines[i].split(" ../")
        email_label = mail_line[0]  # Renamed 'label' to 'email_label'
        path = mail_line[1][:-1]
        ret_indexes.append({"label": email_label, "email_path": path})
    return ret_indexes


def parse_emails(email_index):
    """
    Parses the emails based on the given index.
    Args:
        email_index (dict): The index containing the label and email path.
    Returns:
        tuple: A tuple containing the parsed email and its label.
    """
    p = Parser()
    parsed_email = p.parse(email_index["email_path"])
    return parsed_email, email_index["label"]

# probamos parse_emails con los primeros 10 emails
# indexes = parse_index("../ML/datasets/trec07p/full/index", 10)
# print(indexes)


# Probamos parse_emails para el primer email
# index = parse_index("../ML/datasets/trec07p/full/index", 1)
# mail, label = parse_emails(index[0])
# print("El correo es: ", label)
# print("El contenido del correo es: ", mail)


# aplicamos countVectorizer
# preparacion del email en una cadena de texto
# prep_email = [" ".join(mail["subject"])+" ".join(mail["body"])]
# vectorizer = CountVectorizer()
# x = vectorizer.fit(prep_email)
# print("Email ", prep_email, '\n')
# print("Caracteristicas de entrada: ", vectorizer.get_feature_names_out())

# X = vectorizer.transform(prep_email)
# print("\nValores de datos de entrada:\n ", X.toarray())

# Aplicamos one hot encoding a las etiquetas
# prep_email = [[w]for w in mail['subject']+mail['body']]
# enc = OneHotEncoder(handle_unknown='ignore')
# X = enc.fit_transform(prep_email)
# print("\nValores de datos de entrada:\n ", X.toarray())

# Funcion auxiliar para el procesamiento del conjunto de datos
def create_prep_dataset(index_path, n_elements):
    """
    Creates a preprocessed dataset based on the given index file.
    Args:
        index_path (str): The path to the index file.
        n_elements (int): The number of elements to include in the dataset.
    Returns:
        tuple: A tuple containing the preprocessed email texts and their corresponding labels.
    """
    x = []
    y = []
    indexes = parse_index(index_path, n_elements)
    for i in range(n_elements):
        print("\rParsing email: ", i, " "*5, end="")
        mail, label = parse_emails(indexes[i])
        x.append(" ".join(mail["subject"])+" ".join(mail["body"]))
        y.append(label)
    return x, y


# leemos unicamente un subconjunto de 100 correos
# X_train, Y_train = create_prep_dataset(
#     "../ML/datasets/trec07p/full/index", 100)
# # print(X_train)

# Aplicamos vectorizacion a los datos
# vectorizer = CountVectorizer()
# X_train = vectorizer.fit_transform(X_train)
# print(pd.DataFrame(X_train.toarray(), columns=vectorizer.get_feature_names_out()))

# Entrenamineto del algoritmo de regresion logistica con los primeros 100 correos (clf=clasificador)
# clf = LogisticRegression()
# clf.fit(X_train, Y_train)

# # Probamos el algoritmo
# X, Y = create_prep_dataset("../ML/datasets/trec07p/full/index", 150)
# X_test = X[100:]
# Y_test = Y[100:]
# X_test = vectorizer.transform(X_test)
# Y_pred = clf.predict(X_test)
# print("\nPredicciones: ", Y_pred)
# print("Etiquetas reales: ", Y_test)

# # evaluamos el algoritmo
# print("Precision: ", accuracy_score(Y_test, Y_pred))


# ------------------------------------------------------------
# Aplicamos el algoritmo a 12000 correos
X, Y = create_prep_dataset(
    "../ML/datasets/trec07p/full/index", 12000)
X_test = X[10000:]
Y_test = Y[10000:]
X_train = X[:10000]
Y_train = Y[:10000]
# Aplicamos vectorizacion a los datos a X_train
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(X_train)
# Entrenamineto del algoritmo de regresion logistica con los primeros 10000 correos (clf=clasificador)
clf = LogisticRegression()
clf.fit(X_train, Y_train)
# Probamos el algoritmo con los 2000 correos restantes
X_test = vectorizer.transform(X_test)
Y_pred = clf.predict(X_test)
print("precision: ", accuracy_score(Y_test, Y_pred))
