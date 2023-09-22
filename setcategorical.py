from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np

class SetCategorical:

    def __init__(self):
        self.__encoder = OneHotEncoder(drop='first', sparse=False)
        self.__fitModel = True

    def convert(self, data):
        """ Realiza o tratamento dos dados categóricos 
        convertendo-os para OneHotEncoder
        """
        # Seleciona os dados categóricos
        X = data.select_dtypes(exclude=[np.number])
        # Encoding dos dados coletados, com exceção da classe
        X_encoded = self.__set_onehotencoder(X.values[:,0:-1])

        # tratamento do nome das colunas após o encoder
        column_name = pd.Series(X.columns.values[0:-1])
        column_name = self.__encoder.get_feature_names_out(column_name)

        # Juntando as variáveis novamente
        return pd.concat([data.select_dtypes(exclude=['object']),
                          pd.DataFrame(X_encoded, columns=column_name), 
                          data['class_type']], axis=1, join='inner')

    def __set_onehotencoder(self, X):
        # definindo o transformador como one hot encoding (com Dummy variable encoder)
        # encoder = OneHotEncoder(drop='first', sparse=False)
        if self.__fitModel:
            self.__encoder.fit(X)
            self.__fitModel = False

        return self.__encoder.transform(X)
