from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np

class Normalizer:

    def __init__(self):
        self.__scaler = MinMaxScaler()
        self.__fitModel = True

    def set(self, dataset):
        """ Realiza a normalização dos dados numéricos 
        convertendo-os entre 0 e 1
        """
        X = dataset.copy()
        X.pop('class_type')
        # definindo o transformador como min max scaler
        # scaler = MinMaxScaler()
        if self.__fitModel:
            self.__scaler.fit(X)
            self.__fitModel = False

        # transformando os dados (apenas variáveis contínuas)
        dataset[X.columns] = self.__scaler.transform(X)

        return dataset
