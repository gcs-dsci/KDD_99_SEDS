import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklweka.dataset import load_arff, to_nominal_labels

class DataSplitter:

    def split(self, dataset, train_size, val_size=None, seed=7):
        # Se o percentual de validação não for usado, define como zero
        if val_size is None:
            val_size = 0.0

        # divisão em treino, validação e teste, com normalização inclusa
        test_size = 1 - (train_size + val_size)
        train, test = train_test_split(dataset, test_size=(1 - train_size), random_state=seed)
        # Divide o restante em validação e teste, caso validation_size != 0
        if (val_size > 0):
            train, val = train_test_split(train, test_size=(test_size/(test_size + val_size)), random_state=seed)
            print(len(train), 'train examples')
            print(len(val), 'validation examples')
            print(len(test), 'test examples')

            # Retorna os dados formatados
            return (train, val, test)

        # Caso não haja validação, retorna apenas treino e teste
        print(len(train), 'train examples')
        print(len(test), 'test examples')

        return (train, test)

    def setXY(self, dataset):
        X = dataset.copy()
        y = X.pop('class_type')
        y = to_nominal_labels(y)
        return X, y