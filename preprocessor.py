from setcategorical import SetCategorical
from normalizer import Normalizer
from datasplitter import DataSplitter

class PreProcessor:

    # Tools for pre-processing stage
    def __init__(self):
        self.__encoder = SetCategorical()
        self.__scaler = Normalizer()
        self.__splitter = DataSplitter()

    def preprocess(self, dataset, train_size, test_size, seed=7, normalized=True):
        """ Cuida de todo o pré-processamento. """
        # limpeza dos dados
        dataset = self.__encoder.convert(dataset)

        # divisão de treino, validação e teste, com normalização inclusa
        train, val, test = \
        self.__prepare_holdout(dataset, train_size, test_size, seed, normalized)

        X_train, Y_train = self.__splitter.setXY(train)
        X_val, Y_val = self.__splitter.setXY(val)
        X_test, Y_test = self.__splitter.setXY(test)

        # Retorna os dados formatados
        return (X_train, X_val, X_test, Y_train, Y_val, Y_test)

    def __prepare_holdout(self, dataset, train_size, validation_size, seed, normalize):
        """ Divide os dados em treino, validação e teste usando o método holdout.
        Consideramos que a variável target está na última coluna.
        O parâmetro train_size é o percentual de dados de treino 
        e validation_size é o percentual de dados para validação.
        """

        # normalização/padronização
        if normalize:
            dataset = self.__scaler.set(dataset)

        # Retorna os parâmetros 
        return self.__splitter.split(dataset, train_size, validation_size, seed)


    # WIP
    def evaluate_novel_data(self, X, normalize=True):
        """ Realiza pré processamento de dados novos
        """
        # limpeza dos dados
        X = self.__encoder.convert(X)

        # definindo o transformador como min max scaler
        if normalize:
           X = self.__scaler.set(X)

        return X
