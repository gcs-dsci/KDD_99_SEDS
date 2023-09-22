from sklweka.classifiers import WekaEstimator
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

class MLModel:
    """ Classe abstrata de referência.
    """
    def __init__(self, name):
        self.__name = name

    def get_name(self):
        return self.__name

    def train(self):
        pass

class TreesModel(MLModel):
    """ Cria e treina um modelo de Árvore de Decisão.
    """
    def __init__(self):
        self.__name = 'CART'

    def get_name(self):
        return self.__name

    def train(self, X_train, Y_train):
        model = DecisionTreeClassifier()
        model.fit(X_train, Y_train)
        return model


class SVMModel(MLModel):
    """ Cria e treina um modelo SVM.
    """
    def __init__(self):
        self.__name = 'SVM'

    def get_name(self):
        return self.__name

    def train(self, X_train, Y_train):
        model = LinearSVC()
        model.fit(X_train, Y_train)
        return model


class RFEnsembleModel(MLModel):
    """ Cria e treina um modelo de Random Forests.
    """
    def __init__(self, num_estimator=10):
        self.__name = 'RF'
        self.__estimators = num_estimator

    def get_name(self):
        return self.__name

    def train(self, X_train, Y_train):
        model = RandomForestClassifier(n_estimators=self.__estimators)
        model.fit(X_train, Y_train)
        return model


class WekaModel(MLModel):
    """ Classe abstrata de referência.
    """
    def __init__(self, algo, params=None):
        self.__name = algo.split('.')[-1]
        self.__params = params
        self.__algo = algo

    def get_name(self):
        return self.__name

    def get_command(self):
        model = WekaEstimator(classname=self.__algo)
        return model.to_commandline()

    def train(self, X, y):
        model = WekaEstimator(classname=self.__algo)
        model.fit(X, y)
        return model

