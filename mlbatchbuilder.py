class MlBatchBuilder:

    def __init__(self, features=None):
        self.__data = []
        self.__features = features

    def insert_model(self, model, X_train, Y_train):
        if self.__features is None:
            self.__data.append((model.get_name(), model.train(X_train, Y_train)))
        else:
            X_selected = X_train.iloc[:, self.__features]
            self.__data.append((model.get_name(), model.train(X_selected, Y_train)))

    def get_models(self):
        return self.__data

    def set_features(self, X):
        if self.__features is None:
            return X
        else:
            return X.iloc[:, self.__features]
