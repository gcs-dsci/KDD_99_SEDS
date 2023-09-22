from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate

class MlEvaluator:

    def evaluate_cross_validation(self, model, X_test, Y_test, \
                                  num_folds=10, scoring='accuracy'):
        """ Faz uma predição e avalia o modelo. Poderia parametrizar o tipo de
        avaliação, entre outros.
        """
        kfold = KFold(n_splits=num_folds)
        if (type(scoring) == list):
            cv_results = cross_validate(model, X_test, Y_test, \
                                     cv=kfold, scoring=scoring)
        else:
            cv_results = cross_val_score(model, X_test, Y_test, \
                                     cv=kfold, scoring=scoring)
        return cv_results

    def predict(self, model, X_test):
        """ Exibe a predição do modelo.
        """
        return model.predict(X_test)
