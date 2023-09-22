import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

class AttributeSelector:

    def selectChiSquare(self, X, Y, k_attr=10):
        """ Seleção de atributos baseada em distribuição Chi-quadrado.
        """
        # Função para seleção de atributos
        N = np.shape(X)[1]
        best_var = SelectKBest(score_func=chi2, k=k_attr)

        # Executa a função de pontuação em (X, Y) e obtém os atributos selecionados
        fit = best_var.fit(X, Y)

        # Reduz X para os atributos selecionados
        features = fit.transform(X)

        # Resultados
        print('\nOriginal feature set size:', X.shape[1])
        print('\nReduced set size:', features.shape[1])

        # Exibe as pontuações de cada atributos
        # (Basta mapear manualmente o índice dos nomes dos respectivos atributos)
        np.set_printoptions(precision=3) # 3 casas decimais
        # print(fit.scores_)

        # Imprime o dataset apenas com as colunas selecionadas
        # print(features)

        # ordena e seleciona os 20 melhores atributos
        sorted_attributes = np.argsort(fit.scores_)

        # lista os nomes dos atributos selecionados
        return sorted_attributes[N-k_attr:N]

    def selectExtraTrees(self, X, Y, k_attr=10):
        """ Seleção de atributos baseada em Extra Trees.
        """
        # Criação do modelo para seleção de atributos
        N = np.shape(X)[1]
        model = ExtraTreesClassifier(n_estimators=100)
        model.fit(X, Y)
        # Exibe a pontuação de importância para cada atributo
        # quanto maior a pontuação, mais importante é o atributo. 
        # print(model.feature_importances_)

        # ordena e seleciona os k melhores atributos
        sorted_attributes = np.argsort(model.feature_importances_)

        # lista os nomes dos atributos selecionados
        return sorted_attributes[N-k_attr:N]
