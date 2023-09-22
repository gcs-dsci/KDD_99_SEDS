def models_set:
    # Criação dos modelos
    feat_select_models = []
    selector = []
    selector.append(('None', None))
    selector.append(('Chi-2', sorted_attr_KB))
    selector.append(('XTrees', sorted_attr_ET))
    for name, features in feat_select_models:
        print("Gerando classificação para o modelo", name)
        ml_batch_builder = MlBatchBuilder(features)
        ml_batch_builder.insert_model(WekaModel("weka.classifiers.trees.J48"), X_train, Y_train)    # DTs
        ml_batch_builder.insert_model(WekaModel("weka.classifiers.functions.SMO"), X_train, Y_train) # SVM
        ml_batch_builder.insert_model(WekaModel("weka.classifiers.lazy.IBk"), X_train, Y_train)  # kNN - k = 1
        ml_batch_builder.insert_model(WekaModel("weka.classifiers.trees.RandomForest"), X_train, Y_train) # Random Forests
        ml_batch_builder.insert_model(WekaModel("weka.classifiers.bayes.NaiveBayes"), X_train, Y_train) # Naive Bayes        ml_batch_builder.insert_model(WekaModel("weka.classifiers.functions.MultilayerPerceptron"), X_train, Y_train)
        feat_select_models.append((name, ml_batch_builder))
    return feat_select_models