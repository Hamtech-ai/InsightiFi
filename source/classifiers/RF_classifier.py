
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from joblib import dump, load

def RF_classifier(train_data, train_labels, test_data, eval_data, \
    params=None, model_saving_folder=None, saving_prefix=None, model_name=None, eval_mode=False):
    if params is not None:
        n_estimator = params["n_estimator"] if "n_estimator" in params else 50
        max_depth = params["max_depth"] if "max_depth" in params else None
    else:
        n_estimator = 50
        max_depth = None

    #Create a Gaussian Classifier
    clf = RandomForestClassifier(n_estimators=n_estimator, criterion='entropy', class_weight='balanced_subsample', \
                                max_depth=max_depth)

    if eval_mode is False:
        #Training
        clf.fit(train_data, train_labels)

        # model save
        if (model_saving_folder is not None) and (saving_prefix is not None) and (model_name is not None):
            dump(clf, model_saving_folder + '/' + saving_prefix + '_' + model_name + '.joblib') 
    else:
        # model load
        if (model_saving_folder is not None) and (saving_prefix is not None) and (model_name is not None):
            clf = load(model_saving_folder + '/' + saving_prefix + '_' + model_name + '.joblib') 

    # data prediction
    train_pred = clf.predict(train_data)
    test_pred = clf.predict(test_data)
    evaluation_pred = clf.predict(eval_data)

    # prediction probability
    train_pred_proba = clf.predict_proba(train_data)
    test_pred_proba = clf.predict_proba(test_data)
    evaluation_pred_proba = clf.predict_proba(eval_data) 

    # feature importance
    feature_importance = clf.feature_importances_
    
    # returning the outputs
    return train_pred, test_pred, evaluation_pred, train_pred_proba, \
        test_pred_proba, evaluation_pred_proba, feature_importance