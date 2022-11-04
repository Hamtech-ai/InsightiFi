from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier

def RFClf(train_data, train_label, test_data, test_label):

    clf = RandomForestClassifier(
        n_estimators = 50,
        max_depth = 4,
        criterion = 'entropy',
        class_weight = 'balanced_subsample',
        random_state = 0
    )

    clf.fit(train_data, train_label)
    
    trainPred = clf.predict(train_data)
    testPred = clf.predict(test_data)

    trainProb = clf.predict_proba(train_data)
    testProb = clf.predict_proba(test_data)

    featureImport = clf.feature_importances_

    trainClassReport = metrics.classification_report(train_label, trainPred)
    testClassReport = metrics.classification_report(test_label, testPred)

    return trainPred, testPred, trainProb, testProb, featureImport, trainClassReport, testClassReport

