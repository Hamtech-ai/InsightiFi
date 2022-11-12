import json as json

# parameters
TECHNICAL_INDICATOR_CONFIG = 'configs/technical_indicator_parameters.json'
CLASSIFIER_CONFIG = 'configs/classifiers_parameters.json'
CLASSIFIER_FEATURE_INFO_KEY = 'feature_info'
CLASSIFIER_LABEL_INFO_KEY = 'label_info'
CLASSIFIER_HORIZON_KEY = 'classifier_horizon'

def technical_parameters(method):
    path = TECHNICAL_INDICATOR_CONFIG
    with open(path, 'rb') as infile:
        ta_config = json.load(infile)
    
    return ta_config[method]

def classifier_parameters(horizon, model_name):
    path = CLASSIFIER_CONFIG
    feature_info_key = CLASSIFIER_FEATURE_INFO_KEY
    label_info_key = CLASSIFIER_LABEL_INFO_KEY
    horizon_key = CLASSIFIER_HORIZON_KEY
    with open(path, 'rb') as infile:
        clf_config = json.load(infile)
    
    classifier = clf_config[horizon_key][horizon][model_name]
    
    # selected features
    selected_features = classifier["selected_features"]
    if selected_features is None: # All
        classifier["selected_features"] = None
    elif type(selected_features) != list:
        feature_info = clf_config[feature_info_key]
        classifier["selected_features"] = feature_info[selected_features]
    
    # label map
    label_name = classifier["label_name"]
    if "label_map" not in classifier.keys():
        label_info = clf_config[label_info_key]
        classifier["label_map"] = label_info[label_name]["label_map"]
    
    return classifier

def all_horizons():
    path = CLASSIFIER_CONFIG
    horizon_key = CLASSIFIER_HORIZON_KEY
    with open(path, 'rb') as infile:
        clf_config = json.load(infile)
    
    horizons = clf_config[horizon_key]
    
    return [key for key in horizons.keys()]

def horizon_classifiers(horizon, just_model=False, just_ensemble=False):
    path = CLASSIFIER_CONFIG
    horizon_key = CLASSIFIER_HORIZON_KEY
    with open(path, 'rb') as infile:
        clf_config = json.load(infile)
    
    classifiers = clf_config[horizon_key][horizon]
    
    result = list(classifiers.keys())
    
    # just models
    if just_model is True:
        result = [item for item in result if item.startswith('model')]
    
    # just ensembles
    if just_ensemble is True:
        result = [item for item in result if item.startswith('ensemble')]
    
    return result

def label_name_horizon_classifiers(horizon):
    path = CLASSIFIER_CONFIG
    horizon_key = CLASSIFIER_HORIZON_KEY
    label_info_key = CLASSIFIER_LABEL_INFO_KEY
    with open(path, 'rb') as infile:
        clf_config = json.load(infile)
    
    classifiers = clf_config[horizon_key][horizon]

    result_dict = dict()
    for clf_name,clf in classifiers.items():
        if clf["label_name"] not in result_dict:
            result_dict[clf["label_name"]] = list()
        result_dict[clf["label_name"]].append(clf_name)
    
    # labels info
    labels_info = clf_config[label_info_key]
    
    return result_dict, labels_info