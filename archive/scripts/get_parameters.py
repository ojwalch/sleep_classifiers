import numpy as np
import time

from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from sklearn.utils import class_weight

import utilities
import classify_sleep

verbose = False


def find_best(method_key, feature_set, training_subjects):
    # Load up all the data
    data_dict = utilities.build_data_dictionary(feature_set)

    # Initialize holders
    training_set_features = np.array([])
    training_set_labels = np.array([])

    # Build vectors for training subjects
    for subject in training_subjects:
        score_features, full_features = utilities.get_features(subject, data_dict)
        if np.shape(training_set_features)[0] == 0:
            training_set_features = full_features
            training_set_labels = score_features
        else:
            training_set_features = np.vstack((training_set_features, full_features))
            training_set_labels = np.vstack((training_set_labels, score_features))

    # Convert raw scores from 0-5 to binary,or 0-2
    training_set_labels = utilities.process_raw_scores(training_set_labels, classify_sleep.run_flag)

    if method_key == 'Logistic Regression':
        parameters = {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'penalty': ['l1', 'l2']}
        classifier = LogisticRegression()

    if method_key == 'KNeighbors':
        parameters = {'n_neighbors': [500, 1000, 2000]}
        classifier = KNeighborsClassifier()

    if method_key == 'MLP':
        parameters = {'solver': ['lbfgs'], 'max_iter': [1000], 'alpha': 10.0 ** -np.arange(1, 4),
                      'hidden_layer_sizes': [(30, 30, 30)]}
        classifier = MLPClassifier()

    if method_key == 'Random Forest':
        max_depth = [int(x) for x in np.linspace(10, 110, num=2)]
        max_depth.append(None)
        max_depth = [10, 50, 100]
        min_samples_split = [10]
        min_samples_leaf = [32]
        parameters = {'n_estimators': [50], 'max_features': [None], 'max_depth': max_depth,
                      'min_samples_split': min_samples_split, 'min_samples_leaf': min_samples_leaf, 'bootstrap': [True]}
        classifier = RandomForestClassifier()

    class_weights = class_weight.compute_class_weight('balanced',
                                                      classes=np.unique(training_set_labels),
                                                      y=training_set_labels)
    class_weight_dict = {0: class_weights[0], 1: class_weights[1]}

    if len(class_weights) > 2:
        class_weight_dict = {0: class_weights[0], 1: class_weights[1], 2: class_weights[2]}

    classifier.class_weight = class_weight_dict

    if classify_sleep.run_flag == utilities.RUN_REM:
        scoring = 'neg_log_loss'
    else:
        scoring = 'roc_auc'

    clf = GridSearchCV(classifier, parameters, scoring=scoring)

    clf.fit(training_set_features, training_set_labels)

    if verbose:
        print('Best parameters for set:')
        print(clf.best_params_)
        print('Score on training data: ' + str(clf.score(training_set_features, training_set_labels)))

    save_name = 'parameters/' + method_key + utilities.string_from_features(feature_set) + '.npy'
    np.save(save_name, clf.best_params_)

    return clf.best_params_


