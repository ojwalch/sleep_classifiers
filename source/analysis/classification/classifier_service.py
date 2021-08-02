import time
from functools import partial
from multiprocessing import Pool, cpu_count

import numpy as np
from sklearn.utils import class_weight

from source.analysis.classification.classifier_input_builder import ClassifierInputBuilder
from source.analysis.classification.parameter_search import ParameterSearch
from source.analysis.performance.raw_performance import RawPerformance
from source.constants import Constants


class ClassifierService(object):

    @staticmethod
    def run_sw(data_splits, classifier, subject_dictionary, feature_set):
        return ClassifierService.run_in_parallel(ClassifierService.run_single_data_split_sw,
                                                 data_splits, classifier,
                                                 subject_dictionary, feature_set)

    @staticmethod
    def run_three_class(data_splits, classifier, subject_dictionary, feature_set):
        return ClassifierService.run_in_parallel(ClassifierService.run_single_data_split_three_class,
                                                 data_splits, classifier,
                                                 subject_dictionary, feature_set)

    @staticmethod
    def run_three_class_with_loaded_model(data_splits, classifier, subject_dictionary, feature_set):

        raw_performances = []
        for ind in range(len(data_splits)):
            data_split = data_splits[ind]
            if ind == 0:
                training_x, training_y = ClassifierInputBuilder.get_three_class_inputs(data_split.training_set,
                                                                                       subject_dictionary=subject_dictionary,
                                                                                       feature_set=feature_set)
                classifier = ClassifierService.train_classifier(training_x, training_y, classifier, 'neg_log_loss')

            testing_x, testing_y = ClassifierInputBuilder.get_three_class_inputs(data_split.testing_set,
                                                                                 subject_dictionary=subject_dictionary,
                                                                                 feature_set=feature_set)
            class_probabilities = classifier.predict_proba(testing_x)

            raw_performance = RawPerformance(true_labels=testing_y, class_probabilities=class_probabilities)
            raw_performances.append(raw_performance)

        return raw_performances

    @staticmethod
    def run_in_parallel(function, data_splits, classifier, subject_dictionary, feature_set):
        pool = Pool(cpu_count())

        single_run_wrapper = partial(function,
                                     attributed_classifier=classifier,
                                     subject_dictionary=subject_dictionary,
                                     feature_set=feature_set)

        results = pool.map(single_run_wrapper, data_splits)

        return results

    @staticmethod
    def run_single_data_split_sw(data_split, attributed_classifier, subject_dictionary, feature_set):

        training_x, training_y = ClassifierInputBuilder.get_sleep_wake_inputs(data_split.training_set,
                                                                              subject_dictionary=subject_dictionary,
                                                                              feature_set=feature_set)
        testing_x, testing_y = ClassifierInputBuilder.get_sleep_wake_inputs(data_split.testing_set,
                                                                            subject_dictionary=subject_dictionary,
                                                                            feature_set=feature_set)

        return ClassifierService.run_single_data_split(training_x, training_y, testing_x, testing_y,
                                                       attributed_classifier)

    @staticmethod
    def run_single_data_split_three_class(data_split, attributed_classifier, subject_dictionary, feature_set):

        training_x, training_y = ClassifierInputBuilder.get_three_class_inputs(data_split.training_set,
                                                                               subject_dictionary=subject_dictionary,
                                                                               feature_set=feature_set)
        testing_x, testing_y = ClassifierInputBuilder.get_three_class_inputs(data_split.testing_set,
                                                                             subject_dictionary=subject_dictionary,
                                                                             feature_set=feature_set)
        return ClassifierService.run_single_data_split(training_x, training_y, testing_x, testing_y,
                                                       attributed_classifier, 'neg_log_loss')

    @staticmethod
    def run_single_data_split(training_x, training_y, testing_x, testing_y, attributed_classifier, scoring='roc_auc'):
        start_time = time.time()

        classifier = ClassifierService.train_classifier(training_x, training_y, attributed_classifier, scoring)
        class_probabilities = classifier.predict_proba(testing_x)

        raw_performance = RawPerformance(true_labels=testing_y, class_probabilities=class_probabilities)

        if Constants.VERBOSE:
            print('Completed data split in ' + str(time.time() - start_time))

        return raw_performance

    @staticmethod
    def train_classifier(training_x, training_y, attributed_classifier, scoring='roc_auc'):
        classifier = attributed_classifier.classifier

        classifier.class_weight = ClassifierService.get_class_weights(training_y)
        parameters = ParameterSearch.run_search(attributed_classifier, training_x, training_y, scoring=scoring)
        classifier.set_params(**parameters)
        classifier.fit(training_x, training_y)
        return classifier

    @staticmethod
    def get_class_weights(training_y):
        class_weights = class_weight.compute_class_weight('balanced',
                                                          classes=np.unique(training_y),
                                                          y=training_y)
        class_weight_dict = {}

        if len(class_weights) == 2:
            class_weight_dict = {0: class_weights[0], 1: class_weights[1]}

        if len(class_weights) == 3:
            class_weight_dict = {0: class_weights[0], 1: class_weights[1], 2: class_weights[2]}

        return class_weight_dict
