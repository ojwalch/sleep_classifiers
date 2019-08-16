from unittest import TestCase, mock
from unittest.mock import MagicMock

from sklearn.ensemble import RandomForestClassifier

from source.analysis.classification.classifier_service import ClassifierService
from source.analysis.setup.data_split import DataSplit
from source.analysis.performance.raw_performance import RawPerformance

import numpy as np

from test.test_helper import TestHelper


class TestClassifierService(TestCase):

    @mock.patch('source.analysis.classification.classifier_service.Pool')
    @mock.patch('source.analysis.classification.classifier_service.cpu_count')
    @mock.patch('source.analysis.classification.classifier_service.partial')
    def test_runs_training_and_testing_in_parallel(self, mock_partial, mock_cpu_count, mock_pool_constructor):
        expected_partial = "I am a partial"
        mock_partial.return_value = expected_partial

        mock_pool = MagicMock()
        mock_pool_constructor.return_value = mock_pool

        data_splits = [DataSplit(training_set=["subjectA", "subjectB", "subjectC"], testing_set=["subjectD"]),
                       DataSplit(training_set=["subjectA", "subjectB", "subjectD"], testing_set=["subjectC"])]
        classifier = RandomForestClassifier()
        subject_dictionary = {}
        feature_set = {}
        mock_pool.map.return_value = expected_pool_return = [3, 4]

        expected_number_of_cpus = 32
        mock_cpu_count.return_value = expected_number_of_cpus

        results = ClassifierService.run_sw(data_splits, classifier, subject_dictionary, feature_set)

        mock_partial.assert_called_once_with(ClassifierService.run_single_data_split_sw,
                                             attributed_classifier=classifier,
                                             subject_dictionary=subject_dictionary, feature_set=feature_set)

        mock_pool_constructor.assert_called_once_with(expected_number_of_cpus)
        mock_pool.map.assert_called_once_with(expected_partial, data_splits)
        self.assertEqual(expected_pool_return, results)

    @mock.patch.object(ClassifierService, 'get_class_weights')
    @mock.patch('source.analysis.classification.classifier_service.ParameterSearch')
    @mock.patch('source.analysis.classification.classifier_service.ClassifierInputBuilder.get_sleep_wake_inputs')
    def test_run_sleep_wake(self, mock_get_sleep_wake_inputs, mock_parameter_search, mock_class_weights):
        mock_classifier = MagicMock()
        mock_classifier.classifier.predict_proba.return_value = class_probabilities = np.array([[0.1, 0.9], [0, 1]])

        training_x = np.array([1, 2, 3, 4])
        training_y = np.array([0, 0, 0, 0])

        testing_x = np.array([5, 6, 7, 8])
        testing_y = np.array([0, 1, 0, 1])

        mock_get_sleep_wake_inputs.side_effect = [(training_x, training_y), (testing_x, testing_y)]

        mock_parameter_search.run_search.return_value = {}
        mock_class_weights.return_value = {0: 0.2, 1: 0.8}

        subject_dictionary = {}
        feature_set = {}

        data_split = DataSplit(training_set=["subjectA", "subjectB", "subjectC"],
                               testing_set=["subject1"])

        raw_performance = ClassifierService.run_single_data_split_sw(data_split, mock_classifier, subject_dictionary,
                                                                     feature_set)

        self.assertListEqual(testing_y.tolist(), raw_performance.true_labels.tolist())
        self.assertListEqual(class_probabilities.tolist(), raw_performance.class_probabilities.tolist())

        mock_class_weights.assert_called_once_with(training_y)
        mock_parameter_search.run_search.assert_called_once_with(mock_classifier, training_x, training_y,
                                                                 scoring='roc_auc')
        mock_classifier.classifier.fit.assert_called_once_with(training_x, training_y)
        mock_classifier.classifier.predict_proba.assert_called_once_with(testing_x)
