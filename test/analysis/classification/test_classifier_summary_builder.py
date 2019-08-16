from unittest import TestCase, mock
from unittest.mock import call

from sklearn.linear_model import LogisticRegression

from source.analysis.classification.classifier_summary_builder import SleepWakeClassifierSummaryBuilder
from source.analysis.performance.raw_performance import RawPerformance
from source.analysis.setup.attributed_classifier import AttributedClassifier
from source.analysis.setup.data_split import DataSplit
from source.analysis.setup.feature_type import FeatureType

import numpy as np


class TestClassifierSummaryBuilder(TestCase):

    @mock.patch('source.analysis.classification.classifier_summary_builder.SubjectBuilder')
    @mock.patch('source.analysis.classification.classifier_summary_builder.ClassifierService')
    @mock.patch('source.analysis.classification.classifier_summary_builder.TrainTestSplitter')
    def test_build_summary_by_fraction(self, mock_train_test_splitter, mock_classifier_service, mock_subject_builder):
        attributed_classifier = AttributedClassifier(name="Logistic Regression", classifier=LogisticRegression())
        feature_sets = [[FeatureType.cosine, FeatureType.circadian_model], [FeatureType.count]]
        number_of_splits = 5
        test_fraction = 0.3

        mock_subject_builder.get_all_subject_ids.return_value = subject_ids = ["subjectA", "subjectB"]
        mock_subject_builder.get_subject_dictionary.return_value = subject_dictionary = {"subjectA": [], "subjectB": []}

        mock_train_test_splitter.by_fraction.return_value = expected_data_splits = [
            DataSplit(training_set="subjectA", testing_set="subjectB")]

        mock_classifier_service.run_sw.side_effect = raw_performance_arrays = [
            [RawPerformance(true_labels=np.array([1, 2]),
                            class_probabilities=np.array([3, 4])),
             RawPerformance(true_labels=np.array([0, 1]),
                            class_probabilities=np.array([2, 3]))
             ],
            [RawPerformance(true_labels=np.array([1, 1]),
                            class_probabilities=np.array([4, 4])),
             RawPerformance(true_labels=np.array([0, 0]),
                            class_probabilities=np.array([2, 2]))
             ]
        ]

        returned_summary = SleepWakeClassifierSummaryBuilder.build_monte_carlo(attributed_classifier, feature_sets,
                                                                               number_of_splits)

        mock_subject_builder.get_all_subject_ids.assert_called_once_with()
        mock_subject_builder.get_subject_dictionary.assert_called_once_with()
        mock_train_test_splitter.by_fraction.assert_called_once_with(subject_ids, test_fraction=test_fraction,
                                                                     number_of_splits=number_of_splits)

        mock_classifier_service.run_sw.assert_has_calls([call(expected_data_splits,
                                                              attributed_classifier,
                                                              subject_dictionary,
                                                              feature_sets[0]
                                                              ),
                                                         call(expected_data_splits,
                                                              attributed_classifier,
                                                              subject_dictionary,
                                                              feature_sets[1]
                                                              )])
        self.assertEqual(returned_summary.attributed_classifier, attributed_classifier)
        self.assertEqual(returned_summary.performance_dictionary[tuple(feature_sets[0])], raw_performance_arrays[0])
        self.assertEqual(returned_summary.performance_dictionary[tuple(feature_sets[1])], raw_performance_arrays[1])

    @mock.patch('source.analysis.classification.classifier_summary_builder.SubjectBuilder')
    @mock.patch('source.analysis.classification.classifier_summary_builder.ClassifierService')
    @mock.patch('source.analysis.classification.classifier_summary_builder.TrainTestSplitter')
    def test_leave_one_out(self, mock_train_test_splitter, mock_classifier_service, mock_subject_builder):
        attributed_classifier = AttributedClassifier(name="Logistic Regression", classifier=LogisticRegression())
        feature_sets = [[FeatureType.cosine, FeatureType.circadian_model], [FeatureType.count]]

        mock_subject_builder.get_all_subject_ids.return_value = subject_ids = ["subjectA", "subjectB"]
        mock_subject_builder.get_subject_dictionary.return_value = subject_dictionary = {"subjectA": [], "subjectB": []}

        mock_train_test_splitter.leave_one_out.return_value = expected_data_splits = [
            DataSplit(training_set="subjectA", testing_set="subjectB")]

        mock_classifier_service.run_sw.side_effect = raw_performance_arrays = [
            [RawPerformance(true_labels=np.array([1, 2]),
                            class_probabilities=np.array([3, 4])),
             RawPerformance(true_labels=np.array([0, 1]),
                            class_probabilities=np.array([2, 3]))
             ],
            [RawPerformance(true_labels=np.array([1, 1]),
                            class_probabilities=np.array([4, 4])),
             RawPerformance(true_labels=np.array([0, 0]),
                            class_probabilities=np.array([2, 2]))
             ]
        ]

        returned_summary = SleepWakeClassifierSummaryBuilder.build_leave_one_out(attributed_classifier, feature_sets)

        mock_subject_builder.get_all_subject_ids.assert_called_once_with()
        mock_subject_builder.get_subject_dictionary.assert_called_once_with()
        mock_train_test_splitter.leave_one_out.assert_called_once_with(subject_ids)

        mock_classifier_service.run_sw.assert_has_calls([call(expected_data_splits,
                                                              attributed_classifier,
                                                              subject_dictionary,
                                                              feature_sets[0]
                                                              ),
                                                         call(expected_data_splits,
                                                              attributed_classifier,
                                                              subject_dictionary,
                                                              feature_sets[1]
                                                              )])
        self.assertEqual(returned_summary.attributed_classifier, attributed_classifier)
        self.assertEqual(returned_summary.performance_dictionary[tuple(feature_sets[0])], raw_performance_arrays[0])
        self.assertEqual(returned_summary.performance_dictionary[tuple(feature_sets[1])], raw_performance_arrays[1])
