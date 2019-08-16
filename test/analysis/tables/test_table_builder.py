from unittest import TestCase, mock
from unittest.mock import call

import numpy as np
from sklearn.linear_model import LogisticRegression

from source.analysis.classification.classifier_summary import ClassifierSummary
from source.analysis.performance.performance_summarizer import PerformanceSummarizer
from source.analysis.performance.raw_performance import RawPerformance
from source.analysis.performance.epoch_performance import SleepWakePerformance
from source.analysis.setup.attributed_classifier import AttributedClassifier
from source.analysis.setup.feature_type import FeatureType
from source.analysis.tables.table_builder import TableBuilder


class TestTableBuilder(TestCase):

    @mock.patch.object(PerformanceSummarizer, 'summarize_thresholds')
    @mock.patch('source.analysis.tables.table_builder.print')
    def test_print_table_sw(self, mock_print, mock_summarize_thresholds):
        first_raw_performances = [
            RawPerformance(true_labels=np.array([0, 1]),
                           class_probabilities=np.array([[0, 1], [1, 0]])),
            RawPerformance(true_labels=np.array([0, 1]),
                           class_probabilities=np.array([[0.2, 0.8], [0.1, 0.9]]))]

        second_raw_performances = [
            RawPerformance(true_labels=np.array([1, 1]),
                           class_probabilities=np.array([[0, 1], [1, 0]])),
            RawPerformance(true_labels=np.array([0, 0]),
                           class_probabilities=np.array([[0.2, 0.8], [0.1, 0.9]]))]

        performance_dictionary = {tuple([FeatureType.count, FeatureType.heart_rate]): first_raw_performances,
                                  tuple([FeatureType.count]): second_raw_performances}

        attributed_classifier = AttributedClassifier(name="Logistic Regression", classifier=LogisticRegression())
        classifier_summary = ClassifierSummary(attributed_classifier=attributed_classifier,
                                               performance_dictionary=performance_dictionary)

        first_performance = SleepWakePerformance(accuracy=0, wake_correct=0, sleep_correct=0, kappa=0, auc=0,
                                                 sleep_predictive_value=0,
                                                 wake_predictive_value=0)
        second_performance = SleepWakePerformance(accuracy=1, wake_correct=1, sleep_correct=1, kappa=1, auc=1,
                                                  sleep_predictive_value=1,
                                                  wake_predictive_value=1)
        third_performance = SleepWakePerformance(accuracy=0.5, wake_correct=0.5, sleep_correct=0.5, kappa=0.5, auc=0.5,
                                                 sleep_predictive_value=0.5,
                                                 wake_predictive_value=0.5)
        fourth_performance = SleepWakePerformance(accuracy=0.2, wake_correct=0.2, sleep_correct=0.2, kappa=0.2, auc=0.2,
                                                  sleep_predictive_value=0.2,
                                                  wake_predictive_value=0.2)

        mock_summarize_thresholds.side_effect = [([0.3, 0.7], [first_performance, second_performance]),
                                                 ([0.2, 0.8], [third_performance, fourth_performance])]
        TableBuilder.print_table_sw(classifier_summary)

        frontmatter = '\\begin{table}  \\caption{Sleep/wake differentiation performance by Logistic Regression ' \
                      + 'across different feature inputs in the Apple Watch (PPG, MEMS) dataset} ' \
                        '\\begin{tabular}{l*{5}{c}} & Accuracy & Wake correct (specificity) ' \
                        '& Sleep correct (sensitivity) & $\\kappa$ & AUC \\\\ '
        header_line_1 = '\\hline Motion, HR &'
        header_line_2 = '\\hline Motion only &'

        results_line_1 = '& ' + str(first_performance.accuracy) + ' & ' + str(
            first_performance.wake_correct) + ' & ' + str(
            first_performance.sleep_correct) + ' & ' + str(first_performance.kappa) + ' &   \\\\'
        results_line_2 = '& ' + str(second_performance.accuracy) + ' & ' + str(
            second_performance.wake_correct) + ' & ' + str(
            second_performance.sleep_correct) + ' & ' + str(second_performance.kappa) + ' &   \\\\'
        results_line_3 = '& ' + str(third_performance.accuracy) + ' & ' + str(
            third_performance.wake_correct) + ' & ' + str(
            third_performance.sleep_correct) + ' & ' + str(third_performance.kappa) + ' &   \\\\'
        results_line_4 = str(fourth_performance.accuracy) + ' & ' + str(
            fourth_performance.wake_correct) + ' & ' + str(
            fourth_performance.sleep_correct) + ' & ' + str(fourth_performance.kappa) + ' & ' + str(
            fourth_performance.auc) + '  \\\\'

        backmatter = '\\hline \\end{tabular}  \\label{tab:' \
                     + attributed_classifier.name[0:4] \
                     + 'params} \\small \\vspace{.2cm} ' \
                       '\\caption*{Fraction of wake correct, fraction of sleep correct, accuracy, ' \
                       '$\\kappa$, and AUC for sleep-wake predictions of Logistic Regression' \
                       ' with use of motion, HR, clock proxy, or combination of features. PPG, ' \
                       'photoplethysmography; MEMS, microelectromechanical systems; HR, heart rate; ' \
                       'AUC, area under the curve.} \\end{table}'

        mock_print.assert_has_calls([call(frontmatter),
                                     call(header_line_1),
                                     call(results_line_1),
                                     call(results_line_2),
                                     call(header_line_2),
                                     call(results_line_3),
                                     call(results_line_4),
                                     call(backmatter)])

    @mock.patch.object(PerformanceSummarizer, 'average')
    @mock.patch('source.analysis.performance.performance_summarizer.PerformanceBuilder')
    def test_summarize_thresholds(self, mock_performance_builder, mock_performance_summarizer_average):
        raw_performances = [RawPerformance(true_labels=np.array([0, 1]),
                                           class_probabilities=np.array([[0, 1], [1, 0]])),
                            RawPerformance(true_labels=np.array([0, 1]),
                                           class_probabilities=np.array([[0.2, 0.8], [0.1, 0.9]]))]

        mock_performance_builder.build_with_true_positive_rate_threshold.side_effect = ['return1',
                                                                                        'return2',
                                                                                        'return3',
                                                                                        'return4',
                                                                                        'return5',
                                                                                        'return6',
                                                                                        'return7',
                                                                                        'return8'
                                                                                        ]

        mock_performance_summarizer_average.side_effect = averages = ['average1',
                                                                      'average2',
                                                                      'average3',
                                                                      'average4']

        thresholds, returned_averages = PerformanceSummarizer.summarize_thresholds(raw_performances)

        mock_performance_builder.build_with_true_positive_rate_threshold.assert_has_calls(
            [call(raw_performances[0], 0.8),
             call(raw_performances[1], 0.8),
             call(raw_performances[0], 0.9),
             call(raw_performances[1], 0.9),
             call(raw_performances[0], 0.93),
             call(raw_performances[1], 0.93),
             call(raw_performances[0], 0.95),
             call(raw_performances[1], 0.95)])

        mock_performance_summarizer_average.assert_has_calls([call(['return1', 'return2']),
                                                              call(['return3', 'return4']),
                                                              call(['return5', 'return6']),
                                                              call(['return7', 'return8'])
                                                              ])

        self.assertListEqual([0.8, 0.9, 0.93, 0.95], thresholds)
        self.assertListEqual(averages, returned_averages)
