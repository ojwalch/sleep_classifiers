from unittest import TestCase

from source.analysis.performance.performance_summarizer import PerformanceSummarizer
from source.analysis.performance.epoch_performance import SleepWakePerformance
from test_helper import TestHelper


class TestPerformanceSummarizer(TestCase):

    def test_averages(self):
        sleep_wake_performance_1 = SleepWakePerformance(accuracy=0,
                                                        wake_correct=0.1,
                                                        sleep_correct=0.2,
                                                        auc=0.3,
                                                        kappa=0.4,
                                                        wake_predictive_value=0.5,
                                                        sleep_predictive_value=0.6)

        sleep_wake_performance_2 = SleepWakePerformance(accuracy=0.2,
                                                        wake_correct=0.3,
                                                        sleep_correct=0.4,
                                                        auc=0.5,
                                                        kappa=0.6,
                                                        wake_predictive_value=0.7,
                                                        sleep_predictive_value=0.8)

        expected_performance = SleepWakePerformance(accuracy=0.1,
                                                    wake_correct=0.2,
                                                    sleep_correct=0.3,
                                                    auc=0.4,
                                                    kappa=0.5,
                                                    wake_predictive_value=0.6,
                                                    sleep_predictive_value=0.7)

        performance_list = [sleep_wake_performance_1, sleep_wake_performance_2]

        actual_averaged_performance = PerformanceSummarizer.average(performance_list)
        TestHelper.assert_models_equal(self, expected_performance, actual_averaged_performance)
