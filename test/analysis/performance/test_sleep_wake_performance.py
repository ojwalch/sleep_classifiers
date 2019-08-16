from unittest import TestCase

from source.analysis.performance.epoch_performance import SleepWakePerformance


class TestSleepWakePerformance(TestCase):

    def test_has_properties(self):
        accuracy = 0.9
        wake_correct = 0.2
        sleep_correct = 0.99
        kappa = 0.5
        auc = 0.8
        sleep_predictive_value = 0.3
        wake_predictive_value = 0.2
        sleep_wake_performance = SleepWakePerformance(accuracy=accuracy,
                                                      wake_correct=wake_correct,
                                                      sleep_correct=sleep_correct,
                                                      kappa=kappa,
                                                      auc=auc,
                                                      sleep_predictive_value=sleep_predictive_value,
                                                      wake_predictive_value=wake_predictive_value)
        self.assertEqual(sleep_wake_performance.accuracy, accuracy)
        self.assertEqual(sleep_wake_performance.wake_correct, wake_correct)
        self.assertEqual(sleep_wake_performance.sleep_correct, sleep_correct)
        self.assertEqual(sleep_wake_performance.kappa, kappa)
        self.assertEqual(sleep_wake_performance.auc, auc)
        self.assertEqual(sleep_wake_performance.sleep_predictive_value, sleep_predictive_value)
        self.assertEqual(sleep_wake_performance.wake_predictive_value, wake_predictive_value)


