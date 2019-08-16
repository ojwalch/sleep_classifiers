class SleepWakePerformance(object):
    def __init__(self, accuracy, wake_correct, sleep_correct, kappa, auc, sleep_predictive_value,
                 wake_predictive_value):
        self.accuracy = accuracy
        self.wake_correct = wake_correct
        self.sleep_correct = sleep_correct
        self.kappa = kappa
        self.auc = auc
        self.wake_predictive_value = wake_predictive_value
        self.sleep_predictive_value = sleep_predictive_value


class ThreeClassPerformance(object):
    def __init__(self, accuracy, wake_correct, rem_correct, nrem_correct, kappa):
        self.accuracy = accuracy
        self.wake_correct = wake_correct
        self.rem_correct = rem_correct
        self.nrem_correct = nrem_correct
        self.kappa = kappa
