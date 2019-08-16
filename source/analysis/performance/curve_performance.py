from numpy.core.multiarray import ndarray


class ROCPerformance(object):

    def __init__(self, false_positive_rates: ndarray, true_positive_rates: ndarray):
        self.false_positive_rates = false_positive_rates
        self.true_positive_rates = true_positive_rates


class PrecisionRecallPerformance(object):

    def __init__(self, recalls: ndarray, precisions: ndarray):
        self.recalls = recalls
        self.precisions = precisions
