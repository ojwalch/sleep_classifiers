from source.analysis.setup.attributed_classifier import AttributedClassifier


class ClassifierSummary(object):
    def __init__(self, attributed_classifier: AttributedClassifier, performance_dictionary):
        self.attributed_classifier = attributed_classifier
        self.performance_dictionary = performance_dictionary
