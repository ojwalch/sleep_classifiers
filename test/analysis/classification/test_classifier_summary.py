from unittest import TestCase

from sklearn.linear_model import LogisticRegression

from source.analysis.classification.classifier_summary import ClassifierSummary
from source.analysis.performance.raw_performance import RawPerformance
from source.analysis.setup.attributed_classifier import AttributedClassifier
from source.analysis.setup.feature_type import FeatureType

import numpy as np

from test.test_helper import TestHelper


class TestClassifierSummary(TestCase):
    def test_properties(self):
        attributed_classifier = AttributedClassifier(name="Logistic Regression", classifier=LogisticRegression())

        performance_dictionary = {(FeatureType.count, FeatureType.cosine):
                                      [RawPerformance(true_labels=np.array([1, 0, 1, 0]),
                                                      class_probabilities=np.array([0.1, 0.9]))],
                                  FeatureType.count:
                                      [RawPerformance(true_labels=np.array([0, 0, 1, 0]),
                                                      class_probabilities=np.array([0.9, 0.1]))]
                                  }

        classifier_summary = ClassifierSummary(attributed_classifier=attributed_classifier,
                                               performance_dictionary=performance_dictionary)

        TestHelper.assert_models_equal(self, attributed_classifier, classifier_summary.attributed_classifier)
        self.assertDictEqual(performance_dictionary, classifier_summary.performance_dictionary)