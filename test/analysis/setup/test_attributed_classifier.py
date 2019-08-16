from unittest import TestCase

from sklearn.neighbors import KNeighborsClassifier

from source.analysis.setup.attributed_classifier import AttributedClassifier


class TestAttributedClassifier(TestCase):
    def test_properties(self):
        classifier = KNeighborsClassifier()
        name = "k-Nearest Neighbors"
        attributed_classifier = AttributedClassifier(name=name, classifier=classifier)
        self.assertEqual(name, attributed_classifier.name)
        self.assertEqual(classifier, attributed_classifier.classifier)
