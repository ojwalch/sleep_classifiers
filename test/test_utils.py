from unittest import TestCase, mock

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from source import utils
from source.analysis.setup.attributed_classifier import AttributedClassifier
from source.analysis.setup.feature_type import FeatureType
from test.test_helper import TestHelper

import numpy as np


class TestUtils(TestCase):

    @mock.patch('source.utils.Path', autospec=True)
    def test_gets_root_directory(self, mock_path):
        path_to_return = "path/to/return"

        mock_path_module = mock_path.return_value
        mock_parent = mock_path_module.parent
        mock_parent.parent = path_to_return

        project_root = utils.get_project_root()
        self.assertEqual(project_root, path_to_return)

    def test_get_classifiers(self):
        all_classifiers = utils.get_classifiers()
        TestHelper.assert_models_equal(self, all_classifiers[0], AttributedClassifier(name='Random Forest',
                                                                                      classifier=RandomForestClassifier(
                                                                                          n_estimators=500,
                                                                                          max_features=1.0,
                                                                                          max_depth=10,
                                                                                          min_samples_split=10,
                                                                                          min_samples_leaf=1)))

        TestHelper.assert_models_equal(self, all_classifiers[1], AttributedClassifier(name='Logistic Regression',
                                                                                      classifier=LogisticRegression(
                                                                                          penalty='l1',
                                                                                          solver='liblinear',
                                                                                          verbose=0)))

        TestHelper.assert_models_equal(self, all_classifiers[2], AttributedClassifier(name='k-Nearest Neighbors',
                                                                                      classifier=
                                                                                      KNeighborsClassifier()))

        TestHelper.assert_models_equal(self, all_classifiers[3], AttributedClassifier(name='Neural Net',
                                                                                      classifier=MLPClassifier(
                                                                                          activation='relu',
                                                                                          hidden_layer_sizes=(
                                                                                              30, 30, 30),
                                                                                          max_iter=1000,
                                                                                          alpha=0.01)))

    def test_get_base_feature_sets(self):
        feature_sets = utils.get_base_feature_sets()
        self.assertListEqual([[FeatureType.count],
                              [FeatureType.heart_rate],
                              [FeatureType.count, FeatureType.heart_rate],
                              [FeatureType.count, FeatureType.heart_rate, FeatureType.circadian_model]],
                             feature_sets)

    def test_remove_repeats(self):
        array_with_repeats = np.array([[1, 2, 3], [1, 2, 3], [4, 5, 6]])
        array_no_repeats = np.array([[1, 2, 3], [4, 5, 6]])

        returned_no_repeats = utils.remove_repeats(array_with_repeats)

        self.assertEqual(array_no_repeats.tolist(), returned_no_repeats.tolist())
