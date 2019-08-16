from unittest import TestCase, mock
from unittest.mock import MagicMock

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from source.analysis.classification.parameter_search import ParameterSearch
from source.analysis.setup.attributed_classifier import AttributedClassifier

import numpy as np


class TestParameterSearch(TestCase):

    @mock.patch('source.analysis.classification.parameter_search.GridSearchCV')
    def test_run_best_parameter_search(self, mock_grid_search):
        attributed_classifier = AttributedClassifier(name="Logistic Regression", classifier=LogisticRegression())
        training_x = np.array([1, 2, 3])
        training_y = np.array([4, 5, 6])
        expected_parameter_range = ParameterSearch.parameter_dictionary[attributed_classifier.name]
        scoring = 'roc_auc'
        mock_parameter_search_classifier = MagicMock()
        mock_grid_search.return_value = mock_parameter_search_classifier
        mock_parameter_search_classifier.best_params_ = expected_parameters = {'parameter': 'value'}

        returned_parameters = ParameterSearch.run_search(attributed_classifier, training_x,
                                                         training_y, scoring=scoring)
        mock_grid_search.assert_called_once_with(attributed_classifier.classifier, expected_parameter_range,
                                                 scoring=scoring,
                                                 iid=False,
                                                 cv=3)
        mock_parameter_search_classifier.fit.assert_called_once_with(training_x, training_y)
        self.assertDictEqual(expected_parameters, returned_parameters)
