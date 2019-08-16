from unittest import TestCase, mock
from unittest.mock import MagicMock
import numpy as np
from source.constants import Constants
from source.preprocessing.time.time_based_feature_service import TimeBasedFeatureService


class TestTimeBasedFeatureSetService(TestCase):

    @mock.patch('source.preprocessing.time.time_based_feature_service.pd')
    def test_load_time(self, mock_pd):
        mock_pd.read_csv.return_value = mock_return = MagicMock()
        mock_return.values = expected_return = np.array([1, 2, 3, 4, 5])
        actual_returned_value = TimeBasedFeatureService.load_time("subjectA")

        self.assertListEqual(expected_return.tolist(), actual_returned_value.tolist())
        mock_pd.read_csv.assert_called_once_with(str(TimeBasedFeatureService.get_path_for_time("subjectA")))

    def test_get_time_path(self):
        expected_path = Constants.FEATURE_FILE_PATH.joinpath("subjectA" + '_time_feature.out')

        self.assertEqual(expected_path, TimeBasedFeatureService.get_path_for_time("subjectA"))

    @mock.patch('source.preprocessing.time.time_based_feature_service.np')
    def test_write_time(self, mock_np):
        feature_to_write = np.array([1, 2, 3, 4])
        subject_id = "subjectA"
        TimeBasedFeatureService.write_time(subject_id, feature_to_write)

        mock_np.savetxt.assert_called_once_with(TimeBasedFeatureService.get_path_for_time(subject_id), feature_to_write,
                                                fmt='%f')

    @mock.patch('source.preprocessing.time.time_based_feature_service.pd')
    def test_circadian_model_time(self, mock_pd):
        mock_pd.read_csv.return_value = mock_return = MagicMock()
        mock_return.values = expected_return = np.array([1, 2, 3, 4, 5])
        actual_returned_value = TimeBasedFeatureService.load_circadian_model("subjectA")

        self.assertListEqual(expected_return.tolist(), actual_returned_value.tolist())
        mock_pd.read_csv.assert_called_once_with(
            str(TimeBasedFeatureService.get_path_for_circadian_model("subjectA")), delimiter=' ')

    def test_get_circadian_model_path(self):
        expected_path = Constants.FEATURE_FILE_PATH.joinpath("subjectA" + '_circadian_feature.out')
        self.assertEqual(expected_path, TimeBasedFeatureService.get_path_for_circadian_model("subjectA"))

    @mock.patch('source.preprocessing.time.time_based_feature_service.np')
    def test_write_circadian_model(self, mock_np):
        feature_to_write = np.array([1, 2, 3, 4])
        subject_id = "subjectA"
        TimeBasedFeatureService.write_circadian_model(subject_id, feature_to_write)

        mock_np.savetxt.assert_called_once_with(TimeBasedFeatureService.get_path_for_circadian_model(subject_id),
                                                feature_to_write,
                                                fmt='%f')

    @mock.patch('source.preprocessing.time.time_based_feature_service.pd')
    def test_load_cosine(self, mock_pd):
        mock_pd.read_csv.return_value = mock_return = MagicMock()
        mock_return.values = expected_return = np.array([1, 2, 3, 4, 5])
        actual_returned_value = TimeBasedFeatureService.load_cosine("subjectA")

        self.assertListEqual(expected_return.tolist(), actual_returned_value.tolist())
        mock_pd.read_csv.assert_called_once_with(str(TimeBasedFeatureService.get_path_for_cosine("subjectA")))

    def test_get_cosine_path(self):
        expected_path = Constants.FEATURE_FILE_PATH.joinpath("subjectA" + '_cosine_feature.out')

        self.assertEqual(expected_path, TimeBasedFeatureService.get_path_for_cosine("subjectA"))

    @mock.patch('source.preprocessing.time.time_based_feature_service.np')
    def test_write_cosine(self, mock_np):
        feature_to_write = np.array([1, 2, 3, 4])
        subject_id = "subjectA"
        TimeBasedFeatureService.write_cosine(subject_id, feature_to_write)

        mock_np.savetxt.assert_called_once_with(TimeBasedFeatureService.get_path_for_cosine(subject_id),
                                                feature_to_write,
                                                fmt='%f')
