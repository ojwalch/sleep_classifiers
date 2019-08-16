from unittest import TestCase, mock
from unittest.mock import MagicMock
import numpy as np
from source.constants import Constants
from source.preprocessing.epoch import Epoch
from source.preprocessing.heart_rate.heart_rate_collection import HeartRateCollection
from source.preprocessing.heart_rate.heart_rate_feature_service import HeartRateFeatureService


class TestHeartRateFeatureService(TestCase):

    @mock.patch('source.preprocessing.heart_rate.heart_rate_feature_service.pd')
    def test_load(self, mock_pd):
        mock_pd.read_csv.return_value = mock_return = MagicMock()
        mock_return.values = expected_return = np.array([1, 2, 3, 4, 5])
        actual_returned_value = HeartRateFeatureService.load("subjectA")

        self.assertListEqual(expected_return.tolist(), actual_returned_value.tolist())
        mock_pd.read_csv.assert_called_once_with(str(HeartRateFeatureService.get_path("subjectA")), delimiter=' ')

    def test_get_path(self):
        expected_path = Constants.FEATURE_FILE_PATH.joinpath("subjectA" + '_hr_feature.out')

        self.assertEqual(expected_path, HeartRateFeatureService.get_path("subjectA"))

    @mock.patch('source.preprocessing.heart_rate.heart_rate_feature_service.np')
    def test_write(self, mock_np):
        feature_to_write = np.array([1, 2, 3, 4])
        subject_id = "subjectA"
        HeartRateFeatureService.write(subject_id, feature_to_write)

        mock_np.savetxt.assert_called_once_with(HeartRateFeatureService.get_path(subject_id), feature_to_write,
                                                fmt='%f')

    def test_get_window(self):
        timestamps = np.array([-1000, -500, 32, 50, 60, 800, 1000])
        epoch = Epoch(timestamp=55, index=120)
        expected_indices_in_range = np.array([2, 3, 4])

        actual_indices_in_range = HeartRateFeatureService.get_window(timestamps, epoch)

        self.assertEqual(expected_indices_in_range.tolist(), actual_indices_in_range.tolist())

    @mock.patch.object(HeartRateFeatureService, 'get_feature')
    @mock.patch('source.preprocessing.heart_rate.heart_rate_feature_service.HeartRateService')
    def test_build_feature_array(self, mock_heart_rate_service, mock_get_feature):
        subject_id = "subjectA"
        data = np.array(
            [[1, 10], [10, 220], [20, 0], [40, 500], [70, 200], [90, 0], [100, 0], [400, 4]])
        motion_collection = HeartRateCollection(subject_id=subject_id, data=data)
        mock_heart_rate_service.load_cropped.return_value = motion_collection
        expected_features = [np.array([0.1]), np.array([0.2])]
        mock_get_feature.side_effect = expected_features
        expected_feature_array = np.array(expected_features)

        valid_epochs = [Epoch(timestamp=4, index=1), Epoch(timestamp=50, index=2)]

        returned_feature_array = HeartRateFeatureService.build(subject_id, valid_epochs)

        self.assertEqual(expected_feature_array.tolist(), returned_feature_array.tolist())
