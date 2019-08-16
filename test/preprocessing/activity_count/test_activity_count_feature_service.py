from unittest import TestCase, mock
from unittest.mock import MagicMock, call

import numpy as np
from source.constants import Constants
from source.preprocessing.activity_count.activity_count_feature_service import ActivityCountFeatureService
from source.preprocessing.activity_count.activity_count_service import ActivityCountService
from source.preprocessing.epoch import Epoch
from source.preprocessing.activity_count.activity_count_collection import ActivityCountCollection


class TestActivityCountFeatureService(TestCase):

    @mock.patch('source.preprocessing.activity_count.activity_count_feature_service.pd')
    def test_load(self, mock_pd):
        mock_pd.read_csv.return_value = mock_return = MagicMock()
        mock_return.values = expected_return = np.array([1, 2, 3, 4, 5])
        actual_returned_value = ActivityCountFeatureService.load("subjectA")

        self.assertListEqual(expected_return.tolist(), actual_returned_value.tolist())
        mock_pd.read_csv.assert_called_once_with(str(ActivityCountFeatureService.get_path("subjectA")))

    def test_get_path(self):
        expected_path = Constants.FEATURE_FILE_PATH.joinpath("subjectA" + '_count_feature.out')

        self.assertEqual(expected_path, ActivityCountFeatureService.get_path("subjectA"))

    @mock.patch('source.preprocessing.activity_count.activity_count_feature_service.np')
    def test_write(self, mock_np):
        feature_to_write = np.array([1, 2, 3, 4])
        subject_id = "subjectA"
        ActivityCountFeatureService.write(subject_id, feature_to_write)

        mock_np.savetxt.assert_called_once_with(ActivityCountFeatureService.get_path(subject_id), feature_to_write,
                                                fmt='%f')

    def test_get_window(self):
        timestamps = np.array([-2000, 22, 32, 50, 60, 800, 1000])
        epoch = Epoch(timestamp=55, index=120)
        expected_indices_in_range = np.array([1, 2, 3, 4])

        actual_indices_in_range = ActivityCountFeatureService.get_window(timestamps, epoch)

        self.assertEqual(expected_indices_in_range.tolist(), actual_indices_in_range.tolist())

    @mock.patch.object(ActivityCountFeatureService, 'get_feature')
    @mock.patch.object(ActivityCountService, 'load_cropped')
    def test_build_feature_array(self, mock_load_cropped, mock_get_feature):
        subject_id = "subjectA"
        data = np.array(
            [[1, 10], [10, 220], [20, 0], [40, 500], [70, 200], [90, 0], [100, 0], [120, 4]])
        activity_count_collection = ActivityCountCollection(subject_id=subject_id, data=data)
        mock_load_cropped.return_value = activity_count_collection
        expected_features = [np.array([0.1]), np.array([0.2])]
        mock_get_feature.side_effect = expected_features
        expected_feature_array = np.array(expected_features)

        valid_epochs = [Epoch(timestamp=4, index=1), Epoch(timestamp=50, index=2)]

        returned_feature_array = ActivityCountFeatureService.build(subject_id, valid_epochs)

        self.assertEqual(expected_feature_array.tolist(), returned_feature_array.tolist())

    def test_interpolate(self):
        subject_id = "subjectA"
        data = np.array([[1, 0], [10, 9]])
        activity_count_collection = ActivityCountCollection(subject_id=subject_id, data=data)

        interpolated_timestamps, interpolated_counts = ActivityCountFeatureService.interpolate(
            activity_count_collection)

        self.assertListEqual([0, 1, 2, 3, 4, 5, 6, 7, 8], interpolated_counts.tolist())
        self.assertListEqual([1, 2, 3, 4, 5, 6, 7, 8, 9], interpolated_timestamps.tolist())