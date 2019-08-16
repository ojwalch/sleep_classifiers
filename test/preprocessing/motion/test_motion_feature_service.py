from unittest import TestCase, mock
from unittest.mock import MagicMock
import numpy as np
from source.constants import Constants
from source.preprocessing.motion.motion_feature_service import MotionFeatureService


class TestMotionFeatureService(TestCase):

    @mock.patch('source.preprocessing.motion.motion_feature_service.pd')
    def test_load(self, mock_pd):
        mock_pd.read_csv.return_value = mock_return = MagicMock()
        mock_return.values = expected_return = np.array([1, 2, 3, 4, 5])
        actual_returned_value = MotionFeatureService.load("subjectA")

        self.assertListEqual(expected_return.tolist(), actual_returned_value.tolist())
        mock_pd.read_csv.assert_called_once_with(str(MotionFeatureService.get_path("subjectA")))

    def test_get_path(self):
        expected_path = Constants.FEATURE_FILE_PATH.joinpath("subjectA" + '_motion_feature.out')

        self.assertEqual(expected_path, MotionFeatureService.get_path("subjectA"))

    @mock.patch('source.preprocessing.motion.motion_feature_service.np')
    def test_write(self, mock_np):
        feature_to_write = np.array([1, 2, 3, 4])
        subject_id = "subjectA"
        MotionFeatureService.write(subject_id, feature_to_write)

        mock_np.savetxt.assert_called_once_with(MotionFeatureService.get_path(subject_id), feature_to_write,
                                                fmt='%f')
