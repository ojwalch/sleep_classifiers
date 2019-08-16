from unittest import TestCase, mock

from mock import MagicMock
import numpy as np

from source.mesa.mesa_heart_rate_service import MesaHeartRateService


class TestMesaHeartRateService(TestCase):
    @mock.patch('source.mesa.mesa_heart_rate_service.np')
    @mock.patch('source.mesa.mesa_heart_rate_service.pyedflib')
    @mock.patch('source.mesa.mesa_heart_rate_service.utils')
    def test_load_raw(self, mock_utils, mock_pyedflib, mock_np):
        file_id = '4'
        mock_utils.get_project_root.return_value = project_root = 'project_root'
        expected_edf_file_location = project_root + '/data/mesa/polysomnography/edfs/mesa-sleep-' + file_id + '.edf'

        mock_pyedflib.EdfReader.return_value = mock_edf_file = MagicMock()
        mock_edf_file.getSignalLabels.return_value = ["A", "B", "C", "D"]
        mock_edf_file.getSampleFrequencies.return_value = [1, 2, 3, 4]
        mock_edf_file.readSignal.return_value = heart_rate_signal = [10, 20, 30]
        mock_np.array.return_value = np.array([0, 1, 2])
        mock_np.transpose.return_value = transposed_value = np.array([[1, 2], [3, 4]])
        mock_np.vstack.return_value = vstack_return_value = np.array([[11, 21, 31], [4, 5, 6]])
        mock_utils.remove_nans.return_value = remove_nan_return_value = np.array([[0, 1], [2, 3]])

        heart_rate_collection = MesaHeartRateService.load_raw(file_id)

        mock_utils.get_project_root.assert_called_once()
        mock_pyedflib.EdfReader.assert_called_once_with(expected_edf_file_location)

        mock_edf_file.getSignalLabels.assert_called_once()
        mock_edf_file.getSampleFrequencies.assert_called_once()
        mock_edf_file.readSignal.assert_called_once_with(2)
        mock_np.array.assert_called_once_with(range(0, len(heart_rate_signal)))
        mock_np.transpose.assert_called_once_with(vstack_return_value)
        mock_utils.remove_nans.assert_called_once_with(transposed_value)

        self.assertListEqual(remove_nan_return_value.tolist(), heart_rate_collection.data.tolist())
