from unittest import TestCase, mock

from mock import mock_open, MagicMock

from source.preprocessing.epoch import Epoch
from source.preprocessing.psg.compumedics_processor import CompumedicsProcessor
from source.preprocessing.psg.psg_file_type import PSGFileType
from source.preprocessing.psg.report_summary import ReportSummary
from source.preprocessing.psg.stage_item import StageItem
from source.sleep_stage import SleepStage
from test.test_helper import TestHelper


class TestCompumedicsProcessor(TestCase):
    @mock.patch("source.preprocessing.psg.compumedics_processor.TimeService")
    @mock.patch("builtins.open", new_callable=mock_open, read_data='')
    @mock.patch('source.preprocessing.psg.compumedics_processor.csv')
    def test_parse(self, mock_csv, mock_open, mock_time_service):
        mock_csv.reader.return_value = ['W', '1', 'W', '2', 'R']
        report_summary = ReportSummary(study_date="04/02/2019",
                                       start_epoch=2,
                                       start_time="10:30:13 PM",
                                       file_type=PSGFileType.Compumedics)
        psg_stage_path = 'path/to/file'
        mock_time_service.get_start_epoch_timestamp.return_value = expected_start_timestamp = 1234567890

        expected_data = [StageItem(epoch=Epoch(timestamp=expected_start_timestamp,
                                               index=2), stage=SleepStage.n1),
                         StageItem(epoch=Epoch(timestamp=expected_start_timestamp + Epoch.DURATION,
                                               index=3), stage=SleepStage.wake),
                         StageItem(epoch=Epoch(timestamp=expected_start_timestamp + Epoch.DURATION*2,
                                               index=4), stage=SleepStage.n2),
                         StageItem(epoch=Epoch(timestamp=expected_start_timestamp + Epoch.DURATION*3,
                                               index=5), stage=SleepStage.rem)]

        data = CompumedicsProcessor.parse(report_summary, psg_stage_path)

        TestHelper.assert_models_equal(self, expected_data[0].epoch, data[0].epoch)
        TestHelper.assert_models_equal(self, expected_data[0].stage, data[0].stage)
        TestHelper.assert_models_equal(self, expected_data[1].epoch, data[1].epoch)
        TestHelper.assert_models_equal(self, expected_data[1].stage, data[1].stage)
        TestHelper.assert_models_equal(self, expected_data[2].epoch, data[2].epoch)
        TestHelper.assert_models_equal(self, expected_data[2].stage, data[2].stage)
        TestHelper.assert_models_equal(self, expected_data[3].epoch, data[3].epoch)
        TestHelper.assert_models_equal(self, expected_data[3].stage, data[3].stage)


