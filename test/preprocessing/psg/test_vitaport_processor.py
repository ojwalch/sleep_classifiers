from unittest import TestCase, mock
from mock import mock_open

from source.preprocessing.epoch import Epoch
from source.preprocessing.psg.psg_file_type import PSGFileType
from source.preprocessing.psg.report_summary import ReportSummary
from source.preprocessing.psg.stage_item import StageItem
from source.preprocessing.psg.vitaport_processor import VitaportProcessor
from source.sleep_stage import SleepStage
from test.test_helper import TestHelper


class TestVitaportProcessor(TestCase):

    @mock.patch("source.preprocessing.psg.vitaport_processor.TimeService")
    @mock.patch("builtins.open", new_callable=mock_open, read_data='')
    @mock.patch('source.preprocessing.psg.vitaport_processor.csv')
    def test_parse(self, mock_csv, mock_open, mock_time_service):
        mock_csv.reader.return_value = [['0', '23:20:55'], ['0', '23:21:05'], ['0', '23:21:15'], ['2', '23:21:25'],
                                        ['2', '23:21:35']]
        report_summary = ReportSummary(study_date="04/02/2019",
                                       start_epoch=1,
                                       start_time="11:20:55 PM",
                                       file_type=PSGFileType.Vitaport)
        psg_stage_path = 'path/to/file'
        mock_time_service.get_start_epoch_timestamp.return_value = expected_start_timestamp = 1234567890

        expected_data = [StageItem(epoch=Epoch(timestamp=expected_start_timestamp,
                                               index=1), stage=SleepStage.wake),
                         StageItem(epoch=Epoch(timestamp=expected_start_timestamp + Epoch.DURATION,
                                               index=2), stage=SleepStage.n2)]

        data = VitaportProcessor.parse(report_summary, psg_stage_path)

        TestHelper.assert_models_equal(self, expected_data[0].epoch, data[0].epoch)
        TestHelper.assert_models_equal(self, expected_data[0].stage, data[0].stage)
        TestHelper.assert_models_equal(self, expected_data[1].epoch, data[1].epoch)
        TestHelper.assert_models_equal(self, expected_data[1].stage, data[1].stage)

