from unittest import TestCase

from source.preprocessing.psg.psg_file_type import PSGFileType
from source.preprocessing.psg.report_summary import ReportSummary
from source.preprocessing.time_service import TimeService


class TestTimeService(TestCase):

    def test_build_from_report_summary_after_midnight(self):
        start_time = "2:43:31"
        start_epoch = 1235644
        study_date = "4/10/19"
        report_summary = ReportSummary(study_date=study_date, start_epoch=start_epoch, start_time=start_time,
                                       file_type=PSGFileType.Vitaport)
        expected_epoch = 1554965011
        epoch = TimeService.get_start_epoch_timestamp(report_summary)
        self.assertEqual(expected_epoch, epoch)

    def test_build_from_report_summary_before_midnight(self):
        start_time = "11:43:31 PM"
        start_epoch = 1235644
        study_date = "4/10/2019"
        report_summary = ReportSummary(study_date=study_date, start_epoch=start_epoch, start_time=start_time,
                                       file_type=PSGFileType.Compumedics)
        expected_epoch = 1554954211
        epoch = TimeService.get_start_epoch_timestamp(report_summary)
        self.assertEqual(expected_epoch, epoch)
