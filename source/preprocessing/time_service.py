import datetime as dt

from source.preprocessing.psg.psg_file_type import PSGFileType
from source.preprocessing.psg.report_summary import ReportSummary


class TimeService(object):
    @staticmethod
    def get_start_epoch_timestamp(report_summary: ReportSummary):
        if report_summary.file_type == PSGFileType.Compumedics:
            study_date = dt.datetime.strptime(report_summary.study_date + ' ' + report_summary.start_time,
                                              '%m/%d/%Y %I:%M:%S %p')
            if study_date.strftime('%p') == 'AM':
                study_date += dt.timedelta(days=1)
            return study_date.timestamp()

        if report_summary.file_type == PSGFileType.Vitaport:
            study_date = dt.datetime.strptime(report_summary.study_date + ' ' + report_summary.start_time,
                                              '%m/%d/%y %H:%M:%S')
            if int(study_date.strftime('%H')) < 12:
                study_date += dt.timedelta(days=1)
            return study_date.timestamp()
