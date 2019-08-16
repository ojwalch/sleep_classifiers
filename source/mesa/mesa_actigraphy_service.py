import csv

import numpy as np

from source import utils
from source.preprocessing.activity_count.activity_count_collection import ActivityCountCollection


class MesaActigraphyService(object):
    @staticmethod
    def load_raw(file_id):
        line_align = -1  # Find alignment line between PSG and actigraphy
        project_root = str(utils.get_project_root())

        with open(project_root + '/data/mesa/overlap/mesa-actigraphy-psg-overlap.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            next(csv_file)
            for row in csv_reader:
                if int(row[0]) == int(file_id):
                    line_align = int(row[1])

        activity = []
        elapsed_time_counter = 0

        if line_align == -1:  # If there was no alignment found
            return ActivityCountCollection(subject_id=file_id, data=np.array([[-1], [-1]]))

        with open(project_root + '/data/mesa/actigraphy/mesa-sleep-' + file_id + '.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            next(csv_file)
            for row in csv_reader:
                if int(row[1]) >= line_align:
                    if row[4] == '':
                        activity.append([elapsed_time_counter, np.nan])
                    else:
                        activity.append([elapsed_time_counter, float(row[4])])
                    elapsed_time_counter = elapsed_time_counter + 30

        data = np.array(activity)
        data = utils.remove_nans(data)

        return ActivityCountCollection(subject_id=file_id, data=data)
