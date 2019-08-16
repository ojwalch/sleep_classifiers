from xml.dom import minidom

import numpy as np

from source import utils


class MesaPSGService(object):

    @staticmethod
    def load_raw(file_id):
        stage_to_num = {'Stage 4 sleep|4': 4, 'Stage 3 sleep|3': 3, 'Stage 2 sleep|2': 2, 'Stage 1 sleep|1': 1,
                        'Wake|0': 0, 'REM sleep|5': 5}
        project_root = str(utils.get_project_root())

        xml_document = minidom.parse(
            project_root + '/data/mesa/polysomnography/annotations-events-nsrr/mesa-sleep-' + file_id + '-nsrr.xml')
        list_of_scored_events = xml_document.getElementsByTagName('ScoredEvent')

        stage_data = []

        for scored_event in list_of_scored_events:  # 3 is stage, 5 is start, 7 is duration
            duration = scored_event.childNodes[7].childNodes[0].nodeValue
            start = scored_event.childNodes[5].childNodes[0].nodeValue
            stage = scored_event.childNodes[3].childNodes[0].nodeValue

            if stage in stage_to_num:
                # # For debugging: print(file_id + ' ' + str(stage) + ' ' + str(start) + ' ' + str(duration))
                stage_data.append([stage_to_num[stage], float(start), float(duration)])

        stages = []
        for staged_window in stage_data[:]:  # Ignore last PSG overflow entry: it's long & doesn't have valid HR
            elapsed_time_counter = 0
            stage_value = staged_window[0]
            duration = staged_window[2]

            while elapsed_time_counter < duration:
                stages.append(stage_value)
                elapsed_time_counter = elapsed_time_counter + 1

        return np.array(stages)

    @staticmethod
    def crop(psg_labels, valid_epochs):
        cropped_psg_labels = []

        for epoch in valid_epochs:
            index = int(epoch.timestamp)

            cropped_psg_labels.append(psg_labels[index])
        return np.array(cropped_psg_labels)
