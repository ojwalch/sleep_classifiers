import numpy as np
import pandas as pd

from source.constants import Constants
from source.preprocessing.psg.psg_service import PSGService


class PSGLabelService(object):
    @staticmethod
    def load(subject_id):
        psg_label_path = PSGLabelService.get_path(subject_id)
        feature = pd.read_csv(str(psg_label_path)).values
        return feature

    @staticmethod
    def get_path(subject_id):
        return Constants.FEATURE_FILE_PATH.joinpath(subject_id + '_psg_labels.out')

    @staticmethod
    def build(subject_id, valid_epochs):
        psg_array = PSGService.load_cropped_array(subject_id)
        labels = []
        for epoch in valid_epochs:
            value = np.interp(epoch.timestamp, psg_array[:, 0], psg_array[:, 1])
            labels.append(value)
        return np.array(labels)

    @staticmethod
    def write(subject_id, labels):
        psg_labels_path = PSGLabelService.get_path(subject_id)
        np.savetxt(psg_labels_path, labels, fmt='%f')
