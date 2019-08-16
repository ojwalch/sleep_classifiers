import numpy as np
import pandas as pd

from source import utils


class MesaTimeBasedService(object):

    @staticmethod
    def load_circadian_model(file_id):

        path = utils.get_project_root().joinpath('data/mesa/clock_proxy/' + file_id + '_clock_proxy.out')

        if path.is_file():
            array = pd.read_csv(str(path), delimiter=',').values
            if np.shape(array)[0] > 0:
                array = utils.remove_nans(array)
            if np.shape(array)[0] > 0:
                return array

        return None
