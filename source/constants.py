from source import utils


class Constants(object):
    # WAKE_THRESHOLD = 0.3  # These values were used for scikit-learn 0.20.3, See:
    # REM_THRESHOLD = 0.35  # https://scikit-learn.org/stable/whats_new.html#version-0-21-0
    WAKE_THRESHOLD = 0.5  #
    REM_THRESHOLD = 0.35

    INCLUDE_CIRCADIAN = False
    EPOCH_DURATION_IN_SECONDS = 30
    SECONDS_PER_MINUTE = 60
    SECONDS_PER_DAY = 3600 * 24
    SECONDS_PER_HOUR = 3600
    VERBOSE = True
    CROPPED_FILE_PATH = utils.get_project_root().joinpath('outputs/cropped/')
    FEATURE_FILE_PATH = utils.get_project_root().joinpath('outputs/features/')
    FIGURE_FILE_PATH = utils.get_project_root().joinpath('outputs/figures/')
    LOWER_BOUND = -0.2
    MATLAB_PATH = '/Applications/MATLAB_R2019a.app/bin/matlab'  # Replace with your MATLAB path
