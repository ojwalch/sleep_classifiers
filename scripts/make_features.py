import numpy as np
import math
from scipy import stats
from scipy import signal
import sys
from scipy.signal import butter, lfilter, filtfilt
from matplotlib import pyplot as plt
import os.path

data_path = '../data/cleaned_data/'
save_path = '../data/features/'

DT_SCORES = 30  # seconds
DT_MOTION = 15  # seconds
DT_HR = 2       # seconds
DT_CLOCK = 15   # seconds
WINDOW_SIZE = 11  # number of epochs to consider centered around the time point, must be odd
WINDOW_SIZE_HR = 11  #
WANT_CIRC = True  # Boolean for if we should compute circadian feature

SECONDS_PER_HOUR = 3600.0
HOURS_PER_DAY = 24
HR_SMOOTHING_WINDOW_SIZE = 250


def smooth(y, box_pts):  # TODO: Use or remove.
    box = np.ones(box_pts) / box_pts

    y = np.insert(y, y[0:box_pts / 2], 0)               # Pad by repeating boundary conditions
    y = np.insert(y, y[-box_pts / 2 + 1:], len(y) - 1)
    y_smooth = np.convolve(y, box, mode='valid')

    return y_smooth


def smooth_gauss(y, box_pts):
    box = np.ones(box_pts) / box_pts
    mu = box_pts / 2.0
    sigma = box_pts / 1.0

    for ind in range(0, box_pts):
        box[ind] = np.exp(-1 / 2 * (((ind - mu) / sigma) ** 2))

    box = box / np.sum(box)

    y = np.insert(y, y[0:box_pts / 2], 0)               # Pad by repeating boundary conditions
    y = np.insert(y, y[-box_pts / 2 + 1:], len(y) - 1)
    y_smooth = np.convolve(y, box, mode='valid')

    return y_smooth


def make_features(subject_id):
    """
         Compute features and save to file; save validation images to outputs/ folder
         Args:
             subject_id (int): Subject ID

         """
    is_started = False

    score_output = np.array([])
    motion_output = np.array([])
    hr_output = np.array([])
    clock_output = np.array([])
    circ_model_output = np.array([])

    print('Making features for Subject ' + str(subject_id))

    file_name = data_path + str(subject_id)

    print('-- Reading data...')

    scores = np.genfromtxt(file_name + '_scores.out', delimiter=',')
    hr = np.genfromtxt(file_name + '_hr.out', delimiter=',')
    motion = np.genfromtxt(file_name + '_counts.out', delimiter=',')

    if WANT_CIRC:
        if os.path.isfile(file_name + '_clock_proxy.out'):
            circ_model = np.genfromtxt(file_name + '_clock_proxy.out', delimiter=',')
            circadian_file_exists = True
        else:
            circadian_file_exists = False

    start_time = np.amin(scores[:, 0])
    end_time = min([motion[-1, 0], hr[-1, 0], scores[-1, 0]])      # End time is the minimum of the last valid value.

    duration = int(math.floor(end_time - start_time))

    print('Duration: ' + str((end_time - start_time) / SECONDS_PER_HOUR) + ' hrs')

    hr = process_hr(hr, start_time, end_time)
    # motion = process_motion(motion)   ## TODO: Use or remove

    last_scored_epoch = 0

    print('- Looping over epochs...')

    # These are for plotting the inputs for debugging
    plot_timestamps = []
    plot_activity_counts = []
    plot_scores = []
    plot_heart_rate = []
    plot_circadian_model = []
    plot_cosine_clock = []

    invalid_motion_count = 0
    invalid_hr_count = 0
    total_epoch_count = 0

    for i in range(DT_SCORES / 2, duration, DT_SCORES):  # Loops over all 30s epochs

        total_epoch_count = total_epoch_count + 1

        begin_epoch_time = int(start_time + i - DT_SCORES / 2)
        end_epoch_time = int(start_time + i + DT_SCORES / 2)

        scores_in_range_condition = (np.array(scores[:, 0]) < end_epoch_time) & (np.array(scores[:, 0]) >= begin_epoch_time)
        scores_in_range = np.extract(scores_in_range_condition, scores[:, 1])

        if len(scores_in_range) > 0:
            epoch = stats.mode(scores_in_range)
            epoch = epoch[0]
            last_scored_epoch = epoch
        else:
            epoch = last_scored_epoch

        if (np.mean(scores_in_range) - epoch) != 0 and epoch != 0:  # Catch misalignment
            print('ERROR: Scores in range non-constant. Is something offset by < 30s?')
            print(scores_in_range)
            print('Scored epoch: ' + str(epoch))

        if np.mean(scores_in_range) >= 0:  # If negative, epoch was not scored

            # Set the sample window, centered at i, over which the heart rate and motion will be collected
            sample_begin = int(start_time + i - WINDOW_SIZE * DT_SCORES / 2)
            sample_end = int(start_time + i + WINDOW_SIZE * DT_SCORES / 2)

            sample_begin_hr = int(start_time + i - WINDOW_SIZE_HR * DT_SCORES / 2)
            sample_end_hr = int(start_time + i + WINDOW_SIZE_HR * DT_SCORES / 2)

            # Grab all features in range
            motion_epoch = get_motion_feature(range(sample_begin, sample_end, DT_MOTION), motion)
            hr_epoch = get_hr_feature(range(sample_begin_hr, sample_end_hr, DT_HR), hr)
            clock_epoch = get_clock_feature(i)
            time_epoch = get_time_feature(i)

            if WANT_CIRC and circadian_file_exists:
                circ_model_epoch = get_circ_model_feature(start_time + i, circ_model)

            if motion_epoch[0] != -1:
                motion_valid = True
            else:
                motion_valid = False
                invalid_motion_count = invalid_motion_count + 1
                print('Invalid motion in epoch: ' + str(invalid_motion_count))

            if hr_epoch[0] != -1:
                hr_valid = True
            else:
                hr_valid = False
                invalid_hr_count = invalid_hr_count + 1
                print('Invalid heart rate in epoch: ' + str(invalid_hr_count))

            # Only append if motion data was valid in range
            if motion_valid and hr_valid:

                if is_started is False:
                    is_started = True
                    score_output = np.append(score_output, epoch)
                    motion_output = motion_epoch
                    hr_output = hr_epoch
                    clock_output = clock_epoch
                    time_output = time_epoch

                    if WANT_CIRC and circadian_file_exists:
                        circ_model_output = circ_model_epoch
                else:
                    score_output = np.append(score_output, epoch)
                    motion_output = np.vstack([motion_output, motion_epoch])
                    hr_output = np.vstack([hr_output, hr_epoch])
                    clock_output = np.vstack([clock_output, clock_epoch])
                    time_output = np.vstack([time_output, time_epoch])

                    if WANT_CIRC and circadian_file_exists:
                        circ_model_output = np.vstack([circ_model_output, circ_model_epoch])

                plot_timestamps.append(len(plot_timestamps) + 1)
                if epoch > 0:
                    plot_scores.append(1)
                else:
                    plot_scores.append(0)

                motion_epoch = np.array(motion_epoch)

                plot_activity_counts.append(np.mean(motion_epoch / 50))
                plot_heart_rate.append(hr_epoch[0])

                if WANT_CIRC and circadian_file_exists:
                    plot_circadian_model.append(np.mean(circ_model_epoch))
                plot_cosine_clock.append(np.mean(clock_epoch))

    # Plot for debugging purposes
    plot_heart_rate = np.array(plot_heart_rate)
    plot_heart_rate = 5.0 * plot_heart_rate / np.amax(plot_heart_rate)

    plt.plot(plot_timestamps, plot_activity_counts)
    plt.step(plot_timestamps, plot_scores)
    plt.plot(plot_timestamps, plot_heart_rate)

    if WANT_CIRC and circadian_file_exists:
        plt.plot(plot_timestamps, plot_circadian_model)

    plt.plot(plot_timestamps, plot_cosine_clock)
    plt.ylim(-1, 8)

    plt.savefig('../outputs/feature_validation_' + subject_id + '.png')
    plt.close()

    print('- Saving features...')

    # Save features.
    np.savetxt(save_path + subject_id + '_score_feat.csv', score_output, delimiter=",", fmt='%d')
    np.savetxt(save_path + subject_id + '_motion_feat.csv', motion_output, delimiter=",", fmt='%f')
    np.savetxt(save_path + subject_id + '_hr_feat.csv', hr_output, delimiter=",", fmt='%f')
    np.savetxt(save_path + subject_id + '_clock_feat.csv', clock_output, delimiter=",", fmt='%f')
    np.savetxt(save_path + subject_id + '_time_feat.csv', time_output, delimiter=",", fmt='%f')

    if WANT_CIRC and circadian_file_exists:
        np.savetxt(save_path + subject_id + '_circ_model_feat.csv', circ_model_output, delimiter=",", fmt='%f')


def process_motion(motion):
    # TODO: Use or remove

    return motion


def process_hr(hr, start, end):
    """
         Process heart rate.
         Args:
             hr (np.array): Heart rate data
             start (int): Timestamp in seconds of start time
             end (int): Timestamp in seconds of end time

         Returns:
             np.array : Interpolated heart rate, converted from bpm to seconds
         """

    time_range = range(int(start), int(end), DT_HR)
    indices_where_hr_zero = np.where(hr[:, 1] == 0)[0]
    indices_where_hr_nonzero = np.where(hr[:, 1] > 0)[0]

    hr[indices_where_hr_zero, 1] = 1  # Placeholder value; gets interpolated over later
    hr[:, 1] = 60.0 / hr[:, 1]

    hr[indices_where_hr_zero, 1] = np.interp(indices_where_hr_zero, indices_where_hr_nonzero,
                                             hr[indices_where_hr_nonzero, 1])

    hr_interp = np.interp(time_range, hr[:, 0], hr[:, 1])
    hr_interp = hr_interp / np.std(hr_interp)

    hr_interp = smooth_gauss(hr_interp, HR_SMOOTHING_WINDOW_SIZE)
    hr = np.hstack((np.transpose([time_range]), np.transpose([hr_interp])))

    return hr


def get_motion_feature(sample_range, motion):
    """
         Gets motion feature, doing some smoothing of the data
         Args:
             sample_range ([float]): Timestamps to interpolate motion over
             motion (np.array): Activity count data, first column is timestamps, second is counts

         Returns:
             np.array : motion feature for input to classifiers
         """
    motion_bin = np.interp(sample_range, motion[:, 0], motion[:, 1])
    mu = len(motion_bin) / 2.0
    sigma = len(motion_bin) / 4.0
    convolution = 0
    count = 0
    for m in motion_bin:
        convolution = convolution + m * np.exp(-1 / 2 * (((count - mu) / sigma) ** 2))
        count = count + 1

    return np.array([convolution])


def get_hr_feature(sample_range, hr):
    """
         Gets heart rate feature computing standard deviation of heart rate in sample.
         Args:
             sample_range ([float]): Timestamps to interpolate motion over
             heart rate (np.array): Heart rate data, first column is timestamps, second is heart rate

         Returns:
             np.array : heart rate feature for input to classifiers
         """
    heart_rate_in_range_condition = (np.array(hr[:, 0]) >= sample_range[0]) & (np.array(hr[:, 0]) <= sample_range[-1])
    time_points_in_range = np.extract(heart_rate_in_range_condition, hr[:, 0])
    heart_rate_in_range = np.extract(heart_rate_in_range_condition, hr[:, 1])

    if len(heart_rate_in_range) > 1:  # Checks to make sure heart rate is valid
        raw_hr = np.interp(sample_range, time_points_in_range, heart_rate_in_range)
        abs_hr_derivative = np.abs(heart_rate_in_range[0:-1] - heart_rate_in_range[1:])

        mu = len(abs_hr_derivative) / 2.0
        sigma = len(abs_hr_derivative) / 6.0

        convolution = 0
        count = 0

        for hr_derivative_point in abs_hr_derivative:
            convolution = convolution + hr_derivative_point * np.exp(-1 / 2 * (((count - mu) / sigma) ** 2))
            count = count + 1

        return np.array([np.std(raw_hr)])
        # return np.array([convolution, np.std(raw_hr)])  # Old way of doing it.

    return np.array([-1])


def cosine_clock_proxy(time):
    """
          Cosine
          Args:
              time (float): Timestamp for epoch

          Returns:
              np.array : circadian model feature for input to classifiers from cosine
          """
    sleep_drive_cosine_shift = 5
    return -1 * math.cos((time - sleep_drive_cosine_shift * SECONDS_PER_HOUR) * 2 * math.pi / (SECONDS_PER_HOUR*HOURS_PER_DAY))


def get_time_feature(time):
    """
         Gets homeostat prediction as a function of time over the course of the night
         Args:
             time (float): Timestamp for epoch

         Returns:
             np.array : homeostat model feature for input to classifiers
         """
    homeostat_point_at_time = math.exp(-1 * time / (202.2 * 60.0))

    return np.array([homeostat_point_at_time])


def get_clock_feature(time):
    """
         Gets circadian clock model prediction from a cosine stand-in
         Args:
             time (float): Timestamp for epoch

         Returns:
             np.array : circadian model feature for input to classifiers from cosine
         """
    cosine_point_at_time = cosine_clock_proxy(time)
    base_point = cosine_clock_proxy(0)
    norm_point = (cosine_point_at_time - base_point) / (-1 - base_point)
    clock_feature = np.array([norm_point])

    return clock_feature


def get_circ_model_feature(time, circ_model):
    """
         Gets circadian clock model prediction from the precomputed output of the circadian model (Forger, 1999)
         Args:
             time (float): Timestamp for epoch
             circ_model (np.array): Circadian clock model array; first column is time, second column is clock output

         Returns:
             np.array : circadian model feature for input to classifiers from Forger model
         """
    circadian_point_at_time = np.interp(time + DT_SCORES / 2, circ_model[:, 0], circ_model[:, 1])
    normalized_circadian_point_at_time = (circadian_point_at_time - circ_model[0, 1]) / (np.amin(circ_model[:, 1] - circ_model[0, 1]))
    cm_feature = np.array([normalized_circadian_point_at_time])

    return cm_feature


if __name__ == '__main__':
    make_features(sys.argv[1])
