import os
import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from source import utils
from source.constants import Constants


class DataPlotBuilder(object):
    @staticmethod
    def timestamp_to_string(ts):
        return time.strftime('%H:%M:%S', time.localtime(ts))

    @staticmethod
    def convert_labels_for_hypnogram(labels):
        processed_labels = np.array([])

        for epoch in labels:
            if epoch == -1:
                processed_labels = np.append(processed_labels, 0)
            elif epoch == 5:
                processed_labels = np.append(processed_labels, 1)
            else:
                processed_labels = np.append(processed_labels, -1 * epoch)

        return processed_labels

    @staticmethod
    def tidy_data_plot(x_min, x_max, dt, ax):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        xticks = np.arange(x_min, x_max, dt)
        plt.xticks(xticks)
        labels = []
        for xt in xticks:
            labels.append(DataPlotBuilder.timestamp_to_string(xt))
        ax.set_xticklabels(labels)
        plt.xlim(x_min, x_max)

    @staticmethod
    def make_data_demo(subject_id="16", snippet=False):
        hr_color = [0.8, 0.2, 0.1]
        motion_color = [0.3, 0.2, 0.8]
        circ_color = [0.9, 0.7, 0]
        psg_color = [0.1, 0.7, 0.1]
        font_size = 16
        font_name = "Arial"

        data_path = str(Constants.CROPPED_FILE_PATH) + '/'
        circadian_data_path = str(utils.get_project_root().joinpath('data/circadian_predictions/')) + '/'
        output_path = str(Constants.FIGURE_FILE_PATH) + '/'

        if snippet is False:
            fig = plt.figure(figsize=(10, 12))
        else:
            fig = plt.figure(figsize=(3, 12))

        num_v_plots = 5
        fig.patch.set_facecolor('white')

        if (os.path.isfile(data_path + subject_id + '_cleaned_hr.out') and os.path.isfile(
                data_path + subject_id + '_cleaned_motion.out') and os.path.isfile(
            data_path + subject_id + '_cleaned_psg.out') and
            os.path.isfile(data_path + subject_id + '_cleaned_counts.out') and
            os.stat(data_path + subject_id + '_cleaned_motion.out').st_size > 0) and os.path.isfile(
            circadian_data_path + subject_id + '_clock_proxy.txt'):

            hr = np.genfromtxt(data_path + subject_id + '_cleaned_hr.out', delimiter=' ')
            motion = np.genfromtxt(data_path + subject_id + '_cleaned_motion.out', delimiter=' ')
            scores = np.genfromtxt(data_path + subject_id + '_cleaned_psg.out', delimiter=' ')
            counts = np.genfromtxt(data_path + subject_id + '_cleaned_counts.out', delimiter=',')
            circ_model = np.genfromtxt(circadian_data_path + subject_id + '_clock_proxy.txt', delimiter=',')

            min_time = min(scores[:, 0])
            max_time = max(scores[:, 0])
            dt = 60 * 60

            sample_point_fraction = 0.92

            sample_point = sample_point_fraction * (max_time - min_time) + min_time
            window_size = 10
            if snippet:
                min_time = sample_point
                max_time = sample_point + window_size

            ax = plt.subplot(num_v_plots, 1, 1)
            ax.plot(motion[:, 0], motion[:, 1], color=motion_color)
            ax.plot(motion[:, 0], motion[:, 2], color=[0.4, 0.2, 0.7])
            ax.plot(motion[:, 0], motion[:, 3], color=[0.5, 0.2, 0.6])
            plt.ylabel('Motion (g)', fontsize=font_size, fontname=font_name)
            DataPlotBuilder.tidy_data_plot(min_time, max_time, dt, ax)

            if snippet:
                ax.spines['bottom'].set_visible(True)
                ax.spines['left'].set_visible(True)
                ax.spines['top'].set_visible(True)
                ax.spines['right'].set_visible(True)

                ax.yaxis.label.set_visible(False)

                inds = np.intersect1d(np.where(motion[:, 0] > sample_point)[0],
                                      np.where(motion[:, 0] <= sample_point + window_size)[0])
                y_min = np.amin(motion[inds, 1:3])
                plt.ylim(y_min - 0.005, y_min + 0.025)

                # Get rid of the ticks
                ax.set_xticks([])
                ax.yaxis.set_ticks_position("right")

                plt.ylabel('')
                plt.xlabel(str(window_size) + ' sec window', fontsize=font_size, fontname=font_name)
            else:
                y_min = -3.2
                y_max = 2.5
                plt.ylim(y_min, y_max)
                current_axis = plt.gca()
                current_axis.add_patch(
                    Rectangle((sample_point, y_min), window_size, y_max - y_min, alpha=0.7, facecolor="gray"))

            ax = plt.subplot(num_v_plots, 1, 2)
            ax.plot(counts[:, 0], counts[:, 1], color=[0.2, 0.2, 0.7])
            DataPlotBuilder.tidy_data_plot(min_time, max_time, dt, ax)
            plt.ylabel('Counts', fontsize=font_size, fontname=font_name)

            if snippet:
                plt.axis('off')
                plt.ylim(-1, -1)

            ax = plt.subplot(num_v_plots, 1, 3)
            ax.plot(hr[:, 0], hr[:, 1], color=hr_color)
            plt.ylabel('Heart rate (bpm)', fontsize=font_size, fontname=font_name)
            DataPlotBuilder.tidy_data_plot(min_time, max_time, dt, ax)

            sample_point = sample_point_fraction * (max_time - min_time) + min_time
            window_size = 1200

            if snippet:
                min_time = sample_point
                max_time = sample_point + window_size
                DataPlotBuilder.tidy_data_plot(min_time, max_time, dt, ax)

                ax.spines['bottom'].set_visible(True)
                ax.spines['left'].set_visible(True)
                ax.spines['top'].set_visible(True)
                ax.spines['right'].set_visible(True)

                ax.yaxis.label.set_visible(False)

                ax.set_xticks([])
                ax.yaxis.set_ticks_position("right")

                plt.ylabel('')
                plt.xlabel(str(window_size) + ' sec window', fontsize=font_size, fontname=font_name)
                plt.ylim(35, 100)

            else:
                y_min = 40
                y_max = 130
                plt.ylim(y_min, y_max)
                current_axis = plt.gca()
                current_axis.add_patch(
                    Rectangle((sample_point, y_min), window_size, y_max - y_min, alpha=0.35, facecolor="gray"))
                plt.ylim(40, 130)

            ax = plt.subplot(num_v_plots, 1, 4)
            ax.plot(circ_model[:, 0], -circ_model[:, 1], color=circ_color)
            plt.ylabel('Clock Proxy', fontsize=font_size, fontname=font_name)
            DataPlotBuilder.tidy_data_plot(min_time, max_time, dt, ax)
            if snippet:
                plt.axis('off')
                plt.ylim(-1, -1)
            else:
                plt.ylim(.2, 1.2)

            ax = plt.subplot(num_v_plots, 1, 5)

            relabeled_scores = DataPlotBuilder.convert_labels_for_hypnogram(scores[:, 1])
            ax.step(scores[:, 0], relabeled_scores, color=psg_color)
            plt.ylabel('Stage', fontsize=font_size, fontname=font_name)
            plt.xlabel('Time', fontsize=font_size, fontname=font_name)
            DataPlotBuilder.tidy_data_plot(min_time, max_time, dt, ax)
            ax.set_yticks([-4, -3, -2, -1, 0, 1])
            ax.set_yticklabels(['N4', 'N3', 'N2', 'N1', 'Wake', 'REM'])

            if snippet:
                plt.axis('off')
                plt.ylim(5, 5)
            else:
                plt.ylim(-5, 2)

            if not snippet:
                plt.savefig(output_path + 'data_validation_' + subject_id + '.png', bbox_inches='tight', pad_inches=0.1,
                            dpi=300)
            else:
                plt.savefig(output_path + 'data_validation_zoom_' + subject_id + '.png', bbox_inches='tight',
                            pad_inches=0.1, dpi=300)
            plt.close()
