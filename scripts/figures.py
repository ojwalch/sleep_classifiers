import math
import sys
import os.path
import numpy as np
import time
import random

from PIL import Image, ImageDraw, ImageFont

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Rectangle

import csv
import mesa
import classify_sleep
import utilities

# Plotting colors
hr_color = [0.8, 0.2, 0.1]
motion_color = [0.3, 0.2, 0.8]
circ_color = [0.9, 0.7, 0]
psg_color = [0.1, 0.7, 0.1]

font_size = 16
font_name = "Arial"

data_path = '../data/cleaned_data/'
output_path = '../outputs/'


def timestamp_to_string(ts):
    return time.strftime('%H:%M:%S', time.localtime(ts))


# Makes plot nicer
def tidy_fig(x_min, x_max, dt, ax):
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
        labels.append(timestamp_to_string(xt))
    ax.set_xticklabels(labels)
    plt.xlim(x_min, x_max)


# Plots subject data
def make_data_demo(subject_id=16, snippet=False):
    subject_id = str(subject_id)

    if snippet is False:
        fig = plt.figure(figsize=(10, 12))
    else:
        fig = plt.figure(figsize=(3, 12))

    num_v_plots = 5
    fig.patch.set_facecolor('white')

    if (os.path.isfile(data_path + subject_id + '_hr.out') and os.path.isfile(
            data_path + subject_id + '_motion.out') and os.path.isfile(data_path + subject_id + '_scores.out') and
        os.path.isfile(data_path + subject_id + '_counts.out') and
        os.stat(data_path + subject_id + '_motion.out').st_size > 0) and os.path.isfile(
        data_path + subject_id + '_clock_proxy.out'):

        hr = np.genfromtxt(data_path + subject_id + '_hr.out', delimiter=',')
        motion = np.genfromtxt(data_path + subject_id + '_motion.out', delimiter=',')
        scores = np.genfromtxt(data_path + subject_id + '_scores.out', delimiter=',')
        counts = np.genfromtxt(data_path + subject_id + '_counts.out', delimiter=',')
        circ_model = np.genfromtxt(data_path + subject_id + '_clock_proxy.out', delimiter=',')

        min_time = min(scores[:, 0])
        max_time = max(scores[:, 0])
        dt = 60 * 60

        sample_point_fraction = 0.93

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
        tidy_fig(min_time, max_time, dt, ax)

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
        tidy_fig(min_time, max_time, dt, ax)
        plt.ylabel('Counts', fontsize=font_size, fontname=font_name)

        if (snippet):
            plt.axis('off')
            plt.ylim(-1, -1)

        ax = plt.subplot(num_v_plots, 1, 3)
        ax.plot(hr[:, 0], hr[:, 1], color=hr_color)
        plt.ylabel('Heart rate (bpm)', fontsize=font_size, fontname=font_name)
        tidy_fig(min_time, max_time, dt, ax)

        sample_point = sample_point_fraction * (max_time - min_time) + min_time
        window_size = 1200

        if snippet:
            min_time = sample_point
            max_time = sample_point + window_size
            tidy_fig(min_time, max_time, dt, ax)

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
        tidy_fig(min_time, max_time, dt, ax)
        if snippet:
            plt.axis('off')
            plt.ylim(-1, -1)
        else:
            plt.ylim(.2, 1.2)

        ax = plt.subplot(num_v_plots, 1, 5)
        ax.step(scores[:, 0], utilities.process_raw_scores(scores[:, 1], utilities.RUN_ALL), color=psg_color)
        plt.ylabel('Stage', fontsize=font_size, fontname=font_name)
        plt.xlabel('Time', fontsize=font_size, fontname=font_name)
        tidy_fig(min_time, max_time, dt, ax)
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


# Check each subject's data
def validate_all_data():
    print('Making figure for:')
    for subject_id in range(1, 43):
        print('...Subject ' + str(subject_id))
        make_data_demo(subject_id)


# Generate ROC curves for sleep/wake classification
def generate_sw_rocs(trial_count=10):
    classify_sleep.run_all(utilities.RUN_SW, trial_count)


# Generate ROC curves for wake/NREM/REM classification
def generate_rem_rocs(trial_count=1):
    classify_sleep.run_all(utilities.RUN_REM, trial_count)


# Sample data figure
def figure_data(subject_id):
    # Save regular data and zoomed versions
    make_data_demo(subject_id, False)
    make_data_demo(subject_id, True)

    path = '../outputs/'
    names = [path + 'data_validation_' + str(subject_id) + '.png',
             path + 'data_validation_zoom_' + str(subject_id) + '.png']
    images = map(Image.open, names)
    widths, heights = zip(*(i.size for i in images))

    max_height = max(heights)

    crop_frac = 0.2
    box_crop = (int(widths[1] * crop_frac), 0, widths[1], heights[1])
    new_im = Image.new('RGB', (int(widths[0] + (1 - crop_frac) * widths[1]), max_height), "white")

    images[1] = images[1].crop(box_crop)

    new_im.paste(images[0], (0, 0))
    new_im.paste(images[1], (widths[0], 0))

    new_im.save('figure_data_' + str(subject_id) + '.png')


# Combines ROC curve output images into a single figure
def figure_roc(rem_flag):
    METHOD_ARR = ['Logistic Regression', 'KNeighbors', 'Random Forest', 'MLP']
    comb_names = []
    for name in METHOD_ARR:
        if rem_flag == utilities.RUN_REM:
            comb_names.append(name + '_' + str(num_run) + 'output_rem__roc.png')
        else:
            comb_names.append(name + '_' + str(num_run) + 'output_sw__roc.png')

    images = map(Image.open, comb_names)
    widths, heights = zip(*(i.size for i in images))

    max_width = max(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (2 * max_width, 2 * max_height))

    count = 0
    for im in images:
        x_offset = int((count % 2) * max_width)
        y_offset = int(math.floor(count / 2) * max_height)

        new_im.paste(im, (x_offset, y_offset))
        count = count + 1

    if rem_flag == utilities.RUN_REM:
        new_im.save('figure_rem_roc.png')
    else:
        new_im.save('figure_sw_roc.png')


# Generates MESA ROC curves
def figure_mesa(run_flag):
    mesa.make_roc_mesa(run_flag)


# Combines the MESA output images into a single figure
def combine_mesa():
    comb_names = ['figure_mesa_roc.png', 'figure_mesa_roc_rem.png']
    images = map(Image.open, comb_names)
    widths, heights = zip(*(i.size for i in images))
    new_im = Image.new('RGB', (2 * widths[0], heights[0]))
    new_im.paste(images[0], (0, 0))
    new_im.paste(images[1], (widths[0], 0))
    new_im.save('figure_mesa.png')


# Prints the metadata for the Apple Watch subjects
def print_metadata_aw():
    ages = []
    tib = []
    tst = []
    sol = []
    waso = []
    slp_eff = []
    rem = []
    nrem = []

    metadata_dict = {}

    with open('../data/AppleWatchSubjects.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        csv_reader.next()
        for row in csv_reader:
            if (len(row[1]) > 0):
                subject_dict = {}
                subject_num = int(row[1])
                subject_dict['age'] = int(row[2])
                subject_dict['gender'] = row[3]
                subject_dict['tib'] = float(row[5])
                subject_dict['tst'] = float(row[6])
                subject_dict['waso'] = float(row[7])
                subject_dict['slp_eff'] = float(row[8])
                subject_dict['sol'] = float(row[9])
                subject_dict['rem'] = float(row[11]) * float(row[6]) / 100.0
                subject_dict['nrem'] = (float(row[12]) + float(row[13]) + float(row[14])) / 100.0 * float(row[6])
                metadata_dict[subject_num] = subject_dict

    women_count = 0
    for subject_num in utilities.FULL_SET:
        if subject_num in metadata_dict:

            subject_dict = metadata_dict[subject_num]
            if subject_dict['gender'] == 'Female':
                women_count = women_count + 1
            ages.append(subject_dict['age'])
            tib.append(subject_dict['tib'])
            tst.append(subject_dict['tst'])
            waso.append(subject_dict['waso'])
            slp_eff.append(subject_dict['slp_eff'])
            sol.append(subject_dict['sol'])
            rem.append(subject_dict['rem'])
            nrem.append(subject_dict['nrem'])

    ages = np.array(ages)
    tst = np.array(tst)
    waso = np.array(waso)
    slp_eff = np.array(slp_eff)
    sol = np.array(sol)
    rem = np.array(rem)
    nrem = np.array(nrem)

    print('N women: ' + str(women_count))

    latex = True
    if (latex):
        print(
            '\\begin{table} \caption{Sleep summary statistics}  \label{tab:sleepsummary} \small  \\begin{tabularx}{\columnwidth}{X | X | X  }\hline Parameter & Mean (SD) & Range \\\\ \hline')
    else:
        print('Parameter, Mean (SD), Range')

    print(utilities.data_to_line('Age (years)', ages, latex))
    print(utilities.data_to_line('TST (minutes)', tst, latex))
    print(utilities.data_to_line('TIB (minutes)', tib, latex))
    print(utilities.data_to_line('SOL (minutes)', sol, latex))
    print(utilities.data_to_line('WASO (minutes)', waso, latex))
    print(utilities.data_to_line('SE (\%)', slp_eff, latex))
    print(utilities.data_to_line('Time in REM (minutes)', rem, latex))
    print(utilities.data_to_line('Time in NREM (minutes)', nrem, latex))

    if (latex):
        print('\end{tabularx} \end{table}')


fig_key = sys.argv[1]

if fig_key == 'table':
    print_metadata_aw()

if fig_key == 'roc_rem':
    num_run = 2
    generate_rem_rocs(num_run)
    figure_roc(utilities.RUN_REM)

if fig_key == 'data':
    figure_data(16)

if fig_key == 'roc_sw':
    num_run = 2
    generate_sw_rocs(num_run)
    figure_roc(utilities.RUN_SW)

if fig_key == 'mesa_sw':
    figure_mesa(utilities.RUN_SW)

if fig_key == 'mesa_rem':
    figure_mesa(utilities.RUN_REM)

if fig_key == 'mesa_all':
    combine_mesa()
