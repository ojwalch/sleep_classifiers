# sleep_classifiers

This code uses scikit-learn to classify sleep based on acceleration and photoplethymography-derived heart rate from the Apple Watch. The paper associated with the work is available [here](https://academic.oup.com/sleep/article/42/12/zsz180/5549536).

## Basics

This code has been updated for Python 3.9. The original version of the code used Python 3.7. You can get the source code to generate the figures in the paper here, but you'll need to pull the data from [here](https://alpha.physionet.org/content/sleep-accel/1.0.0/) and add it to the `data` folder to run the pre-processing step. 

### Data

- Data collected using the Apple Watch is available on PhysioNet: [link](https://alpha.physionet.org/content/sleep-accel/1.0.0/)

- The MESA dataset is available for download at the [National Sleep Research Resource](https://sleepdata.org). You will have to request access from NSRR for the data.

## Pre-processing the data

To convert the raw data into features that can be interpreted by the classifiers, you want to run `preprocessing_runner.py.` This script is located in the `source/preprocessing` directory. Here are the steps you need to follow to run that code: 

1. Download the [data](https://alpha.physionet.org/content/sleep-accel/1.0.0/).
2. Paste the `heart_rate`, `labels`, and `motion` folders into the `data` directory in this repository, overwriting the empty folders 
3. Run `preprocessing_runner.py`. This will take in the raw data for each subject, and use it to generate features to be read in by the classifier. The saved features will get saved to the folder `outputs/features/`, with the filename corresponding to the type of feature. For instance, the activity count feature for subject 781756 will appear as `781756_count_feature.out`. 

#### Notes
- Generating all the features should take about five minutes
- The features are text files, and if you want to see what they contain, simply open them in a text editor. 
- You should see print statements that look like this: `Cropping data from subject 8692923...`
- Followed by print statements that look like this: `Getting valid epochs... Building features...`
 


## Making the figures.

The file `analysis_runner.py` can be used to generate figures showing classifier performance. This script is located in the `source/analysis` directory. You can comment and uncomment the call list of functions at the bottom of the script to only produce the figures you want to run. 

The figure will be saved in the folder `outputs/figures` as .png files. 

The list of available analysis functions are as follows: 
- `figure_leave_one_out_roc_and_pr()`: Makes PR and ROC curves for the Apple Watch data  by training on all but one subject, and testing on the holdout subject
- `figures_mc_sleep_wake()`: Makes PR and ROC curves for the Apple Watch data  by training on a random subset of subjects, testing on the holdout subjects, and then repeating this process many times.
- `figures_mc_three_class()`: Trains on a random subset of Apple Watch subjects, tests on the remaining holdouts, and generates the "three-class" ROC described in the paper
- `figures_leave_one_out_sleep_wake_performance()`: For Apple Watch subjects, trains on everybody but one subject, tests on that holdout subject, and then generates Bland-Altman plots for sleep and wake metrics.
- `figures_leave_one_out_three_class_performance()`:  For Apple Watch subjects, trains on everybody but one subject, tests on that holdout subject, and then generates Bland-Altman plots for wake, REM, NREM metrics.

- `figures_mesa_sleep_wake()`: For MESA data, generates ROC/PR curves for sleep-wake 
- `figures_mesa_three_class()`: For MESA data, generates wake/NREM/REM analysis

- `figures_compare_time_based_features()`: Comparison of time-based features for Apple Watch data; requires MATLAB to run. 



#### Notes
- This code will not execute if you haven't run `preprocessing_runner.py` to generate the features first. 
- These figures can take a _long_ time to run. 

## General notes
- In the blue motion-only classifier performance lines in Figures 4 and 8 in [the paper](https://academic.oup.com/sleep/article/42/12/zsz180/5549536), labels for REM and NREM sleep are switched. NREM sleep is the dashed line and REM is the dotted line.
- The subset of the MESA dataset used for comparison in the paper are the first 188 subjects with valid data, in order of increasing Subject ID.
- Enough people didn't have MATLAB that I've made the circadian feature (which uses MATLAB) not included by default. Instead, the default circadian-like feature is a cosine waveform. You can re-add the circadian feature by setting `INCLUDE_CIRCADIAN = True`, though this requires MATLAB. 

## License

This software is open source and under an MIT license.
