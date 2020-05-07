# sleep_classifiers

This code uses scikit-learn to classify sleep based on acceleration and photoplethymography-derived heart rate from the Apple Watch. The paper associated with the work is available [here](https://academic.oup.com/sleep/article/42/12/zsz180/5549536).

## Getting Started

This code uses Python 3.7.

## Data

Data collected using the Apple Watch is available on PhysioNet: [link](https://alpha.physionet.org/content/sleep-accel/1.0.0/)

The MESA dataset is available for download at the [National Sleep Research Resource](https://sleepdata.org). You will have to request access from NSRR for the data.

## Features + figures

All raw data are cleaned and features are generated in ```preprocessing_runner.py.```

The file ```analysis_runner.py``` can be used to generate figures showing classifier performance.  You can comment and uncomment the figures you want to run. 

## Notes
- In the blue motion-only classifier performance lines in Figures 4 and 8 in [the paper](https://academic.oup.com/sleep/article/42/12/zsz180/5549536), labels for REM and NREM sleep are switched. NREM sleep is the dashed line and REM is the dotted line.
- The subset of the MESA dataset used for comparison in the paper are the first 188 subjects with valid data, in order of increasing Subject ID.

## License

This software is open source and under an MIT license.
