# sleep_classifiers

This code uses scikit-learn to classify sleep based on acceleration and photoplethymography-derived heart rate from the Apple Watch. 

## Getting Started

This code uses Python 3.7.

## Data

Data collected using the Apple Watch is available on PhysioNet: [link](https://alpha.physionet.org/content/sleep-accel/1.0.0/)

The MESA dataset is available for download at the [National Sleep Research Resource](https://sleepdata.org)

## Features + figures

All raw data are cleaned and features are generated in ```preprocessing_runner.py.```

The file ```analysis_runner.py``` can be used to generate figures showing classifier performance. 

## License

This software is open source and under an MIT license.
