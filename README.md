# CharityML Project

The main goal of the project is to predict persons who are willing to donate.
It's an ML pipeline that can be run with any new data.

## Pre-requisites
* Python v3.7
* Pandas v0.25.1
* Numpy v1.17.2
* Scikit-learn v0.21.3 
* Joblib v0.14

## Getting Started
The whole project is decomposed into five parts. So each part of the ML
project can run independently.

### Feature Engineering (Preprocess)
Run `./run preprocess` from the main directory. It will automatically
preprocess the data, split into training and test and save it into models directory.

### Train
Run `./run train <algorithm_name>`. There are three algorithms implemented: `ada_boost`, `random_forest`, `gradient_boosting`. New algorithms can be added via changes to `./src/dispatcher.py`. Run training and see which model performs best. Pick the best model and use it in the next step.

### Optimize
Run `./run optimize <algorithm_name>`. The parameters can be changed in
`./src/dispatcher.py`. Only for two algorithms, the parameters are set. The
parameters can be adjusted at any time.

### Predict
Run `./run predict <algorithm_name>`. Currently, it runs with the test data from the training part. But it can be with any newly available data. 

## Support
Patches are encouraged and may be submitted by forking this project
and submitting a pull request through GitHub. 

## Licence
Released under the [MIT License](./License.md)
