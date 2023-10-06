# Educational Predictive Model Fairness

This repository is a base at exploring fairness in educational predictive models. It is based on the NeurIPS 2020 Educational Challenge put forth by the Eedi project [https://eedi.com/projects/neurips-education-challenge]

## Data

A data dictionary and details can be found in the arxiv paper on the challenge: [https://arxiv.org/pdf/2007.12061.pdf]

Download the Eedi dataset from [https://dqanonymousdata.blob.core.windows.net/neurips-public/data.zip] and init the repository with the following:

```sh
mkdir data
wget -qO- https://dqanonymousdata.blob.core.windows.net/neurips-public/data.zip | bsdtar -xf - ./data/

```

## Model Submissions

Two submissions to the challenge are considered, and are initialized as forks from the submission repositories:

1. Option Tracing: Beyond Binary Knowledge Tracing Aritra Ghosh and Andrew S. Lan, forked from [https://github.com/arghosh/NeurIPSEducation2020] to [https://github.com/educational-technology-collective/eedi_fairness_arghosh]
2. Practical Strategies for Improving the Performance of Student Response Prediction Daichi Takehara and Yuto Shinahara
forked from [https://github.com/haradai1262/NeurIPS-Education-Challenge-2020] to [https://github.com/educational-technology-collective/eedi_fairness_haradai1262]

## Task

The task analyzed is *Task 1*, whether the student will choose the correct answer.

## Setup

Each solution needs to be set up in its own way. Ensure that you have downloaded the data and put it in repo_home/data.

Submissions have been modified in good faith, with bugs addressed where possible and telemetry logging added with MLFlow.

### Arghosh

The solution by Ghosh and Lan requires some data preprocessing scripts to be run. First create a soft link to the public dataset:

```sh
cd entries/arghosh
ln -s ../../data/ public_data
```

Next, generate the expected JSON formatted data.

```sh
mkdir public_data/personal_data
python preprocessing.py
```

Then download the starter kit and create the second set of preprocessed data.

```sh
mkdir starter_kit
mkdir public_data/converted_datasets
wget -qO- https://dqanonymousdata.blob.core.windows.net/neurips-public/starter_kit.zip | bsdtar -xf - ./starter_kit
python preprocessing_2.py
```

## Running Experiments

This repository is set up to log telemetry with MLFLow. It is expected that you have installed MLFlow and configured the following three environment variables: `MLFLOW_TRACKING_USERNAME`, `MLFLOW_TRACKING_PASSWORD`, `MLFLOW_TRACKING_URI`.

The main entry script for running experiments is `run_experiments.py`