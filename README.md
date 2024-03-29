# Educational Predictive Model Fairness

This repository is a base at exploring fairness in educational predictive models. It is based on the NeurIPS 2020 Educational Challenge put forth by the Eedi project [https://eedi.com/projects/neurips-education-challenge]

## Data

A data dictionary and details can be found in the arxiv paper on the challenge: [https://arxiv.org/pdf/2007.12061.pdf]

Download the Eedi dataset from [https://dqanonymousdata.blob.core.windows.net/neurips-public/data.zip] and init the repository with the following:

```sh
mkdir data
wget -qO- https://dqanonymousdata.blob.core.windows.net/neurips-public/data.zip | bsdtar -xf - ./data/
```

There is also a starter kit which was given to participants. It can be downloaded with the following:

```sh
cd data
wget -qO- https://dqanonymousdata.blob.core.windows.net/neurips-public/starter_kit.zip | bsdtar -xf - ./starter_kit/
```

This dataset includes a few specific files relevant to this analysis:
1. `data/train_data/train_task_1_2.csv` which has 15867851 observations. This data is used to train competition models.
2. `data/test_data/test_public_answers_task_1.csv` which has 1983482 observations. This data is used to evaluate the competition models but is not blinded from the participants.
3. `data/test_data/test_private_answers_task_1.csv` which as 1983483 observations. This data is used as a hold out set for evaluation of the competition models.

The starter kit has one dataset of interest:
1. `data/starter_kit/submission_task_1_2.csv` -- this file is the same as the private holdout data but has no target variable (whether the question was answered correctly or not), and it is expected that submissions will use this to generate an output file to validate.

**Note: There are potential issues of data leakage, in that the participants are not blinded to the students and questions they will be evaluated on. It would be possible, for instance, for a model to focus on generating good predictions for the students that appear more frequestly in the holdout dataset at the expense of predictions for students who appear less frequently in the houldout data but are in the training data.**

## Model Submissions

Two submissions to the challenge are considered, and are initialized as forks from the submission repositories:

1. Option Tracing: Beyond Binary Knowledge Tracing Aritra Ghosh and Andrew S. Lan, forked from [https://github.com/arghosh/NeurIPSEducation2020] to [https://github.com/educational-technology-collective/eedi_fairness_arghosh]
2. Practical Strategies for Improving the Performance of Student Response Prediction Daichi Takehara and Yuto Shinahara
forked from [https://github.com/haradai1262/NeurIPS-Education-Challenge-2020] to [https://github.com/educational-technology-collective/eedi_fairness_haradai1262]

In addition, the starter kit for the competition also includes a base implementation

3. Diagnostic questions: The neurips 2020 education challenge. Wang, Zichao and Lamb, Angus and Saveliev, Evgeny and Cameron, Pashmina and Zaykov, Yordan and Hernandez-Lobato, Jose Miguel and Turner, Richard E and Baraniuk, Richard G and Barton, Craig and Jones, Simon Peyton and Woodhead, Simon and Zhang, Cheng. arXiv preprint arXiv:2007.12061

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