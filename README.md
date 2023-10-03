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
