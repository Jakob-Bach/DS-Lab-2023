# Data Science Laboratory Course 2023

This is the supervisor repo of the ["Data Science Laboratory Course"](https://dbis.ipd.kit.edu/3211_3244.php) at KIT in 2023.
Students worked on two subtasks:

1) a practice competition from the platform [drivendata.org](https://www.drivendata.org/)
2) one of two research tasks:
  a) active learning for SAT solving
  b) meta-learning for encoder selection

The repo provides files for preparing the datasets, some basic exploration, course-internal splitting, scoring, and demo submissions for that.
Additionally, `Surveys/` contains exports of questionnaires (created with ILIAS `v7`) to evaluate the students' satisfaction
(one survey at start of course, one after first task, one after second task).

## Setup

We use Python with version `3.8`.
We recommend to set up a virtual environment to install the dependencies, e.g., with `virtualenv`:

```bash
python -m virtualenv -p <path/to/right/python/executable> <path/to/env/destination>
```

or with `conda`:

```bash
conda create --name ds-lab-2023 python=3.8
```

Next, activate the environment with either

- `conda activate ds-lab-2023` (`conda`)
- `source <path/to/env/destination>/bin/activate` (`virtualenv`, Linux)
- `<path\to\env\destination>\Scripts\activate` (`virtualenv`, Windows)

Install the dependencies with

```bash
python -m pip install -r requirements.txt
```

If you make changes to the environment and you want to persist them, run

```bash
python -m pip freeze > requirements.txt
```

To make this environment available for notebooks, run

```
ipython kernel install --user --name=ds-lab-2023-kernel
```

To actually launch `Jupyter Notebook`, run

```
jupyter notebook
```

## Task 1: Earthquake Damage Prediction (`Task_1_Earthquake_Damage_Prediction/`)

We solve the challenge ["Richter's Predictor: Modeling Earthquake Damage"](https://www.drivendata.org/competitions/57/nepal-earthquake/).
It is an imbalanced ordinal classification/regression problem (with three classes), scored by micro-averaged F1 score (= accuracy here).
Most features are categorical, many of them binary.

### Preparation

To use our code, download the files `train_values.csv`, `test_values.csv`, and `train_labels.csv`
from the competition platform (no need to store the 4th file, which demonstrates the submission format).
Place the data files in the folder `data/` within the current folder.

### Exploration

The notebook `Exploration.ipynb` contains some basic exploration (mainly statistics) of the CSVs.

### Scoring

The competition itself uses the micro-averaged F1 score, which corresponds to accuracy for single-label prediction.
For course-internal scoring, we use Matthews Correlation Coefficient (MCC) instead.

- `split.py` creates one stratified holdout split.
- `score.py` scores submissions for the holdout split.

### Demo Submissions

We provide small demo scripts for creating submissions.
The submission format is valid for the competition website as well as our course-internal scoring.

- `predict_majority.py` creates a baseline solution that constantly predicts the majority class.
- `predict_tree.py` uses a `sklearn` decision tree (you can easily switch the model).
  The only preprocessing is encoding of categorical features.

## Task 2a: Active Learning for SAT Solving

This task stems from the field of [SAT solving](https://en.wikipedia.org/wiki/Boolean_satisfiability_problem).
We use features of SAT instances from the [Global Benchmark Database (`GBD`)](https://gbd.iti.kit.edu/)
and runtimes from the [SAT Competition 2022](https://satcompetition.github.io/2022/) `Anniversary Track`.
We have two prediction targets:

- Is the instance satisfiable or not (column `result` in database `meta`)?
- Does a solver run into a timeout on an instance or not (based on the runtime data)?

Besides exploring the data, students should use classification approaches.
We investigate a traditional "passive" as well as an active-learning scenario for predictions.
For active learning, we consider "standard" queries (which always return a label) as well as
"timeout" queries (which only run for a limited amount of time).
The latter can potentially save query costs but might not return a class label
(if the solver did not finish within the given timeout).

### Preparation

`prepare_data.py` pre-processes the dataset:

- download databases with meta data and instance features from `GBD`
- download runtime data from SAT-Competition website
- merge databases
- filter instances:
  - 2022 Anniversary Track
  - known satisfiability result
  - no NAs in instance features

### Exploration

The notebook `Active_Learning_Demo.ipynb` demonstrates "standard" active learning with `scikit-activeml`.
The notebook `Timeout_Active_Learning_Demo.ipynb` uses a timeout-aware active-learning oracle
that we implemented in the module `al_oracle`, while the query strategies still originate from `scikit-activeml`.

### Scoring

We provide a scoring mechanism for the "passive" prediction scenario, using MCC as metric:

- `split.py` creates one stratified holdout split for both targets, satisfiability and timeouts.
  From the 28 available solvers, we choose the winner of the Anniversary Track for timeout prediction
  (which is not the fastest solver on our filtered instance sample but close to it).
- `score.py` scores submissions for the holdout split, considering both prediction targets.

For the active-learning scenario, we do not have dedicated splitting and scoring scripts
since students should plot and compare complete learning curves.
Thus, students should determine the prediction quality for each learning iteration on their own.
For active learning with timeouts, `al_oracle.py` provides splitting and (MCC-based) scoring
functionality in the class `ALOracle`, but there still is no scoring script.

### Demo Submissions

We provide small demo prediction scripts for the "passive" scenario, using the same approaches as for Task 1:

- `predict_majority.py` creates a baseline solution that constantly predicts the majority class.
- `predict_tree.py` uses a `sklearn` decision tree (you can easily switch the model) without any preprocessing.

## Task 2b: Meta-Learning for Encoder Selection
