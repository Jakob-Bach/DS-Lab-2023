# Data Science Laboratory Course 2023

This is the supervisor repo of the ["Data Science Laboratory Course"](https://dbis.ipd.kit.edu/3211_3244.php) at KIT in 2023.
Students worked on two subtasks:

- a practive competition from the platform [drivendata.org](https://www.drivendata.org/)
- one of two research tasks

The repo provides files for preparing the datasets, some basic exploration, course-internal splitting, scoring, and demo submissions for that.
Additionally, `Surveys/` contains exports of questionnaires (created with ILIAS `v7`) to evaluate the students' satisfication
(one survey at start of course, one after first task, one after second task).

## Setup

We use Python with version `3.10`.
We recommend to set up a virtual environment to install the dependencies, e.g., with `virtualenv`:

```bash
python -m virtualenv -p <path/to/right/python/executable> <path/to/env/destination>
```

or with `conda`:

```bash
conda create --name ds-lab-2023 python=3.10
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

## Task 2a: Active Learining for SAT Solving

## Task 2b: Meta-Learning for Encoder Selection
