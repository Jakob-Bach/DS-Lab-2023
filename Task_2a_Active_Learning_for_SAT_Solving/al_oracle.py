"""Active-learning oracle for the SAT-solving task

Helper module. See class `ALOracle` for more information.
"""


from typing import Dict, Iterable, Optional, Tuple, Union

import pandas as pd
import sklearn.metrics
import sklearn.model_selection


DEFAULT_SOLVER = 'runtimes.Kissat_MAB_ESA'  # winner of 2022 SAT Competition's Anniversary Track
COMPETITION_TIMEOUT = 5000

LABEL_MISSING = float('nan')
LABEL_NOTIMEOUT = 0
LABEL_TIMEOUT = 1
LABEL_SAT = 0
LABEL_UNSAT = 1


class ALOracle:
    """Active-learning oracle

    Class for splitting the dataset, querying labels, and scoring the quality of predictions. You
    need to create an object of this class for all this functionality. After that, split the
    dataset first, since label queries and scoring depend on that particular data split
    (ground-truth labels and solver runtimes are stored within the object).
    """

    def __init__(self):
        """Default initializer

        Initialize private fields.

        Returns:
            None.
        """
        self.__runtimes_train = None  # needed to assess query costs during active learning
        self.__y_train = None
        self.__y_test = None

    def split_data(self, dataset: pd.DataFrame, target: str, test_size: float = 0.2,
                   random_state: int = 25) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create a holdout split

        Split the passed `dataset` with a (single) stratified holdout split. Use the base + gate
        features as "X" and, according to `target`, either the satisfiability result or whether a
        particular solver times out as "y". Return the feature part and internally store the target
        part of the data plus the actual solver runtimes. For "result" as `target`, consider the
        runtimes of the `DEFAULT_SOLVER` and discard all instances where this solver times out
        (since the solver cannot determine satisfiability for these instances).

        Args:
            dataset (pd.DataFrame): The dataset containing instance features and solver runtimes.
            target (str): The prediction target. Needs to be either "result" or one of the
                "runtimes." features.
            test_size (float): The fraction of instances going into the test set. Should be a
                value in [0,1), i.e. having no test set is possible.
            random_state (int, optional): A seed to ensure reproducibility of the holdout split.
                Defaults to 25.

        Returns:
            X_train (pd.DataFrame): The feature part of the training set.
            X_test (pd.DataFrame): The feature part of the test set.
        """
        if not isinstance(dataset, pd.DataFrame):
            raise ValueError('Method expects "dataset" to be a DataFrame, not a numpy array etc.')
        if target not in dataset.columns:
            raise ValueError('Desired "target" is not a column in the "dataset".')
        if (test_size < 0) or (test_size >= 1):
            raise ValueError('Size of the test set should be a relative value in [0,1).')
        if target == 'result':
            dataset = dataset[dataset[DEFAULT_SOLVER] != 2 * COMPETITION_TIMEOUT]
            runtimes = dataset[DEFAULT_SOLVER]
            y = dataset[target].replace({'unsat': LABEL_UNSAT, 'sat': LABEL_SAT})
        elif target.startswith('runtimes.'):
            runtimes = dataset[target]
            y = (runtimes == 2 * COMPETITION_TIMEOUT).replace(
                {False: LABEL_NOTIMEOUT, True: LABEL_TIMEOUT})
        else:
            raise ValueError('"target" needs to be the SAT result or the runtimes of a solver.')
        assert y.isin([0, 1]).all()  # all values properly encoded
        X = dataset[[x for x in dataset.columns if x.startswith('base.') or x.startswith('gate.')]]
        if test_size == 0:
            X_train, X_test, self.__y_train, self.__y_test, self.__runtimes_train = \
                X, None, y, None, runtimes
        else:
            X_train, X_test, self.__y_train, self.__y_test, self.__runtimes_train, _ = \
                sklearn.model_selection.train_test_split(
                    X, y, runtimes, test_size=test_size, shuffle=True, stratify=y,
                    random_state=random_state)
        return (X_train, X_test)

    def query_labels(
            self, query_indices: Iterable[int], query_timeouts: Optional[Iterable[float]] = None
            ) -> Iterable[Dict[str, Union[float, int]]]:
        """Query target labels

        Return the labels of the queried instances subject to a timeout. I.e., if the passed timeout
        is lower than the actual solver runtime, the returned label is `NaN`. Else, the actual label
        is returned. The query cost amounts to the minimum of the actual solver runtime and the
        passed timeout.

        Args:
            query_indices (Iterable[int]): The indices of the (training) instances for which labels
                should be returned.
            query_timeouts (Optional[Iterable[float]], optional): The timeouts for each label query.
                Needs to have the same length as `query_indices`. Defaults to None (in which case
                each solver is run till the timeout of the SAT Competition, which means that the
                actual label is guaranteed to be returned).

        Returns:
            Iterable[Dict[str, Union[float, int]]]: Label and query-cost information for the queried
                indices.
        """
        results = []
        if query_timeouts is None:
            query_timeouts = [COMPETITION_TIMEOUT] * len(query_indices)
        elif len(query_indices) != len(query_timeouts):
            raise ValueError('Length of query indices and of timeouts need to match.')
        for query_idx, query_timeout in zip(query_indices, query_timeouts):
            actual_runtime = self.__runtimes_train.iloc[query_idx]
            if query_timeout < actual_runtime:
                label = LABEL_MISSING
                cost = query_timeout
            else:
                label = self.__y_train.iloc[query_idx]
                cost = actual_runtime
            results.append({'query_index': query_idx, 'label': label, 'cost': cost})
        return results

    def score(self, y_pred: Iterable[float]) -> float:
        """Score predictions

        Compute the MCC score for the passed predictions.

        Args:
            y_pred (Iterable[float]): The predictions. Needs to match either the length of the
                training data or the test data (the corresponding ground truth will be chosen
                accordingly). Needs to contain only proper binary-classification labels, no
                placeholders for missing values (which might also be returned by the query method).

        Returns:
            float: The MCC score.
        """
        if len(y_pred) == len(self.__y_train):
            y_true = self.__y_train
        elif len(y_pred) == len(self.__y_test):
            y_true = self.__y_test
        else:
            raise ValueError('Length of "y_pred" needs to correspond to training or test data.')
        y_pred = pd.Series(y_pred)
        if not y_pred.isin([0, 1]).all():
            raise ValueError('Invalid label(s) in "y_pred".')
        return sklearn.metrics.matthews_corrcoef(y_true=y_true, y_pred=y_pred)
