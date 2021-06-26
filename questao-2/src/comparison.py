from dataclasses import dataclass
from typing import Dict
import numpy as np

from scipy.stats import friedmanchisquare
from metrics import compute_raw_metrics
import scikit_posthocs as sp


@dataclass
class Experiment:
    y_true: list[int]
    model_names: list[str]
    predictions: list[list[int]]


def nonparametric_test_posthoc(data: np.ndarray, alpha: float = 0.05):
    """Friedman test and nemenyi posthoc test.

    For a series of k treatments (in our case, models) and b blocks (in our
    case, partitions of the dataset), it performs the ranking procedure and
    the friedman non-parametric chisquare test among the k treatments and b
    blocks.

    Parameters
    ----------
    data : np.ndarray
        A matrix where each row is a block and each column a treatment.
    alpha : float, optional
        Significance level, by default 0.05
    """
    p_value = friedmanchisquare(*data)[1]
    if p_value < alpha:
        p_values = sp.posthoc_nemenyi_friedman(data.T)
        return {
            "p_value": p_value,
            "H0": f"All sampling distributions are equal",
            "H1": f"Some distributions are different",
            "conclusion": "Reject H0",
            "post_hoc": p_values,
        }
    else:
        return {
            "p_value": p_value,
            "H0": f"All sampling distributions are equal",
            "H1": f"Some distributions are different",
            "conclusion": "Fail in rejecting H0",
        }


def compare_models(experiments: list[Experiment], alpha: float = 0.05):
    """Compare n models in repeated experiments over different data.

    With the provided significance, does the friedman chisquare test, followed
    by the nemenyi test when there is significant difference between the
    distributions. The comparison is done among the metrics: accuracy, recall,
    precision and f1. The result is a dictionary with a structure like a
    hypothesis test.

    Parameters
    ----------
    experiments : list[Experiment]
        A list of experiments of the same models over different datasets or
        partitions of the same dataset.
    alpha : float, optional
        Significance level, by default 0.05
    """
    labels = None

    accuracy = []
    precision = []
    recall = []
    f1 = []

    for experiment in experiments:
        if labels != None and experiment.model_names != labels:
            raise "Can't compare different experiments"
        else:
            labels = experiment.model_names

        metrics = compute_raw_metrics(experiment.y_true, experiment.predictions)

        precision.append(metrics[0])
        recall.append(metrics[1])
        f1.append(metrics[2])
        accuracy.append(metrics[3])

    accuracy = nonparametric_test_posthoc(np.array(accuracy), alpha)
    precision = nonparametric_test_posthoc(np.array(precision), alpha)
    recall = nonparametric_test_posthoc(np.array(recall), alpha)
    f1 = nonparametric_test_posthoc(np.array(f1), alpha)

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}
