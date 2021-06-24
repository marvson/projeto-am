from __future__ import annotations
from dataclasses import dataclass
from scipy.stats import t, ttest_ind_from_stats

import numpy as np

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)


@dataclass
class SampleMean:
    """Holds statistical metrics about a sample of a population.

    It holds the sample size, mean value and standard deviation. These values
    are then used to compute confidence intervals, to compute hypothesis tests
    and others operations.

    Attributes
    ----------
    sample_size : int
        Number of samples taken from the population.
    mean: float
        Mean value of the samples.
    std: float
        Standard deviation of the samples.

    Methods
    -------
    confidence_interval(alpha=0.95)
        Returns the confidence interval of the sample mean.
    from_observations(observations)
        Creates a new SampleMean from a list of observations.

    Examples
    --------
    >>> SampleMean.from_observations([1, 3, 1, 3, 2, 2, 2, 1.5, 6]) < SampleMean.from_observations([1, 14, 0.2, 20, 3.4, 15, 12, 1.5, 5])
    False

    >>> SampleMean.from_observations([1, 3, 1, 3, 2, 2, 2, 1.5, 6])
    1.21 <= X <= 3.57

    >>> SampleMean(sample_size=30, mean=2, std=10).confidence_interval(0.95)
    (-22.542755705592434, 26.542755705592434)

    >>> SampleMean.from_observations([1, 3, 1, 3, 2, 2, 2, 1.5, 6]).mean
    2.388888888888889

    """

    sample_size: int
    mean: float
    std: float

    @classmethod
    def from_observations(cls, observations: list[float]):
        return cls(len(observations), np.mean(observations), np.std(observations))

    def __repr__(self):
        min_trust, max_trust = self.confidence_interval()
        return f"""
        Confidence interval:
        {min_trust:.2f} <= X <= {max_trust:.2f}
        
        Mean: {self.mean}
        Std: {self.std}
        Sample Size: {self.sample_size}
        """

    def __gt__(self, other):
        p_value = ttest_ind_from_stats(
            self.mean,
            self.std,
            self.sample_size,
            other.mean,
            other.std,
            other.sample_size,
            alternative="greater",
        )[1]
        return p_value > 0.05

    def __lt__(self, other):
        p_value = ttest_ind_from_stats(
            self.mean,
            self.std,
            self.sample_size,
            other.mean,
            other.std,
            other.sample_size,
            alternative="less",
        )[1]
        return p_value > 0.05

    def __eq__(self, other):
        p_value = ttest_ind_from_stats(
            self.mean,
            self.std,
            self.sample_size,
            other.mean,
            other.std,
            other.sample_size,
            alternative="two-sided",
        )[1]
        return p_value > 0.05

    def confidence_interval(self, alpha: float = 0.95):
        return t.interval(
            alpha=alpha,
            df=self.sample_size - 1,
            loc=self.mean,
            scale=self.std / np.sqrt(self.sample_size - 1),
        )


def ttest(s1: SampleMean, s2: SampleMean, alternative: str):
    """Does a t test of the two sample populations.

    Assuming dependent samples, which means, samples from the same classifier
    in different instants in time, then we perform a t-student test of the
    difference of the means of the two population samples s1 and s2, returning
    the t-value and p-value of the test. The degrees of freedom are calculated
    from the sample size.

    Parameters
    ----------
    s1 : SampleMean
        First population sample mean value.
    s2 : SampleMean
        Second population sample mean value.
    alternative : str
        Can be "less" if the alternative hypothesis is that mu(s1) < mu(s2),
        "greater" if the alternative hypothesis is that mu(s1) > mu(s2) and
        "two_sided" if the alternative hypothesis is that mu(s1) != mu(s2).

    Returns
    -------
    Tuple[float, float]
        Returns the values of (t_value, p_value) in this order as a tuple.
    """
    return ttest_ind_from_stats(
        s1.mean,
        s1.std,
        s1.sample_size,
        s2.mean,
        s2.std,
        s2.sample_size,
        alternative,
    )


def ttest_greater(s1: SampleMean, s2: SampleMean, alpha: float = 0.05):
    """Tests the hypothesis that s1 > s2.

    Does a hypothesis test (t-test) with the two samples. The null hypothesis
    is that mu(s1) <= mu(s2), the alternative hypothesis is that mu(s1) > mu(s2).
    The test returns an informative summary of the comparison under the specified
    alpha value.

    Parameters
    ----------
    s1 : SampleMean
        First population sample mean value.
    s2 : SampleMean
        Second population sample mean value.
    alpha : float, optional
        The confidence value, by default 0.05
    """
    t_value, p_value = ttest(s1, s2, "greater")
    return {
        "alpha": alpha,
        "H0": f"{s1.mean} <= {s2.mean}",
        "H1": f"{s1.mean} > {s2.mean}",
        "t_value": t_value,
        "p_value": p_value,
        "result": "reject H0" if p_value < alpha else "fail in rejecting H0",
    }


def ttest_less(s1: SampleMean, s2: SampleMean, alpha: float = 0.05):
    """Tests the hypothesis that s1 < s2.

    Does a hypothesis test (t-test) with the two samples. The null hypothesis
    is that mu(s1) >= mu(s2), the alternative hypothesis is that mu(s1) < mu(s2).
    The test returns an informative summary of the comparison under the specified
    alpha value.

    Parameters
    ----------
    s1 : SampleMean
        First population sample mean value.
    s2 : SampleMean
        Second population sample mean value.
    alpha : float, optional
        The confidence value, by default 0.05
    """
    t_value, p_value = ttest(s1, s2, "less")
    return {
        "alpha": alpha,
        "H0": f"{s1.mean} >= {s2.mean}",
        "H1": f"{s1.mean} < {s2.mean}",
        "t_value": t_value,
        "p_value": p_value,
        "result": "reject H0" if p_value < alpha else "fail in rejecting H0",
    }


def ttest_equal(s1: SampleMean, s2: SampleMean, alpha: float = 0.05):
    """Tests the hypothesis that s1 = s2.

    Does a hypothesis test (t-test) with the two samples. The null hypothesis
    is that mu(s1) = mu(s2), the alternative hypothesis is that mu(s1) != mu(s2).
    The test returns an informative summary of the comparison under the specified
    alpha value.

    Parameters
    ----------
    s1 : SampleMean
        First population sample mean value.
    s2 : SampleMean
        Second population sample mean value.
    alpha : float, optional
        The confidence value, by default 0.05
    """
    t_value, p_value = ttest(s1, s2, "two-sided")
    return {
        "alpha": alpha,
        "H0": f"{s1.mean} = {s2.mean}",
        "H1": f"{s1.mean} != {s2.mean}",
        "t_value": t_value,
        "p_value": p_value,
        "result": "reject H0" if p_value < alpha else "fail in rejecting H0",
    }


def compute_metrics(y_true: list[float], y_preds: list[list[float]]):
    """Given a summary of a model's prediction, calculates some metrics.

    The metrics computed are f-score, accuracy, precision and recall. The number
    of false positives, true positives, and others are summed to calculate the
    micro measure of the total score for all classes (since we have a multi-label
    classification problem).

    Parameters
    ----------
    y_true : list[float]
        The true values to be predicted.
    y_preds : list[list[float]]
        The predictions of the model, repeated with different partitions of the
        dataset for statistical significance.
    """
    precision = [precision_score(y_true, y_pred, average="micro") for y_pred in y_preds]
    recall = [recall_score(y_true, y_pred, average="micro") for y_pred in y_preds]
    f1 = [f1_score(y_true, y_pred, average="micro") for y_pred in y_preds]
    accuracy = [accuracy_score(y_true, y_pred) for y_pred in y_preds]

    return {
        "precision": SampleMean.from_observations(precision),
        "recall": SampleMean.from_observations(recall),
        "f_score": SampleMean.from_observations(f1),
        "accuracy": SampleMean.from_observations(accuracy),
    }
