import numpy as np

import pytest
import json
import jsonpickle

from metrics import (
    Statistic,
    ttest,
    ttest_greater,
    ttest_less,
    ttest_equal,
    compute_metrics,
)


def test_successfully_stores_statistics():
    """
    Sample size = 3
    Mean = 2
    Standard deviation = 1
    """
    s = Statistic(3, 2, 1)
    assert s.mean == 2
    assert s.std == 1
    assert s.sample_size == 3


def test_computes_mean_from_observations():
    """
    μ = (1 + 2 + 3 + 4)/4 = 10/4 = 2.5
    """
    s = Statistic.from_observations([1, 2, 3, 4])
    assert s.mean == 2.5


def test_computes_sampling_std_from_observations():
    """
    μ = 2.5
    std^2 = ((1 - 2.5)^2 + (2 - 2.5)^2 + (3 - 2.5)^2 + (4 - 2.5)^2)/(4 - 1)
    std^2 = ((-1.5)^2 + (-0.5)^2 + (0.5)^2 + (1.5)^2)/3
    std^2 = (2.25 + 0.25 + 0.25 + 2.25)/3 = 5/3
    std = sqrt(5/3)
    """
    s = Statistic.from_observations([1, 2, 3, 4])
    assert s.std == np.sqrt(5 / 3)


def test_computes_sample_size_from_observations():
    """
    Sample size of [1, 2, 3, 4] is 4, since there are 4 observations
    """
    s = Statistic.from_observations([1, 2, 3, 4])
    assert s.sample_size == 4


def test_confidence_interval_exact():
    """
    The function shouldn't fail if all samples are the same. The confidence
    interval should then be equal to the only sample value sampled instead of
    failing.
    """
    s = Statistic.from_observations([3, 3, 3, 3])
    assert s.confidence_interval() == (3, 3)


def test_confidence_interval():
    """
    Example taken from http://www.stat.yale.edu/Courses/1997-98/101/confint.htm:

    The dataset "Normal Body Temperature, Gender, and Heart Rate" contains 130
    observations of body temperature, along with the gender of each individual
    and his or her heart rate.

    Variable        N     Mean   StDev
    TEMP            130   98.249 0.733

    Let's assume that the sample was taken randomly, that the sampling distribution
    approximates a normal distribution and the samples are independent, so we
    can use the t statistic to estimate a confidence interval for the body
    temperature. Assume the significance level alpha of 95% (0.05).

    The number of degrees of freedom will be:
    dof = N - 1 = 129

    Since we want a confidence interval, the central area under the curve will
    have a p-value sum of 0.95, then there will be a 0.025 for each side
    remaining. Then, we want the critical t value for 129 degrees of freedom
    and the significance of 0.025. Using the cfd of the t distribution, we then
    have:
    t* = 1.978524

    Notice that at the link of the yale university, an approximate t-value was
    used by looking up a t-table and using 120 degrees of freedom.

    The sampling standard deviation then will be StDev/sqrt(N) = 0.733 / sqrt(130)

    Finally, our confidence interval will be:
    μ +- t* SEMean
    μ +- 1.978524 * 0.733 / sqrt(130)
    """
    mu = 98.249
    std = 0.733
    n = 130

    t = 1.978524
    sem = std / np.sqrt(n)
    s = Statistic(n, mu, std)

    min_true, max_true = (mu - t * sem, mu + t * sem)
    min_comp, max_comp = s.confidence_interval()

    # Since our original values have 3 decimal places, we could approximate up
    # to 3 decimal places for comparison, but instead, let's use the maximum of
    # precision we have from the t value (5 significant digits)
    assert pytest.approx(min_comp, 1e-5) == pytest.approx(min_true, 1e-5)
    assert pytest.approx(max_comp, 1e-5) == pytest.approx(max_true, 1e-5)


def test_ttest_two_tailed():
    """
    For this test, let's use an example as reference.
    https://en.wikipedia.org/wiki/Student%27s_t-test

    The sample values are:
    A1 = {30.02, 29.99, 30.11, 29.97, 30.01, 29.99}
    A2 = {29.89, 29.93, 29.72, 29.98, 30.02, 29.98}
    """
    a1 = [30.02, 29.99, 30.11, 29.97, 30.01, 29.99]
    a2 = [29.89, 29.93, 29.72, 29.98, 30.02, 29.98]

    s1 = Statistic.from_observations(a1)
    s2 = Statistic.from_observations(a2)

    t_value, p_value = ttest(s1, s2, "two-sided")

    assert pytest.approx(s1.mean - s2.mean, 1e-3) == 0.095
    assert pytest.approx(t_value, 1e-3) == 1.959
    assert pytest.approx(p_value, 1e-3) == 0.07857


def test_ttests_one_tailed():
    """
    For this test, let's use an example as reference.
    https://www.statstutor.ac.uk/resources/uploaded/unpaired-t-test.pdf
    """
    poultry = [
        129,
        132,
        102,
        106,
        94,
        102,
        87,
        99,
        170,
        113,
        135,
        142,
        86,
        143,
        152,
        146,
        144,
    ]
    beef = [
        186,
        181,
        176,
        149,
        184,
        190,
        158,
        139,
        175,
        148,
        152,
        111,
        141,
        153,
        190,
        157,
        131,
        149,
        135,
        132,
    ]

    s1 = Statistic.from_observations(poultry)
    s2 = Statistic.from_observations(beef)

    result = ttest(s1, s2, "less")
    assert result[1] < 0.001


def test_return_of_ttest_greater(snapshot):
    # Snapshot testing of the return of this function
    s1 = Statistic(30, 10, 2)
    s2 = Statistic(30, 8, 1)
    result = jsonpickle.encode(ttest_greater(s1, s2))
    snapshot.assert_match(json.dumps(result), "result.txt")


def test_return_of_ttest_less(snapshot):
    # Snapshot testing of the return of this function
    s1 = Statistic(30, 10, 2)
    s2 = Statistic(30, 8, 1)
    result = jsonpickle.encode(ttest_less(s1, s2))
    snapshot.assert_match(json.dumps(result), "result.txt")


def test_return_of_ttest_equal(snapshot):
    # Snapshot testing of the return of this function
    s1 = Statistic(30, 10, 2)
    s2 = Statistic(30, 8, 1)
    result = jsonpickle.encode(ttest_equal(s1, s2))
    snapshot.assert_match(json.dumps(result), "result.txt")


def test_return_of_compute_metrics(snapshot):
    # Snapshot testing of the return of this function
    s1 = [1, 2, 3, 3]
    s2 = [[2, 3, 3, 2], [2, 2, 3, 3], [3, 3, 3, 3]]
    result = jsonpickle.encode(compute_metrics(s1, s2))
    snapshot.assert_match(json.dumps(result), "result.txt")
