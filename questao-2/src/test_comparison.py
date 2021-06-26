from comparison import Experiment, nonparametric_test_posthoc, compare_models
import numpy as np
import jsonpickle
import json


def test_compare_models_function(snapshot):
    snapshot.snapshot_dir = "snapshots"
    np.random.seed(42)
    experiments = [
        Experiment(
            np.random.choice(4, 30),
            ["m1", "m2", "m3"],
            [np.random.choice(4, 30), np.random.choice(4, 30), np.random.choice(4, 30)],
        )
        for k in range(16)
    ]
    result = jsonpickle.encode(compare_models(experiments))
    snapshot.assert_match(json.dumps(result), "compare_models.txt")


def test_nonparametric_test_posthoc_function(snapshot):
    snapshot.snapshot_dir = "snapshots"
    np.random.seed(42)
    data = np.random.randn(30, 4)
    result = jsonpickle.encode(nonparametric_test_posthoc(data))
    snapshot.assert_match(json.dumps(result), "friedman_nemenyi.txt")
