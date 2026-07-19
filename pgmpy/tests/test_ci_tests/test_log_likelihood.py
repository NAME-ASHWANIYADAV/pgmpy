import numpy as np
import pandas as pd
import pytest

from pgmpy.ci_tests import LogLikelihood


@pytest.fixture
def test_ll():
    df_adult = pd.read_csv("pgmpy/tests/test_estimators/testdata/adult.csv")
    test = LogLikelihood(data=df_adult)

    return test


def test_discrete_tests(test_ll):
    assert not test_ll("Age", "Immigrant", [], significance_level=0.05)
    assert not test_ll("Age", "Race", [], significance_level=0.05)
    assert not test_ll("Age", "Sex", [], significance_level=0.05)
    assert not test_ll(
        "Education",
        "HoursPerWeek",
        ["Age", "Immigrant", "Race", "Sex"],
        significance_level=0.05,
    )
    assert test_ll("Immigrant", "Sex", [], significance_level=0.05)
    assert not test_ll("Education", "MaritalStatus", ["Age", "Sex"], significance_level=0.05)


def test_exactly_same_vars():
    x = np.random.choice([0, 1], size=1000)
    y = x.copy()
    df = pd.DataFrame({"x": x, "y": y})

    test = LogLikelihood(data=df)
    test("x", "y", [])
    assert test.dof_ == 1
    assert test.p_value_ == pytest.approx(0, abs=1e-2)


def test_empty_stratum_finite():
    # (A, B) = (1, 1) never occurs, so the conditioning product space contains
    # an empty stratum. The xlogy-based statistic must stay finite.
    df = pd.DataFrame(
        {
            "X": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            "Y": [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
            "A": [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0],
            "B": [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1],
        }
    )

    test = LogLikelihood(data=df)
    test.run_test("X", "Y", ["A", "B"])
    assert np.isfinite(test.statistic_)
    assert test.dof_ == 3
    assert test.p_value_ == pytest.approx(1.0)
