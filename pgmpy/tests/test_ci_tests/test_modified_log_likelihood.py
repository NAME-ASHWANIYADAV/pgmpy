import numpy as np
import pandas as pd
import pytest

from pgmpy.ci_tests import ModifiedLogLikelihood


@pytest.fixture
def test_mll():
    df_adult = pd.read_csv("pgmpy/tests/test_estimators/testdata/adult.csv")
    test = ModifiedLogLikelihood(data=df_adult)

    return test


def test_discrete_tests(test_mll):
    assert not test_mll("Age", "Immigrant", [], significance_level=0.05)
    assert not test_mll("Age", "Race", [], significance_level=0.05)
    assert not test_mll("Age", "Sex", [], significance_level=0.05)
    assert not test_mll(
        "Education",
        "HoursPerWeek",
        ["Age", "Immigrant", "Race", "Sex"],
        significance_level=0.05,
    )

    assert test_mll("Immigrant", "Sex", [], significance_level=0.05)
    assert not test_mll("Education", "MaritalStatus", ["Age", "Sex"], significance_level=0.05)


def test_exactly_same_vars():
    x = np.random.choice([0, 1], size=1000)
    y = x.copy()
    df = pd.DataFrame({"x": x, "y": y})

    test = ModifiedLogLikelihood(data=df)
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

    test = ModifiedLogLikelihood(data=df)
    test.run_test("X", "Y", ["A", "B"])
    assert np.isfinite(test.statistic_)
    assert test.dof_ == 3
    assert test.p_value_ == pytest.approx(1.0)
