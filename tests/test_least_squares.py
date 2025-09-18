import numpy as np
from ml_linear_regression_uni.linear import fit_least_squares


def test_least_squares_two_points_matches_exact():
    x = np.array([1.0, 2.0])
    y = np.array([300.0, 500.0])
    w, b = fit_least_squares(x, y)
    assert abs(w - 200.0) < 1e-9
    assert abs(b - 100.0) < 1e-9
