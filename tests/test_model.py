import numpy as np
from ml_linear_regression_uni.linear import f_wb, LinearRegressorUni, cost


def test_two_points_fit():
    x = np.array([1.0, 2.0])
    y = np.array([300.0, 500.0])
    model = LinearRegressorUni.from_two_points(x, y)
    assert abs(model.w - 200.0) < 1e-9
    assert abs(model.b - 100.0) < 1e-9
    assert np.allclose(f_wb(x, model.w, model.b), y)
    assert cost(x, y, model.w, model.b) == 0.0
