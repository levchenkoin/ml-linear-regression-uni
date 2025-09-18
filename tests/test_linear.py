import numpy as np
from src.linear import f_wb, fit_least_squares, LinearRegressorUni, cost, gradient_descent

def test_two_points_fit():
    x = np.array([1.0, 2.0])
    y = np.array([300.0, 500.0])
    model = LinearRegressorUni.from_two_points(x, y)
    assert abs(model.w - 200.0) < 1e-9
    assert abs(model.b - 100.0) < 1e-9
    assert np.allclose(f_wb(x, model.w, model.b), y)
    assert cost(x, y, model.w, model.b) == 0.0

def test_least_squares_two_points_matches_exact():
    x = np.array([1.0, 2.0]); y = np.array([300.0, 500.0])
    w, b = fit_least_squares(x, y)
    assert abs(w - 200.0) < 1e-9
    assert abs(b - 100.0) < 1e-9

def test_gradient_descent_monotonic_nonincreasing_on_simple_data():
    # Простые данные с шумом
    rng = np.random.default_rng(0)
    x = np.linspace(0, 5, 50)
    y = 3.0 * x + 2.0 + rng.normal(0, 0.1, size=x.shape)
    w, b, hist = gradient_descent(x, y, w0=0.0, b0=0.0, alpha=0.05, epochs=200)
    # Стоимость должна убывать или не расти
    assert all(hist[i+1] <= hist[i] + 1e-9 for i in range(len(hist)-1))
    # И быть разумно малой
    assert cost(x, y, w, b) < cost(x, y, 0.0, 0.0)