# tests/test_plotting.py
import numpy as np
import matplotlib.pyplot as plt
import os
import pytest
from ml_linear_regression_uni.linear import fit_least_squares

@pytest.mark.plot
def test_fit_and_save_plot(tmp_path):
    x = np.array([1.0, 2.0])
    y = np.array([300.0, 500.0])
    w, b = fit_least_squares(x, y)

    xs = np.linspace(x.min()*0.9, x.max()*1.1, 50)
    ys = w*xs + b
    plt.plot(xs, ys); plt.scatter(x, y)
    out = tmp_path/"fit_preview.png"
    plt.savefig(out)
    assert out.exists() and out.stat().st_size > 0