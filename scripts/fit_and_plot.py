import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from ml_linear_regression_uni.linear import LinearRegressorUni


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--x", nargs="+", type=float, default=[1.0, 2.0])
    parser.add_argument("--y", nargs="+", type=float, default=[300.0, 500.0])
    parser.add_argument("--predict_x", type=float, default=1.2)
    args = parser.parse_args()

    x = np.array(args.x)
    y = np.array(args.y)

    model = LinearRegressorUni.from_two_points(x, y)
    print(f"Fitted w={model.w:.3f}, b={model.b:.3f}")

    xs = np.linspace(x.min() * 0.9, x.max() * 1.1, 100)
    plt.scatter(x, y, marker="x", label="Data")
    plt.plot(xs, np.asarray(model.predict(xs)), label="Model")
    plt.title("Housing Prices")
    plt.xlabel("Size (1000 sqft)")
    plt.ylabel("Price (in 1000s of dollars)")
    plt.legend()
    os.makedirs("reports", exist_ok=True)
    plt.savefig("reports/fit_and_plot.png", dpi=150)
    plt.show()

    pred = model.predict(np.array([args.predict_x]))[0]
    print(f"Prediction for x={args.predict_x}: ${pred:.0f} thousand")


if __name__ == "__main__":
    main()
