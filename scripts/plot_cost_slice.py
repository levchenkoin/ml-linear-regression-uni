import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from src.linear import cost

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--x", nargs="+", type=float, default=[1.0, 2.0])
    parser.add_argument("--y", nargs="+", type=float, default=[300.0, 500.0])
    parser.add_argument("--b", type=float, default=100.0)
    parser.add_argument("--wmin", type=float, default=0.0)
    parser.add_argument("--wmax", type=float, default=400.0)
    parser.add_argument("--steps", type=int, default=200)
    args = parser.parse_args()

    x = np.array(args.x)
    y = np.array(args.y)

    ws = np.linspace(args.wmin, args.wmax, args.steps)
    costs = [cost(x, y, w, args.b) for w in ws]

    plt.plot(ws, costs)
    plt.title(f"Cost vs w (b={args.b})")
    plt.xlabel("w")
    plt.ylabel("J(w,b)")
    plt.grid(True)

    os.makedirs("reports", exist_ok=True)
    plt.savefig("reports/cost_vs_w.png", dpi=150)
    plt.show()

if __name__ == "__main__":
    main()