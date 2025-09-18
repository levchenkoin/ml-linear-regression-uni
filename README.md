# Machine Learning Specialization — Labs in Python

[![CI](https://github.com/levchenkoin/ml-linear-regression-uni/actions/workflows/ci.yml/badge.svg)](https://github.com/levchenkoin/ml-linear-regression-uni/actions/workflows/ci.yml)

A hands-on, from-scratch implementation of key machine learning algorithms in Python,  
based on concepts taught in the **Machine Learning Specialization by Andrew Ng (DeepLearning.AI)**.  

The code is refactored out of the original Jupyter notebooks into a clean, reproducible project structure with CLI scripts, tests, and CI.

## 🚀 Quick Start

Clone the repo and run training + plotting:

```bash
# 1. Clone the repo
git clone https://github.com/levchenkoin/ml-linear-regression-uni.git
cd ml-linear-regression-uni

# 2. Create virtual environment & install dependencies
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. Run training + plotting
python scripts/fit_and_plot.py --x 1.0 2.0 --y 300.0 500.0 --predict_x 1.2
```
## 📂 Project Structure

```bash
src/               # reusable code: model & cost functions
scripts/           # CLI scripts (fit_and_plot.py, predict.py)
tests/             # pytest tests for model correctness
requirements.txt   # dependencies
README.md          # this file
LICENSE            # MIT license
```
## ✨ Features

- Clean, reproducible implementation of core ML algorithms in **pure NumPy**

- **Linear regression**: model function $f_{w,b}(x)=wx+b$, cost $J(w,b)$

- Gradient descent training with cost curve visualization

- Closed-form solution from two points

- CLI-based scripts for training, prediction, and plotting

- Unit tests with `pytest`

- Continuous Integration via GitHub Actions

## 📖 Attribution

This project is **inspired by** the [Machine Learning Specialization](https://www.coursera.org/specializations/machine-learning-introduction), but all code here is **written by me** for learning and demonstration purposes.

## 📝 License

This project is licensed under the [MIT License](https://github.com/levchenkoin/ml-linear-regression-uni/tree/main?tab=MIT-1-ov-file).
