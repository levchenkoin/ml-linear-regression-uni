# Univariate Linear Regression — From Scratch

My hands-on implementation of univariate linear regression with Python, inspired by the **Machine Learning Specialization by Andrew Ng (DeepLearning.AI)**.  
This repository contains **my own code**, refactored outside the original notebooks, with reproducible structure, CLI scripts, and basic tests.

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

- Implementation of **linear model**: $f_{w,b}(x) = wx + b$

- **Cost function** $J(w,b)$ with vectorized computation

- Fitting model from two points (closed-form solution)

- CLI + Matplotlib plots

- Unit tests (`pytest`) for reproducibility

## 📖 Attribution

This project is **inspired by** the [Machine Learning Specialization](https://www.coursera.org/specializations/machine-learning-introduction), but all code here is **written by me** for learning and demonstration purposes.

## 📝 License

This project is licensed under the [MIT License](https://github.com/levchenkoin/ml-linear-regression-uni/tree/main?tab=MIT-1-ov-file).
