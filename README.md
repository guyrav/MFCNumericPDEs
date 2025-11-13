# MFC Numerical Analysis Project - Burgers Equation

This repository give the implementation of the FTCS scheme and other numerical schemes for solving the viscous Burgers equation, and experiments that demonstrate its properties.

Written by Andrea Suarez and Guy Raveh.

---

## Contents

| File                    | Description                                      |
|-------------------------|--------------------------------------------------|
| `requirements.txt`      | Libraries required for running this code         |
| `main.py`               | Entry point for running the experiments          |
| `schemes.py`            | Numerical scheme implementations                 |
| `params.py`             | Parameter classes for the numerical schemes      |
| `experiments.py`        | Implementations of numerical experiments         |
| `initial_conditions.py` | Initial conditions to be used in the experiments |
| `state_properties.py`   | Utilities for analyzing function values          |
| `plots.py`              | Functions for plotting experiment results        |
| `report.pdf`            | Write-up of results and discussion               |

---

## Requirements

See `requirements.txt`

---

## Installation

Run `pip install -r requirements.txt`

---

## Usage

To run different experiments, use the functions in `main.py` by uncommenting the relevant function calls in the `main()` function. Then run `main.py`.