# Screen Time Battery Prediction

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Numerical%20Computing-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-Machine%20Learning-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-11557C?style=for-the-badge&logo=python&logoColor=white)
![Pytest](https://img.shields.io/badge/Pytest-Testing-0A9EDC?style=for-the-badge&logo=pytest&logoColor=white)

</div>

---

## Overview

A simple machine learning baseline that predicts smartphone battery percentage used from screen time, using Simple Linear Regression. Covers the full workflow — data loading, EDA, training, evaluation, prediction, and visualization — plus a test suite.

---

## Model

```text
battery_used = -1.423278 + 0.205933 × screen_time
```

---

## Dataset

10 sample observations of screen time vs. battery used.

| Screen Time (min) | Battery Used (%) |
| -----------------: | -----------------: |
| 10  | 2.0  |
| 30  | 5.0  |
| 45  | 6.5  |
| 60  | 10.0 |
| 75  | 14.0 |
| 90  | 18.0 |
| 105 | 20.0 |
| 120 | 23.0 |
| 135 | 26.0 |
| 150 | 30.0 |

---

## Results

| Metric | Value    |
| ------ | -------: |
| MSE    | 0.101410 |
| RMSE   | 0.318450 |
| R²     | 0.999080 |

---

## Usage

```python
from screen_time_battery.prediction import predict_battery_used

predict_battery_used(60)
```

---

## Installation

```bash
git clone <your-repository-url>
cd screen-time-battery-prediction
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

---

## Testing

```bash
pytest -v
```

30 tests covering data loading, EDA, model training, prediction, and visualization.

---

## Built With

- **Python**
- **NumPy**
- **Scikit-learn**
- **Matplotlib**
- **Pytest**

---

## Limitations

Real battery use also depends on brightness, app load, network activity, background processes, and hardware — none of which this model accounts for. It's a demo of a regression workflow, not a real predictor.

---
