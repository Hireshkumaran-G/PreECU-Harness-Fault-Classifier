# PreECU-Harness-Fault-Classifier
**AI-Driven Wiring Harness Fault Detection with Digital Twin Dashboard**

Developed during internship at **Ashok Leyland Ltd., Unit-II, Hosur** — Chassis Assembly Line 2
Internship Period: 20 April 2026 – 19 May 2026

---

## Overview

Detects and classifies faults in a 12-node vehicle wiring harness before ECU connection, using machine learning trained on a physics-informed synthetic dataset. Results are visualised through a real-time SCADA-style digital twin dashboard.

---

## Repository Structure

```
AI_Harness_Project/
│
├── harness_dataset_generator.py    # Physics-informed dataset generator
├── harness_classifier.py           # ML training, evaluation, artifact export
├── harness_dashboard.py            # Digital twin dashboard (SCADA UI)
├── environment.yml                 # Conda environment
├── requirements.txt                # pip dependencies
│
├── classifier_output/              # Generated on training run
│   ├── best_model.pkl
│   ├── model_metadata.pkl
│   ├── classification_report.txt
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   ├── model_comparison.png
│   ├── shap_random_forest.png
│   ├── shap_xgboost.png
│   └── ...
│
├── dataset_output/                 # Generated on dataset run
│   ├── harness_dataset_combined.csv
│   ├── harness_dataset_realistic.csv
│   ├── harness_dataset_demo.csv
│   └── ...
│
└── dashboard_logs/                 # Auto-saved dashboard renders
    └── dashboard_<timestamp>.png
```

---

## Fault Classes

| Class | Description | Severity |
|---|---|---|
| `HEALTHY` | Baseline — no fault | 0% |
| `OPEN_CIRCUIT` | Wire break → downstream voltage drop | 76% |
| `SHORT_TO_GROUND` | Node shorted to chassis ground | 92% |
| `WIRE_WIRE_SHORT` | Low-resistance bridge between non-adjacent wires | 61% |
| `HIGH_RESISTANCE` | Elevated junction resistance → progressive voltage sag | 38% |
| `ROUTING_MISMATCH` | Incorrect connector routing → branch asymmetry | 42% |

---

## Harness Topology (12-Node Network)

```
N0 (FuseBox / 5V Source)
└── N1 (Junction A)
        ├── N2 (Coolant Sensor)
        ├── N3 (Oil Pressure Sensor)
        └── N4 (Junction B)
                ├── N5 (Brake Signal)
                ├── N6 (Throttle Signal)
                └── N7 (Junction C)
                        ├── N8 (ABS Sensor)
                        ├── N9 (Fuel Sensor)
                        └── N10 (Junction D)
                                └── N11 (Terminal Node)
```

Wire resistance: 0.05 Ω/unit · Sensor load: 1000 Ω at N2, N3, N5, N6, N8, N9, N11

---

## Setup

### Option A — Conda (Recommended)
```bash
conda env create -f environment.yml
conda activate harness_ai
```

### Option B — pip
```bash
pip install -r requirements.txt
```

**Python 3.11 required**

---

## Quick Start

### Step 1 — Generate Dataset
```bash
python harness_dataset_generator.py
```
Generates 12,000 labelled samples (6 classes × 2,000) in `dataset_output/`.
Estimated time: 2–5 minutes.

### Step 2 — Train Classifier
```bash
python harness_classifier.py
```
Trains RF, DT, k-NN, SVM, XGBoost, and Ensemble models. Saves artifacts, SHAP plots, confusion matrix, and classification report to `classifier_output/`.
Estimated time: 10–25 minutes.

### Step 3 — Launch Dashboard
```bash
# Single fault scenario
python harness_dashboard.py --fault SHORT_TO_GROUND

# All available faults
python harness_dashboard.py --fault HEALTHY
python harness_dashboard.py --fault OPEN_CIRCUIT
python harness_dashboard.py --fault HIGH_RESISTANCE
python harness_dashboard.py --fault WIRE_WIRE_SHORT
python harness_dashboard.py --fault ROUTING_MISMATCH

# Demo mode — cycles through all faults
python harness_dashboard.py --demo
```

Dashboard renders are auto-saved to `dashboard_logs/`.

---

## Model Performance

| Model | Test Accuracy | Confident Accuracy (≥75%) | ROC-AUC |
|---|---|---|---|
| Random Forest | 80.08% | 98.06% | 0.9608 |
| Decision Tree | 81.08% | 85.33% | 0.9133 |
| k-NN (k=5) | 79.08% | 89.70% | 0.9465 |
| SVM (RBF) | 81.25% | 98.04% | 0.9586 |
| **XGBoost** | **82.17%** | **90.21%** | **0.9645** |
| **Ensemble (RF+XGB+SVM)** ★ | **81.92%** | **99.61%** | **0.9630** |

**Deployed model:** Ensemble (RF+XGB+SVM) — chosen for highest confident accuracy (99.6%) on production-critical fault types.
Confidence threshold: 75% — predictions below threshold return `UNCERTAIN`.

---

## Inference Example

```python
import pickle
import numpy as np
from harness_classifier import FaultConfirmer

# Load artifacts
with open('classifier_output/best_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('classifier_output/model_metadata.pkl', 'rb') as f:
    meta = pickle.load(f)
with open('classifier_output/selector.pkl', 'rb') as f:
    selector = pickle.load(f)

# Debounce — requires 3 consecutive consistent readings to confirm fault
confirmer = FaultConfirmer(n_required=3)

# Build feature vector from measured node values
# feat_dict = { 'V_N1': 4.99, 'V_N2': 0.45, ..., 'V_N1_tmean': 4.98, ... }
X = np.array([feat_dict[c] for c in meta['feature_cols']]).reshape(1, -1)
X = selector.transform(X)

pred      = model.predict(X)[0]
prob      = model.predict_proba(X).max()
label     = pred if prob >= meta['conf_threshold'] else 'UNCERTAIN'
confirmed = confirmer.update(label)

print(f"Fault: {label}  |  Confidence: {prob*100:.1f}%  |  Confirmed: {confirmed}")
```

---

## Explainability

SHAP values are computed using **Random Forest as a surrogate** for the deployed Ensemble model (RF+XGB+SVM does not natively support TreeExplainer).

Top SHAP drivers:
1. `routing_signature` — primary discriminator for ROUTING_MISMATCH
2. `branch_symmetry_error` — branch health indicator
3. `V_N2` / `V_N3` deviation — junction-level fault signal
4. `isolation_index` — overall network isolation health
5. `V_N2_tmean`, `V_N3_tstd` — temporal stability indicators

---

## Digital Twin

A companion circuit model (Altair Twin Activate 2026) replicates the 12-node harness in Simscape Electrical for circuit-level verification. Measured output at Node 2 under test fault: **3.24166V** — consistent with Python solver prediction.

---

## Tech Stack

`Python 3.11` · `scikit-learn` · `XGBoost` · `SHAP` · `Pandas` · `NumPy` · `Matplotlib` · `Altair Twin Activate`

---

## Authors

**Developer:** Hireshkumaran G, Chennai Institute of Technology
**Industry Partner:** Ashok Leyland Ltd., Unit-II, Hosur — Chassis Assembly Line 2
**Industry Mentor:** A. Rajeshkumar, Deputy Manager, Chassis Assembly Line 2

---

## Licence

Developed as part of an academic internship at Ashok Leyland Ltd. Provided for educational and research purposes. Commercial use requires explicit permission from Ashok Leyland Ltd.