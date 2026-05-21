# AI-Driven Pre-ECU Harness Validation System
### Digital Twin Fault Analytics — Chassis Assembly Line 2
**Ashok Leyland Ltd., Unit-II, Hosur**

---

## Project Overview

This repository contains the complete implementation of an **AI-driven, physics-informed wiring harness fault detection and classification system** developed for the Pre-ECU Validation Stage of **Chassis Assembly Line 2** at Ashok Leyland Unit-II, Hosur.

The system detects and classifies six fault types in a 12-node harness network using machine learning models trained on a physics-informed synthetic dataset, and presents results via a real-time industrial SCADA-style digital twin dashboard.

---

## Repository Structure

```
harness_ai/
├── harness_dataset_generator.py    # Physics-informed fault dataset generator (v6.1)
├── harness_classifier.py           # ML training, evaluation, and artifact export (v6.1)
├── harness_dashboard.py            # Digital twin dashboard — SCADA UI (stable version)
├── harness_dashboard_1.py          # Digital twin dashboard — extended industrial UI
├── environment.yml                 # Conda environment specification
├── requirements.txt                # pip dependencies
├── classification_report.txt       # Model evaluation report (auto-generated)
├── README.md                       # This file
│
├── dataset_output/                 # Generated dataset files (created on first run)
│   ├── harness_dataset_realistic.csv
│   ├── harness_dataset_demo.csv
│   ├── harness_dataset_combined.csv
│   └── *.png                       # Distribution and analysis plots
│
├── classifier_output/              # Trained model artifacts (created on first run)
│   ├── best_model.pkl
│   ├── scaler.pkl
│   ├── label_encoder.pkl
│   ├── selector.pkl
│   ├── temporal_scaler.pkl
│   ├── location_model.pkl
│   ├── model_metadata.pkl
│   ├── classification_report.txt
│   ├── model_accuracy.csv
│   ├── top_features.csv
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   ├── model_comparison.png
│   ├── fault_decision_boundaries.png
│   ├── shap_*.png
│   └── calibration_*.png
│
└── dashboard_logs/                 # Auto-saved dashboard renders (created on run)
    └── dashboard_<timestamp>.png
```

---

## Fault Classes

| Class | Description | Confidence | Severity |
|---|---|---|---|
| `HEALTHY` | No fault — baseline voltage profile | 99.5% | 0% |
| `OPEN_CIRCUIT` | Wire segment break → downstream voltage drop | 94.2% | 76% |
| `SHORT_TO_GROUND` | Node shorted to chassis ground | 97.6% | 92% |
| `WIRE_WIRE_SHORT` | Low-resistance bridge between non-adjacent wires | 95.5% | 61% |
| `HIGH_RESISTANCE` | Elevated junction resistance → progressive voltage sag | 91.2% | 38% |
| `ROUTING_MISMATCH` | Incorrect connector routing → branch asymmetry | 94.0% | 42% |

---

## Harness Topology

```
N0 (FuseBox/Source 5V)
    │
    └── N1 (Junction A)
            ├── N2 (Coolant Sensor Line)
            ├── N3 (Oil Pressure Sensor Line)
            └── N4 (Junction B)
                    ├── N5 (Brake Signal Line)
                    ├── N6 (Throttle Signal Line)
                    └── N7 (Junction C)
                            ├── N8 (ABS Sensor Line)
                            ├── N9 (Fuel Sensor Line)
                            └── N10 (Junction D)
                                    └── N11 (Terminal Node)
```

Wire resistance: 0.05 Ω/unit length (realistic mode). Sensor load: 1000 Ω at N2, N3, N5, N6, N8, N9, N11.

---

## Environment Setup

### Option A — Conda (Recommended)

```bash
conda env create -f environment.yml
conda activate harness_ai
```

### Option B — pip

```bash
pip install -r requirements.txt --break-system-packages
```

### Optional — XGBoost & SHAP via conda-forge

```bash
conda install -c conda-forge xgboost shap
```

**Python version required:** 3.11

---

## Quick Start — Run the Full Pipeline

### Step 1 — Generate Dataset

```bash
python harness_dataset_generator.py
```

Generates 12,000 labelled harness samples (6 classes × 2,000 samples each) in two modes:
- `realistic`: Low-noise, physics-accurate (sensor load 1000 Ω, noise 0.0005V)
- `demo`: Higher-noise demonstration mode (sensor load 100 Ω, noise 0.005V)

Output files are saved to `dataset_output/`.

**Estimated time:** 2–5 minutes

### Step 2 — Train Classifier

```bash
python harness_classifier.py
```

Trains 6 ML models (RF, DT, k-NN, SVM, XGBoost, Ensemble) on the generated dataset. Saves model artifacts, SHAP plots, confusion matrices, feature importance, and classification report to `classifier_output/`.

**Estimated time:** 10–25 minutes (depending on hardware)

### Step 3 — Launch Dashboard

```bash
# Single fault scenario (default: SHORT_TO_GROUND)
python harness_dashboard.py --fault SHORT_TO_GROUND

# Available faults:
python harness_dashboard.py --fault HEALTHY
python harness_dashboard.py --fault OPEN_CIRCUIT
python harness_dashboard.py --fault SHORT_TO_GROUND
python harness_dashboard.py --fault HIGH_RESISTANCE
python harness_dashboard.py --fault WIRE_WIRE_SHORT
python harness_dashboard.py --fault ROUTING_MISMATCH

# Demo mode — cycles through all faults
python harness_dashboard.py --demo
```

Dashboard PNG files are auto-saved to `dashboard_logs/`.

#### Alternative dashboard (extended industrial UI):

```bash
python harness_dashboard_1.py --fault SHORT_TO_GROUND
python harness_dashboard_1.py --demo
```

---

## Inference — Using Trained Model in Production

```python
import pickle
import numpy as np
from harness_classifier import FaultConfirmer, compute_physics_severity

# Load artifacts
with open('classifier_output/best_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('classifier_output/model_metadata.pkl', 'rb') as f:
    meta = pickle.load(f)

# Load selector (FIX-9)
selector = None
if meta.get('has_selector'):
    with open('classifier_output/selector.pkl', 'rb') as f:
        selector = pickle.load(f)

# Debounce confirmer (N=3 consecutive readings required to confirm fault)
confirmer = FaultConfirmer(n_required=3)

feat_cols = meta['feature_cols']

# Populate feat_dict with measured values (static + temporal stats)
# feat_dict = { 'V_N1': 4.99, 'V_N2': 0.45, ..., 'V_N1_tmean': 4.98, ... }
X = np.array([feat_dict[c] for c in feat_cols]).reshape(1, -1)

if selector is not None:
    X = selector.transform(X)

pred  = model.predict(X)[0]
prob  = model.predict_proba(X).max()
label = pred if prob >= meta['conf_threshold'] else 'UNCERTAIN'
confirmed = confirmer.update(label)

print(f"Label: {label}  |  Confirmed: {confirmed}  |  Confidence: {prob*100:.1f}%")
```

---

## Model Performance Summary (v6.1)

| Model | Test Accuracy | Confident Acc (≥75%) | CV Accuracy | MCC | ROC-AUC |
|---|---|---|---|---|---|
| Random Forest | 80.08% | 98.06% | 80.96% ±1.11% | 0.7613 | 0.9608 |
| Decision Tree | 81.08% | 85.33% | 79.90% ±1.02% | 0.7745 | 0.9133 |
| k-NN (k=5) | 79.08% | 89.70% | 79.83% ±0.70% | 0.7492 | 0.9465 |
| SVM (RBF) | 81.25% | 98.04% | 81.10% ±1.23% | 0.7812 | 0.9586 |
| **XGBoost ★** | **82.17%** | **90.21%** | **82.79% ±1.55%** | **0.7861** | **0.9645** |
| Ensemble (RF+XGB+SVM) | 81.92% | **99.61%** | 81.94% ±0.96% | 0.7834 | 0.9630 |

**Deployed model:** Ensemble (RF+XGB+SVM) — selected for highest confident accuracy (99.6%) on production-critical fault types.

**Confidence threshold:** 75% (below threshold → `UNCERTAIN` label returned)

---

## Feature Engineering (v6.1 — 60 Selected Features)

| Category | Features | Count |
|---|---|---|
| Node Voltages | V_N1 through V_N11 | 11 |
| Deviation (%) | DEV_N1_pct through DEV_N11_pct | 11 |
| Sensor Currents | I_N2, N3, N5, N6, N8, N9, N11 (mA) | 7 |
| Aggregate Stats | V_min, V_max, V_range, V_mean, V_std, dev_max_abs, nodes_above_threshold, worst_node_isolation | 8 |
| Topology Signatures | routing_signature, branch_symmetry_error, isolation_index, signal_uniformity, max_branch_delta, cross_coupling_estimate | 6 |
| Branch Differentials | N2_N3_diff, N5_N6_diff, N8_N9_diff | 3 |
| Environmental | temperature_C | 1 |
| Temporal Stats | V_Nn_tmean, tstd, tdelta, trange (4 stats × 11 nodes) | 44 |
| **Total candidates** | | **104** |
| **Selected (SelectKBest MI)** | | **60** |

> **Leakage controls:** `worst_node_idx`, `severity`, and `fault_detected` are excluded from training features (v6 fix).

---

## Dataset Generator — Key Parameters

| Parameter | Realistic Mode | Demo Mode |
|---|---|---|
| Source Voltage | 5.0V | 5.0V |
| Wire Resistance | 0.05 Ω/unit | Map-based (5–12 Ω) |
| Sensor Load | 1000 Ω | 100 Ω |
| Noise Std | 0.0005V | 0.005V |
| HRJ Range | 20–200 Ω | 40–400 Ω |
| ADC Resolution | 0.005V steps | 0.005V steps |
| Connector Fault Probability | 8% | 8% |
| Intermittent Fault Probability | 12% | 12% |
| Multi-fault Probability | 5% | 5% |
| Temporal Steps | 5 | 5 |

---

## v6.1 Fixes Applied

| Fix | Description |
|---|---|
| FIX-1 | Temporal feature columns explicitly named and appended to feat_cols |
| FIX-2 | demo_inference() uses same featurise() helper as training pipeline |
| FIX-3 | feat_cols updated AFTER selector applied using get_support() |
| FIX-4 | ENABLE_ENSEMBLE_CALIBRATION = False by default |
| FIX-5 | evaluate_persistent_faults() removed from classifier evaluation |
| FIX-6 | Branch groups fixed to [0,1,2],[3,4,5],[6,7,8],[9,10] |
| FIX-7 | ROC-AUC reorders probability columns to match FAULT_LABELS order |
| FIX-8 | Temporal features scaled with dedicated StandardScaler before SelectKBest |
| FIX-9 | selector.pkl saved and loaded in inference/dashboard |
| FIX-10 | Report header version bumped to v6.1 |
| FIX-A | worst_node_idx excluded (leakage ablation) |
| FIX-B | signal_uniformity = mean(per-branch variance) |
| FIX-C | Physics-informed severity formula |
| FIX-D | SHAP documented as surrogate explainability for ensemble |
| FIX-E | FaultConfirmer debounce class |
| FIX-F | Ensemble calibration flag |
| FIX-G | Accuracy wording corrected in report header |
| FIX-H | Node count (NUM_NODES=12) clarified in metadata |

---

## Simulink Model

A companion Simulink model (Altair Twin Activate 2026) is included for circuit-level verification. The model replicates the 12-node harness backbone in Simscape Electrical and includes:

- **Test Signal Injection** subsystem
- **Harness Backbone** subsystem (N0–N11 resistive network)
- **Diagnostic Conditioning** subsystem (per-node signal conditioning)
- **Output Monitoring** subsystem (voltage displays + oscilloscopes)
- **Signal Monitoring** subsystem (fault detection logic blocks)

Measured output at Node 2 (Diag_OUT_2): **3.24166V** under test fault — consistent with Python solver prediction, confirming physical model accuracy.

---

## Explainability

SHAP values are computed using **Random Forest as a surrogate model** for the deployed Ensemble predictor (RF+XGB+SVM does not natively support TreeExplainer). Top SHAP drivers:

1. `routing_signature` — primary discriminator for ROUTING_MISMATCH
2. `branch_symmetry_error` — branch health indicator
3. `V_N2` / `V_N3` deviation — junction-level fault signal
4. `isolation_index` — overall network isolation health
5. `V_N2_tmean`, `V_N3_tstd` — temporal stability indicators

---

## Authors & Acknowledgements

**Developer:** HIRESHKUMARAN G  
**Institution:** Chennai Institute of Technology
**Industry Partner:** Ashok Leyland Ltd., Unit-II, Hosur — Chassis Assembly Line 2  
**Internship Period:** 20.04.2026 – 19.05.2026  
**Industry Mentor:** A. Rajeshkumar - Deputy Manager, Chassis Assembly Line 2  

---

## Licence

This software was developed as part of an academic internship project at Ashok Leyland Ltd. It is provided for educational and research purposes. Commercial use requires explicit permission from Ashok Leyland Ltd.

---

*Harness AI v6.1 — Pre-ECU Validation System — Chassis Assembly Line 2*
