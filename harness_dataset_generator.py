"""
harness_dataset_generator.py  v6.1
=================================
AI-Driven Pre-ECU Harness Integrity and Routing Validation System
Ashok Leyland — Chassis Assembly Line 2

v6 fixes vs v5
──────────────
  FIX-A  worst_node_idx REMOVED — implicit class shortcut / leakage risk
  FIX-B  signal_uniformity redefined as mean(per-branch variance) — no longer
         duplicates V_std
  FIX-C  Semi-physics severity: weighted_voltage_drop + branch_instability +
         isolation_error + routing_anomaly  (analysis only, not in X)
  FIX-D  Node count clarified: SOURCE_NODE N0 (FuseBox) +
         11 harness measurement nodes N1..N11
  FIX-E  5-step temporal voltage sequence columns V_Nn_t0..t4 added per sample
"""

import numpy as np
import csv
import os
import random

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    PLOT = True
except ImportError:
    PLOT = False

OUT = "dataset_output"
os.makedirs(OUT, exist_ok=True)

np.random.seed(42)
random.seed(42)

# ── TOPOLOGY ──────────────────────────────────────────────────────────────────
# FIX-D: 1 source/fusebox node (N0) + 11 harness measurement nodes (N1-N11)
NUM_NODES    = 12
NUM_HARNESS  = 11
SOURCE_NODE  = 0
SENSOR_NODES = [2, 3, 5, 6, 8, 9, 11]

BASE_SEGS = [
    (0,  1,  0.50), (1,  4,  0.60), (4,  7,  0.70), (7,  10, 0.60),
    (1,  2,  0.30), (1,  3,  0.35), (4,  5,  0.55), (4,  6,  0.65),
    (7,  8,  0.30), (7,  9,  0.40), (10, 11, 0.50),
]

NODE_NAMES = [
    "N0_FuseBox", "N1_Junc1", "N2_SensorA", "N3_SensorB",
    "N4_Junc2",   "N5_SensorC","N6_SensorD", "N7_Junc3",
    "N8_SensorE", "N9_SensorF","N10_Junc4",  "N11_SensorG",
]

BRANCH_GROUPS = [
    [1, 2, 3],      # Branch_A
    [4, 5, 6],      # Branch_B
    [7, 8, 9],      # Branch_C
    [10, 11],       # Branch_D
]

OPEN_SEGS        = [(1, 2), (1, 3), (4, 5), (4, 6), (7, 8), (7, 9), (10, 11)]
GROUND_NODES_P   = [2, 3, 4, 5, 6, 7, 8, 9]
WIRE_SHORT_PAIRS = [(2, 3), (5, 6), (8, 9)]
HRJ_SEGS         = [(1, 4), (4, 7), (7, 8), (7, 9), (10, 11)]
ROUTING_SWAPS    = [(2, 3), (5, 6), (8, 9), (2, 9), (3, 8)]

FAULT_LABELS = {
    0: "HEALTHY", 1: "OPEN_CIRCUIT",    2: "SHORT_TO_GROUND",
    3: "WIRE_WIRE_SHORT", 4: "HIGH_RESISTANCE", 5: "ROUTING_MISMATCH",
}

MODES = {
    "realistic": {
        "source_v": 5.0, "wire_res": 0.05, "sensor_load": 1000.0,
        "noise_std": 0.0005,
        "fp": {
            "open_R": 1e8, "gnd_R": 0.001, "ww_R": 0.001,
            "hrj_min": 20.0, "hrj_max": 200.0,
        },
    },
    "demo": {
        "source_v": 5.0,
        "wire_res_map": {
            (0,1):10,(1,4):10,(4,7):12,(7,10):10,
            (1,2):5,(1,3):6,(4,5):9,(4,6):11,(7,8):5,(7,9):7,(10,11):8,
        },
        "sensor_load": 100.0, "noise_std": 0.005,
        "fp": {
            "open_R": 1e8, "gnd_R": 0.001, "ww_R": 0.001,
            "hrj_min": 40.0, "hrj_max": 400.0,
        },
    },
}

# ── SOLVER ────────────────────────────────────────────────────────────────────
def solve(segments, sensor_load, source_v=5.0, forced_nodes=None, custom_loads=None):
    G = np.zeros((NUM_NODES, NUM_NODES))
    for i, j, R in segments:
        R = max(float(R), 1e-12); g = 1.0 / R
        G[i][i] += g; G[j][j] += g; G[i][j] -= g; G[j][i] -= g
    loads = custom_loads if custom_loads else {n: sensor_load for n in SENSOR_NODES}
    for node, lr in loads.items():
        G[node][node] += 1.0 / max(float(lr), 1e-12)
    I = np.zeros(NUM_NODES)
    Gm = G.copy()
    Gm[SOURCE_NODE, :] = 0.0; Gm[SOURCE_NODE, SOURCE_NODE] = 1.0
    I[SOURCE_NODE] = source_v
    if forced_nodes:
        for node, v in forced_nodes.items():
            Gm[node, :] = 0.0; Gm[node, node] = 1.0; I[node] = v
    try:
        return np.linalg.solve(Gm, I)
    except Exception:
        return np.zeros(NUM_NODES)

def base_segs(cfg, wv=1.0):
    if "wire_res" in cfg:
        return [(i, j, cfg["wire_res"] * l * wv) for i, j, l in BASE_SEGS]
    return [(i, j, cfg["wire_res_map"][(i, j)] * wv) for i, j, _ in BASE_SEGS]

_href = {}
def healthy_ref(cfg):
    k = id(cfg)
    if k not in _href:
        V = solve(base_segs(cfg, 1.0), cfg["sensor_load"], cfg["source_v"])
        _href[k] = V[1:]
    return _href[k]

# ── CONNECTOR / INTERMITTENT ──────────────────────────────────────────────────
def apply_connector_fault(Vn, node):
    fault_type = random.choice(["loose_pin", "oxidized", "partial_insert"])
    if fault_type == "loose_pin":
        Vn[node] *= np.random.uniform(0.88, 0.97)
    elif fault_type == "oxidized":
        Vn[node] = max(0.0, Vn[node] - np.random.uniform(0.05, 0.25))
    else:
        Vn[node] *= np.random.uniform(0.70, 0.90)
    return Vn, fault_type

def apply_intermittent(Vn, node, drop_prob=0.35):
    triggered = False
    if np.random.rand() < drop_prob:
        Vn[node] = max(0.0, Vn[node] - np.random.uniform(0.10, 0.50))
        triggered = True
    return Vn, triggered

# ── FAULT BUILDERS ────────────────────────────────────────────────────────────
def build_segs_for_fault(cfg, fault_id, wv):
    bs = base_segs(cfg, wv)
    fp = cfg["fp"]; sl = cfg["sensor_load"]; sv = cfg["source_v"]
    cl = None; fn = None; loc = "None"
    if fault_id == 0:
        return bs, sl, sv, fn, cl, "None"
    elif fault_id == 1:
        seg = random.choice(OPEN_SEGS)
        segs = [(i, j, r) for i, j, r in bs if not (i == seg[0] and j == seg[1])]
        segs.append((seg[0], seg[1], fp["open_R"]))
        return segs, sl, sv, fn, cl, f"Open_N{seg[0]}_N{seg[1]}"
    elif fault_id == 2:
        node = random.choice(GROUND_NODES_P)
        return bs, sl, sv, {node: 0.0}, cl, f"GND_N{node}"
    elif fault_id == 3:
        pair = random.choice(WIRE_SHORT_PAIRS)
        segs = list(bs); segs.append((pair[0], pair[1], fp["ww_R"]))
        return segs, sl, sv, fn, cl, f"WW_N{pair[0]}_N{pair[1]}"
    elif fault_id == 4:
        seg = random.choice(HRJ_SEGS)
        hrj_R = np.random.uniform(fp["hrj_min"], fp["hrj_max"])
        segs = [(i, j, r) for i, j, r in bs if not (i == seg[0] and j == seg[1])]
        segs.append((seg[0], seg[1], hrj_R))
        return segs, sl, sv, fn, cl, f"HRJ_N{seg[0]}_N{seg[1]}_{hrj_R:.0f}R"
    elif fault_id == 5:
        pair = random.choice(ROUTING_SWAPS); na, nb = pair
        Ra = sl * np.random.uniform(0.30, 0.60)
        Rb = sl * np.random.uniform(1.80, 3.50)
        cl_new = {n: sl for n in SENSOR_NODES}
        cl_new[na] = Rb; cl_new[nb] = Ra
        segs = list(bs); segs.append((na, nb, np.random.uniform(5.0, 25.0)))
        return segs, sl, sv, fn, cl_new, f"Route_Swap_N{na}_N{nb}"
    return bs, sl, sv, fn, cl, "Unknown"

def apply_routing_post_physics(Vn, pair):
    na, nb = pair
    coupling = np.random.uniform(0.05, 0.20)
    va, vb = Vn[na], Vn[nb]
    Vn[na] += coupling * (vb - va); Vn[nb] += coupling * (va - vb)
    Vn[na] *= np.random.uniform(0.75, 0.92)
    Vn[nb]  = min(Vn[nb] * np.random.uniform(1.03, 1.10), 5.0)
    affected = [n for g in BRANCH_GROUPS for n in g if na in g or nb in g]
    bg = np.random.uniform(0.85, 0.95)
    for k in affected:
        if k != na and k != nb and k < len(Vn):
            Vn[k] *= bg
    isolated = na if Vn[na] < Vn[nb] else nb
    Vn[isolated] = max(0.0, Vn[isolated] - np.random.uniform(0.05, 0.20))
    return Vn

# ── FIX-C: SEMI-PHYSICS SEVERITY ─────────────────────────────────────────────
def compute_physics_severity(devs, branch_sym, iso_idx, routing_sig):
    """
    Physics-informed severity [0..100]:
      0.45 * weighted_voltage_drop  (max |deviation| / 100)
      0.25 * branch_instability     (branch std / 1.0 V)
      0.15 * isolation_error        (|1 - iso_idx|)
      0.15 * routing_anomaly        (routing_sig / 2.0 V)
    """
    da = [abs(d) for d in devs]
    wvd = min(1.0, max(da) / 100.0)
    bi  = min(1.0, branch_sym / 1.0)
    ie  = min(1.0, abs(1.0 - iso_idx))
    ra  = min(1.0, routing_sig / 2.0)
    weights = np.array([0.45, 0.25, 0.15, 0.15])
    weights += np.random.normal(0, 0.015, 4)
    weights = np.clip(weights, 0.05, 0.70)
    weights /= weights.sum()
    score = (
        weights[0] * wvd +
        weights[1] * bi +
        weights[2] * ie +
        weights[3] * ra
    )
    return round(min(100.0, score * 100.0), 2)

# ── FIX-A/B: ENGINEERED FEATURES ─────────────────────────────────────────────
def compute_engineered(Vn_nodes, devs, sl, href):
    """
    FIX-A: worst_node_idx REMOVED.
    FIX-B: signal_uniformity = mean of per-branch voltage variance.
    Returns: (eng_dict, routing_sig, branch_sym, iso_idx, cross)
    """
    V = np.array(Vn_nodes)
    da = [abs(d) for d in devs]
    wi = int(np.argmax(da))
    oth = [da[k] for k in range(len(da)) if k != wi]

    eng = {
        "V_min":                round(float(V.min()), 6),
        "V_max":                round(float(V.max()), 6),
        "V_range":              round(float(V.max() - V.min()), 6),
        "V_mean":               round(float(V.mean()), 6),
        "V_std":                round(float(V.std()), 6),
        "dev_max_abs":          round(max(da), 4),
        "nodes_above_threshold":sum(1 for d in da if d > 8.0),
        "N5_N6_diff":           round(abs(V[4] - V[5]), 6),
        "N2_N3_diff":           round(abs(V[1] - V[2]), 6),
        "N8_N9_diff":           round(abs(V[7] - V[8]), 6),
        "worst_node_isolation": round(da[wi] - np.mean(oth) if oth else da[wi], 4),
        # FIX-A: worst_node_idx removed
    }

    routing_sig = abs(V[1]-V[2]) + abs(V[4]-V[5]) + abs(V[7]-V[8])
    eng["routing_signature"] = round(float(routing_sig), 6)

    branch_means = []
    for g in BRANCH_GROUPS:
        idxs = [n - 1 for n in g if 1 <= n <= 11]
        if idxs:
            branch_means.append(float(np.mean(V[idxs])))
    branch_sym = float(np.std(branch_means)) if branch_means else 0.0
    eng["branch_symmetry_error"] = round(branch_sym, 6)

    vmax = float(V.max())
    iso_idx = float(V.min()) / vmax if vmax > 1e-6 else 0.0
    eng["isolation_index"] = round(iso_idx, 6)

    # FIX-B: signal_uniformity = mean of per-branch variance (not global V_std)
    branch_vars = []
    for g in BRANCH_GROUPS:
        idxs = [n - 1 for n in g if 1 <= n <= 11]
        if len(idxs) > 1:
            branch_vars.append(float(np.var(V[idxs])))
    eng["signal_uniformity"] = round(float(np.mean(branch_vars)) if branch_vars else 0.0, 6)

    max_branch_delta = 0.0
    for g in BRANCH_GROUPS:
        idxs = [n - 1 for n in g if 1 <= n <= 11]
        if len(idxs) > 1:
            bv = V[idxs]
            max_branch_delta = max(max_branch_delta, float(bv.max() - bv.min()))
    eng["max_branch_delta"] = round(max_branch_delta, 6)

    pairs = [(1, 2), (4, 5), (7, 8)]
    cross_vals = [abs(devs[a]-devs[b]) for a, b in pairs if a < len(devs) and b < len(devs)]
    cross = float(np.mean(cross_vals)) if cross_vals else 0.0
    eng["cross_coupling_estimate"] = round(cross, 4)

    return eng, float(routing_sig), branch_sym, iso_idx, cross

# ── FIX-E: TEMPORAL SEQUENCE ──────────────────────────────────────────────────
def generate_temporal_steps(segs, sl, sv_v, forced, cl, ns, n_steps=5):
    """Improved temporal dynamics: thermal drift, vibration, intermittent transients, EMI spikes."""
    seq = []
    thermal_drift = np.linspace(
        np.random.uniform(-0.03, 0.00),
        np.random.uniform(0.00, 0.03),
        n_steps
    )
    for t in range(n_steps):
        vibration = np.random.normal(0, 0.01)
        intermittent = 0.0
        if np.random.rand() < 0.15:
            intermittent = -np.random.uniform(0.05, 0.30)
        perturbed = [
            (i, j, r * (1.0 + vibration + thermal_drift[t]))
            for i, j, r in segs
        ]
        V_t = solve(perturbed, sl, sv_v, forced, cl)
        Vn_t = []
        for n in range(1, NUM_NODES):
            val = V_t[n] + intermittent
            if np.random.rand() < 0.02:
                val += np.random.uniform(-0.10, 0.10)
            val += np.random.normal(0, ns)
            adc_step = 0.005
            val = round(val / adc_step) * adc_step
            Vn_t.append(max(0.0, val))
        seq.append(Vn_t)
    return seq

# ── SAMPLE GENERATOR ──────────────────────────────────────────────────────────
def generate_sample(cfg, fault_id, idx, multi_fault=False, add_temporal=True):
    wv   = np.random.uniform(0.90, 1.10)
    sv_v = cfg["source_v"] * np.random.uniform(0.97, 1.03)
    temp = np.random.uniform(20.0, 90.0)
    sl   = cfg["sensor_load"]
    ns   = cfg["noise_std"]

    segs, sl_eff, sv, forced, cl, loc = build_segs_for_fault(cfg, fault_id, wv)
    V = solve(segs, sl_eff, sv_v, forced, cl)

    if fault_id == 5:
        pair_str = loc.replace("Route_Swap_N", "").split("_")
        try:
            pair = (int(pair_str[0]), int(pair_str[1]))
        except Exception:
            pair = (2, 3)
        V = apply_routing_post_physics(V, pair)

    connector_fault = "None"
    if np.random.rand() < 0.08:
        cn = random.choice(SENSOR_NODES)
        V, connector_fault = apply_connector_fault(V, cn)
        if fault_id == 0:
            loc = f"Connector_N{cn}"

    intermittent_triggered = False
    if fault_id != 0 and np.random.rand() < 0.12:
        in_node = random.choice(SENSOR_NODES)
        V, intermittent_triggered = apply_intermittent(V, in_node, 0.40)

    if multi_fault and fault_id != 0 and np.random.rand() < 0.05:
        sec_seg = random.choice(OPEN_SEGS)
        sec_segs = [(i, j, r) for i, j, r in segs if not (i == sec_seg[0] and j == sec_seg[1])]
        sec_segs.append((sec_seg[0], sec_seg[1], 1e8))
        V = solve(sec_segs, sl_eff, sv_v, forced, cl)
        loc += f"+Open_N{sec_seg[0]}_N{sec_seg[1]}"

    Vn = V.copy()
    for n in range(1, NUM_NODES):
        val = Vn[n]
        val += np.random.normal(0, ns)
        if np.random.rand() < 0.015:
            val += np.random.uniform(-0.08, 0.08)
        adc_step = 0.005
        val = round(val / adc_step) * adc_step
        val += np.random.normal(0, 0.002)
        Vn[n] = max(0.0, val)

    href = healthy_ref(cfg)
    devs = []
    for n in range(1, NUM_NODES):
        r = href[n - 1]
        devs.append((Vn[n] - r) / r * 100.0 if abs(r) > 1e-6 else
                    (-100.0 if Vn[n] < 1e-6 else 0.0))

    Vn_nodes = [Vn[n] for n in range(1, NUM_NODES)]
    eng, routing_sig, branch_sym, iso_idx, cross = compute_engineered(Vn_nodes, devs, sl, href)

    # FIX-C: physics-informed severity (analysis only)
    sev = compute_physics_severity(devs, branch_sym, iso_idx, routing_sig)

    row = {
        "sample_id":          idx,
        "fault_label":        FAULT_LABELS[fault_id],
        "fault_id":           fault_id,
        "fault_location":     loc,
        "severity":           sev,
        "fault_detected":     0 if fault_id == 0 else 1,
        "connector_fault":    connector_fault,
        "intermittent":       int(intermittent_triggered),
        "temperature_C":      round(temp, 2),
    }
    for n in range(1, NUM_NODES):
        row[f"V_N{n}"] = round(Vn[n], 6)
    for n in range(1, NUM_NODES):
        row[f"DEV_N{n}_pct"] = round(devs[n - 1], 4)
    for n in SENSOR_NODES:
        row[f"I_N{n}_mA"] = round(Vn[n] / sl * 1000.0, 4)
    row.update(eng)
    row["temperature_C"] = round(temp, 2)

    # FIX-E: 5-step temporal sequence
    if add_temporal:
        temporal = generate_temporal_steps(segs, sl_eff, sv_v, forced, cl, ns)
        for t_idx, Vt in enumerate(temporal):
            for n_idx, v in enumerate(Vt):
                row[f"V_N{n_idx+1}_t{t_idx}"] = round(v, 6)

    return row

# ── GENERATE + SAVE ───────────────────────────────────────────────────────────
def generate_dataset(mode_name, n=1000, add_temporal=True):
    cfg   = MODES[mode_name]
    n_cls = len(FAULT_LABELS)
    print(f"\n[{mode_name.upper()}] {n} x {n_cls} = {n * n_cls} samples")
    rows = []; idx = 0
    for fid in range(n_cls):
        print(f"  {FAULT_LABELS[fid]:<22}...", end="", flush=True)
        for i in range(n):
            rows.append(generate_sample(cfg, fid, idx, multi_fault=True,
                                        add_temporal=add_temporal))
            idx += 1
        print(" done")
    random.shuffle(rows)
    return rows

def save_csv(rows, fname):
    if not rows:
        return
    path = os.path.join(OUT, fname)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"  Saved: {path}  ({len(rows)} rows, {len(rows[0])} cols)")

def plot_distribution(rows_r, rows_d):
    if not PLOT: return
    fig, axes = plt.subplots(1, 2, figsize=(16, 5), facecolor="white")
    cols = ["#27AE60","#C0392B","#B86B00","#5B3FA6","#2471A3","#E74C3C"]
    lbls = list(FAULT_LABELS.values())
    for ax, rows, title in zip(axes, [rows_r, rows_d], ["Realistic", "Demo"]):
        cnts = [sum(1 for r in rows if r["fault_label"] == l) for l in lbls]
        bars = ax.bar(lbls, cnts, color=cols, alpha=0.85, edgecolor="white", lw=1.2)
        ax.set_title(f"Class Distribution — {title}", fontsize=12, fontweight="bold")
        ax.set_ylabel("Count"); ax.set_ylim(0, max(cnts) * 1.2)
        for b, c in zip(bars, cnts):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 5,
                    str(c), ha="center", fontsize=9, fontweight="bold")
        ax.tick_params(axis="x", rotation=20); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUT, "class_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Plot: {path}")

def plot_severity_distribution(rows, mode_name):
    if not PLOT: return
    lbls = list(FAULT_LABELS.values())
    cols = ["#27AE60","#C0392B","#B86B00","#5B3FA6","#2471A3","#E74C3C"]
    fig, ax = plt.subplots(figsize=(12, 6), facecolor="white")
    data = [[float(r["severity"]) for r in rows if r["fault_label"] == l] for l in lbls]
    bp = ax.boxplot(data, patch_artist=True, medianprops=dict(color="black", lw=2))
    for patch, c in zip(bp["boxes"], cols):
        patch.set_facecolor(c); patch.set_alpha(0.7)
    ax.set_xticklabels([l.replace("_", "\n") for l in lbls], fontsize=8)
    ax.set_ylabel("Physics-Informed Severity [0-100]")
    ax.set_title(
        f"Semi-Physics Severity Distribution — {mode_name.title()} v6\n"
        "(0.45*voltage_drop + 0.25*branch_instability + 0.15*isolation_error + 0.15*routing_anomaly)",
        fontsize=10, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUT, f"severity_distribution_{mode_name}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Plot: {path}")

def plot_routing_separation(rows, mode_name):
    if not PLOT: return
    lbls = list(FAULT_LABELS.values())
    cols = ["#27AE60","#C0392B","#B86B00","#5B3FA6","#2471A3","#E74C3C"]
    fig, ax = plt.subplots(figsize=(10, 7), facecolor="white")
    for lbl, col in zip(lbls, cols):
        sub = [r for r in rows if r["fault_label"] == lbl]
        x = [float(r["routing_signature"]) for r in sub]
        y = [float(r["branch_symmetry_error"]) for r in sub]
        ax.scatter(x, y, c=col, label=lbl, alpha=0.3, s=12, edgecolors="none")
    ax.set_xlabel("Routing Signature"); ax.set_ylabel("Branch Symmetry Error")
    ax.set_title(f"Routing Signature vs Branch Symmetry — {mode_name.title()} v6",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, markerscale=3); ax.grid(alpha=0.2)
    plt.tight_layout()
    path = os.path.join(OUT, f"routing_separation_{mode_name}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Plot: {path}")

def plot_boxplots(rows, mode_name):
    if not PLOT: return
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), facecolor="white")
    axes = axes.flatten()
    cols = ["#27AE60","#C0392B","#B86B00","#5B3FA6","#2471A3","#E74C3C"]
    lbls = list(FAULT_LABELS.values())
    for ax_i, node in enumerate([2, 3, 5, 6, 8, 9]):
        ax = axes[ax_i]
        data = [[float(r[f"V_N{node}"]) for r in rows if r["fault_label"] == l] for l in lbls]
        bp = ax.boxplot(data, patch_artist=True, medianprops=dict(color="black", lw=2))
        for patch, c in zip(bp["boxes"], cols):
            patch.set_facecolor(c); patch.set_alpha(0.7)
        ax.set_title(f"N{node} Voltage per Class", fontsize=11, fontweight="bold")
        ax.set_xticklabels([l.replace("_", "\n") for l in lbls], fontsize=7)
        ax.set_ylabel("Voltage (V)"); ax.grid(axis="y", alpha=0.3)
    fig.suptitle(f"Voltage Distributions — {mode_name.title()}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(OUT, f"voltage_boxplot_{mode_name}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Plot: {path}")

def plot_correlation(rows, mode_name):
    if not PLOT: return
    cols = (
        [f"V_N{n}" for n in range(1, 12)] +
        ["V_min","V_max","V_range","V_mean","V_std",
         "N5_N6_diff","N2_N3_diff","N8_N9_diff","worst_node_isolation",
         "routing_signature","branch_symmetry_error","isolation_index",
         "signal_uniformity","max_branch_delta","cross_coupling_estimate"]
    )
    available = [c for c in cols if c in rows[0]]
    try:
        data = np.array([[float(r.get(c, 0)) for c in available] for r in rows])
    except Exception:
        return
    corr = np.corrcoef(data.T)
    fig, ax = plt.subplots(figsize=(16, 14), facecolor="white")
    im = ax.imshow(corr, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, label="Pearson Correlation")
    ax.set_xticks(range(len(available))); ax.set_yticks(range(len(available)))
    ax.set_xticklabels(available, rotation=90, fontsize=7)
    ax.set_yticklabels(available, fontsize=7)
    ax.set_title(f"Feature Correlation — {mode_name.title()} v6", fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(OUT, f"feature_correlation_{mode_name}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Plot: {path}")

def print_summary(rows, mode_name):
    print(f"\n{'='*72}")
    print(f"SUMMARY — {mode_name.upper()}  ({len(rows)} samples, {len(rows[0])} features)")
    print(f"v6: worst_node_idx removed | signal_uniformity=mean(branch-var) | physics severity")
    print(f"{'─'*72}")
    print(f"{'Class':<22} {'N':>5} {'V_N2':>8} {'V_N8':>8} {'RoutSig':>9} {'Sev%':>7}")
    print("-"*56)
    for fid, lbl in FAULT_LABELS.items():
        sub = [r for r in rows if r["fault_label"] == lbl]
        if not sub: continue
        n2  = np.mean([float(r["V_N2"]) for r in sub])
        n8  = np.mean([float(r["V_N8"]) for r in sub])
        rs  = np.mean([float(r.get("routing_signature", 0)) for r in sub])
        sv  = np.mean([float(r.get("severity", 0)) for r in sub])
        print(f"{lbl:<22} {len(sub):>5} {n2:>8.4f} {n8:>8.4f} {rs:>9.4f} {sv:>6.1f}")
    print(f"{'='*72}")

# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("HARNESS DATASET GENERATOR v6")
    print("AI-Driven Pre-ECU Harness Integrity & Routing Validation")
    print("Ashok Leyland — Chassis Assembly Line 2")
    print("=" * 60)
    print("v6: FIX-A worst_node_idx removed | FIX-B signal_uniformity redefined")
    print("    FIX-C physics severity | FIX-D node count | FIX-E temporal cols")

    rows_r = generate_dataset("realistic", 1000)
    rows_d = generate_dataset("demo",      1000)
    for r in rows_r: r["mode"] = "realistic"
    for r in rows_d: r["mode"] = "demo"
    rows_c = rows_r + rows_d; random.shuffle(rows_c)

    print("\nSaving CSVs...")
    save_csv(rows_r, "harness_dataset_realistic.csv")
    save_csv(rows_d, "harness_dataset_demo.csv")
    save_csv(rows_c, "harness_dataset_combined.csv")

    print("\nGenerating plots...")
    plot_distribution(rows_r, rows_d)
    plot_routing_separation(rows_r, "realistic")
    plot_severity_distribution(rows_r, "realistic")
    plot_boxplots(rows_r, "realistic")
    plot_boxplots(rows_d, "demo")
    plot_correlation(rows_r, "realistic")

    print_summary(rows_r, "realistic")
    print_summary(rows_d, "demo")

    print(f"\nOutputs -> {OUT}/")
    print("Next: python harness_classifier.py")

if __name__ == "__main__":
    main()
