"""Network traffic classifier - photonic reservoir vs. a classical baseline.

Labels: 0 = benign, 1 = loud attack, 2 = stealth attack.

Pipeline:
    16 features -> log + robust scale + clip -> values in [0, 1], then:
    Quantum:  16 features + several frozen photonic circuits (16 modes) -> probabilities
              -> linear readout -> 3 classes
    Baseline: the same linear readout on the 16 features, no circuit
    Several reservoir designs are compared by 5-fold cross-validation on train; the best
    one is kept. Weight decay and per-class decision offsets are also chosen by CV, so
    validation stays untouched. The final quantum readout is refit on train + validation
    and writes the hidden-test predictions. A Qiskit version of the encoder
    (RY angles + CNOT chain) is built but not run.
"""
import os
import time

import matplotlib.pyplot as plt
import merlin as ML
import numpy as np
import pandas as pd
import torch
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler

SEED = 42
N_CLASSES = 3            # 0 = benign, 1 = loud attack, 2 = stealth attack
RESERVOIR_BATCH_SIZE = 512
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

plt.rcParams.update({"figure.dpi": 105, "axes.spines.top": False, "axes.spines.right": False})
np.set_printoptions(precision=3, suppress=True)


def section(title):
    print(f"\n=== {title} ===")


def class_counts(labels):
    """Return formatted counts per class, e.g., 'class 0: 4803, class 1: 1827, class 2: 1370'."""
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=N_CLASSES)
    return ", ".join(f"class {c}: {n}" for c, n in enumerate(counts))


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
train = pd.read_csv("data/train.csv")
validation = pd.read_csv("data/validation.csv")
hid_a = pd.read_csv("data/test_hidden_a.csv")
hid_b = pd.read_csv("data/test_hidden_b.csv")

section("Data")
print(f"Samples: train {len(train)} | validation {len(validation)} | "
      f"hidden_a {len(hid_a)} | hidden_b {len(hid_b)} | device {DEVICE}")
print(f"Train labels:      {class_counts(train['label'])}")
print(f"Validation labels: {class_counts(validation['label'])}")

FEATURES = [c for c in train.columns if c not in ("sample_id", "label")]

LOG_COLS = [
    "duration_s", "packet_count", "mean_packet_bytes", "std_packet_bytes",
    "mean_interarrival_ms", "out_in_byte_ratio", "connections_per_min",
    "unique_destinations", "port_diversity",
]


def select_feature(df):
    X = df[FEATURES].copy()
    X[LOG_COLS] = np.log1p(X[LOG_COLS])
    return X.to_numpy(dtype=np.float64)


scaler = RobustScaler().fit(select_feature(train))


def prepare(df):
    X = scaler.transform(select_feature(df))
    X = np.clip(X, -3.0, 3.0)
    return (X + 3.0) / 6.0


X_train, y_train = prepare(train), train["label"].to_numpy(dtype=np.int64)
X_val, y_val = prepare(validation), validation["label"].to_numpy(dtype=np.int64)

USE_CLASS_WEIGHTS = True
CLASS_WEIGHTS_NP = len(y_train) / (N_CLASSES * np.bincount(y_train, minlength=N_CLASSES))
CLASS_WEIGHTS = (
    torch.as_tensor(CLASS_WEIGHTS_NP, dtype=torch.float64, device=DEVICE)
    if USE_CLASS_WEIGHTS
    else None
)


# ----------------------------------------------------------------------------
# Qiskit encoder: one qubit per feature, RY angle encoding + a chain of CNOTs
# ----------------------------------------------------------------------------
def qml_encoder(data_matrix, entanglement=True):
    _, num_features = data_matrix.shape
    angles_matrix = data_matrix * np.pi
    x_params = ParameterVector("x", num_features)
    qc = QuantumCircuit(num_features)
    for i in range(num_features):
        qc.ry(x_params[i], i)
    if entanglement:
        for i in range(num_features - 1):
            qc.cx(i, i + 1)
    return qc, angles_matrix


feature_map, X_train_angles = qml_encoder(X_train, entanglement=True)
n_qubits = feature_map.num_qubits

section("Qiskit encoder")
print(f"Feature map: {n_qubits} qubits, one per feature")
print(f"Training angles: {X_train_angles.shape[0]} samples x {X_train_angles.shape[1]} features "
      f"(RY angles in [0, π])")

observables_list = [
    SparsePauliOp("I" * (n_qubits - 1 - i) + "Z" + "I" * i)
    for i in range(n_qubits)
]
print(f"Observables: {len(observables_list)}, Pauli Z on each qubit")


# ----------------------------------------------------------------------------
# Frozen photonic reservoirs
# ----------------------------------------------------------------------------
N_MODES = 16
N_ENCODINGS = 1

RESERVOIR_CONFIGS = {
    "4 x 2 photons @ 0.5π (current)": [(2, 0.5)] * 4,
    "8 x 2 photons @ 0.5π":           [(2, 0.5)] * 8,
    "6 x 2 photons, mixed scales":    [(2, 0.25), (2, 0.5), (2, 1.0)] * 2,
    "4 x 3 photons @ 0.5π":           [(3, 0.5)] * 4,
}


def build_circuit(seed, n_photons, encoding_scale):
    """Build one frozen photonic reservoir circuit with fixed mixing layers."""
    torch.manual_seed(seed)
    builder = ML.CircuitBuilder(n_modes=N_MODES)
    builder.add_entangling_layer(trainable=False, name="pre")
    modes = list(range(N_MODES))
    for k in range(1, N_ENCODINGS + 1):
        builder.add_angle_encoding(modes=modes, name="qrc_input", scale=encoding_scale)
        builder.add_entangling_layer(trainable=False, name=f"post_{k}")
    builder.add_entangling_layer(trainable=False, name="post_final")

    layer = ML.QuantumLayer(
        input_size=N_ENCODINGS * N_MODES,
        builder=builder,
        n_photons=n_photons,
        measurement_strategy=ML.MeasurementStrategy.probs(
            computation_space=ML.ComputationSpace.UNBUNCHED
        ),
        dtype=torch.float64,
    )
    for parameter in layer.parameters():
        parameter.requires_grad_(False)
    return layer.to(DEVICE).eval()


def build_reservoirs(spec):
    return [
        build_circuit(SEED + r, photons, scale * np.pi)
        for r, (photons, scale) in enumerate(spec)
    ]


def circuit_input(X):
    tiled = np.tile(X, (1, N_ENCODINGS)) if N_ENCODINGS > 1 else X
    return torch.as_tensor(tiled, dtype=torch.float64, device=DEVICE)


def run_circuit_batched(layer, X, batch_size=RESERVOIR_BATCH_SIZE):
    """Evaluate a frozen quantum layer in batches to bound memory usage."""
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(X), batch_size):
            batch_in = circuit_input(X[start:start + batch_size])
            outputs.append(layer(batch_in).cpu().numpy())
    return np.vstack(outputs)


def reservoir_features(X, circuits):
    """Concatenate scaled input features with probabilities from every reservoir."""
    parts = [run_circuit_batched(layer, X) for layer in circuits]
    return np.hstack([X] + parts)


# ----------------------------------------------------------------------------
# Readout: linear model, weight decay and class offsets chosen by cross-validation
# ----------------------------------------------------------------------------
N_FOLDS = 5
WD_GRID = (1e-3, 1e-2)
OFFSET_GRID = np.linspace(-1.5, 1.5, 31)

# Precompute (1, 1, 3) offset grid once for vectorized tuning
_g1, _g2 = np.meshgrid(OFFSET_GRID, OFFSET_GRID, indexing="ij")
_OFFSET_CANDIDATES = np.stack(
    [np.zeros_like(_g1), _g1, _g2], axis=-1
).reshape(-1, N_CLASSES)  # shape (G, 3)


def make_standardizer(F):
    """Return a transform that standardizes columns of F and drops near-constant ones."""
    mean = F.mean(axis=0)
    std = F.std(axis=0)
    keep = std > 1e-6 * max(float(std.max()), 1e-12)
    mean_k, std_k = mean[keep], std[keep]

    def transform(G):
        return (G[:, keep] - mean_k) / std_k

    return transform


def train_readout(features, labels, epochs=500, lr=None, weight_decay=1e-4, seed=SEED):
    X = torch.as_tensor(features, dtype=torch.float64, device=DEVICE)
    y = torch.as_tensor(labels, dtype=torch.long, device=DEVICE)
    if lr is None:
        lr = 0.05 * min(1.0, 136 / X.shape[1])

    torch.manual_seed(seed)
    model = torch.nn.Linear(X.shape[1], N_CLASSES, dtype=torch.float64, device=DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    history = []

    for _ in range(epochs):
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(X), y, weight=CLASS_WEIGHTS)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        history.append(float(loss.item()))

    return model.eval(), history


def predict_proba(model, features):
    with torch.inference_mode():
        logits = model(torch.as_tensor(features, dtype=torch.float64, device=DEVICE))
        return torch.softmax(logits, dim=1).cpu().numpy()


def predict_with_offsets(proba, offsets):
    return (np.log(proba + 1e-12) + offsets).argmax(axis=1)


def balanced_accuracy(y, pred):
    return float(np.mean([(pred[y == c] == c).mean() for c in range(N_CLASSES)]))


def tune_offsets_fast(proba, y):
    """Vectorized search over all (o1, o2) pairs maximizing balanced accuracy."""
    log_p = np.log(proba + 1e-12)  # (N, 3)
    # scores: (G, N, 3) -> preds: (G, N)
    scores = log_p[None, :, :] + _OFFSET_CANDIDATES[:, None, :]
    preds = scores.argmax(axis=2)

    class_masks = [(y == c) for c in range(N_CLASSES)]
    class_counts_arr = np.array([m.sum() for m in class_masks], dtype=np.float64)

    # mean recall across classes for every offset candidate: shape (G,)
    recalls = np.column_stack([
        (preds[:, mask] == c).mean(axis=1)
        for c, mask in enumerate(class_masks)
        if class_counts_arr[c] > 0
    ])
    bal_accs = recalls.mean(axis=1)

    best_idx = int(np.argmax(bal_accs))
    default_score = balanced_accuracy(y, proba.argmax(axis=1))
    if bal_accs[best_idx] > default_score:
        return float(bal_accs[best_idx]), _OFFSET_CANDIDATES[best_idx].copy()
    return default_score, np.zeros(N_CLASSES, dtype=np.float64)


def out_of_fold_proba(F, y, wd):
    proba = np.zeros((len(y), N_CLASSES), dtype=np.float64)
    folds = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
    for fit_idx, held_idx in folds.split(F, y):
        transform = make_standardizer(F[fit_idx])
        model, _ = train_readout(transform(F[fit_idx]), y[fit_idx], weight_decay=wd)
        proba[held_idx] = predict_proba(model, transform(F[held_idx]))
    return proba


def cv_select(F_tr, name):
    best = None
    for wd in WD_GRID:
        oof = out_of_fold_proba(F_tr, y_train, wd)
        score, offsets = tune_offsets_fast(oof, y_train)
        print(f"    {name}: weight decay {wd:g} -> CV balanced accuracy {score:.3f}")
        if best is None or score > best["score"]:
            best = dict(score=score, wd=wd, offsets=offsets)
    return best


def fit_and_validate(F_tr, F_va, choice):
    transform = make_standardizer(F_tr)
    model, history = train_readout(transform(F_tr), y_train, weight_decay=choice["wd"])
    proba = predict_proba(model, transform(F_va))
    return dict(
        choice,
        model=model,
        history=history,
        proba=proba,
        pred=predict_with_offsets(proba, choice["offsets"]),
    )


# ----------------------------------------------------------------------------
# Reservoir search: keep the design with the best CV score on train
# ----------------------------------------------------------------------------
section(f"Reservoir search ({N_FOLDS}-fold cross-validation on train)")
best_name, best_result = None, None
for name, spec in RESERVOIR_CONFIGS.items():
    start = time.time()
    circuits_try = build_reservoirs(spec)
    F_try = reservoir_features(X_train, circuits_try)
    print(f"  {name}: {F_try.shape[1]} features, circuits took {time.time() - start:.1f}s")
    choice = cv_select(F_try, name)
    if best_result is None or choice["score"] > best_result["score"]:
        best_name = name
        best_result = dict(choice, circuits=circuits_try, F_train=F_try)

print(f"  Chosen: {best_name} (CV balanced accuracy {best_result['score']:.3f})")
circuits = best_result["circuits"]
F_train = best_result["F_train"]
F_val = reservoir_features(X_val, circuits)

section("Baseline (16 features, no circuit)")
base_choice = cv_select(X_train, "baseline")

q = fit_and_validate(F_train, F_val, best_result)
b = fit_and_validate(X_train, X_val, base_choice)

readout, history, val_proba, val_pred = q["model"], q["history"], q["proba"], q["pred"]
base_pred = b["pred"]

F_all = np.vstack([F_train, F_val])
y_all = np.concatenate([y_train, y_val])
transform_all = make_standardizer(F_all)
final_readout, _ = train_readout(transform_all(F_all), y_all, weight_decay=q["wd"])


# ----------------------------------------------------------------------------
# What the chosen circuit does: sweep one input from 0 to 1, hold the other 15 at 0
# ----------------------------------------------------------------------------
circuit = circuits[0]
circuit_dim = circuit.output_size
sweep = np.linspace(0.0, 1.0, 80)
all_probabilities = []
for i in range(N_MODES):
    batch = np.zeros((len(sweep), N_MODES), dtype=np.float64)
    batch[:, i] = sweep
    all_probabilities.append(run_circuit_batched(circuit, batch))

vmax = max(float(p.max()) for p in all_probabilities)
fig, axes = plt.subplots(
    4, 4, figsize=(14, 12), sharex=True, sharey=True, constrained_layout=True
)
img = None
for i, (ax, probs) in enumerate(zip(axes.flat, all_probabilities)):
    img = ax.imshow(
        probs.T,
        aspect="auto",
        origin="lower",
        extent=[0, 1, -0.5, circuit_dim - 0.5],
        cmap="viridis",
        vmin=0,
        vmax=vmax,
    )
    ax.set_title(f"Sweep mode {i}", fontsize=9)
for ax in axes[-1, :]:
    ax.set_xlabel("Input value")
for ax in axes[:, 0]:
    ax.set_ylabel("Output feature")
if img is not None:
    fig.colorbar(img, ax=axes, label="Measurement probability", shrink=0.8)
fig.suptitle("The first chosen photonic circuit transforms its inputs (other modes = 0)")
plt.show()


# ----------------------------------------------------------------------------
# Metrics: confusion matrix, per-class F1 and a printed report
# ----------------------------------------------------------------------------
def confusion_matrix(y, yh):
    idx = np.asarray(y, dtype=np.int64) * N_CLASSES + np.asarray(yh, dtype=np.int64)
    return np.bincount(idx, minlength=N_CLASSES * N_CLASSES).reshape(N_CLASSES, N_CLASSES)


def f1_per_class(cmat):
    tp = np.diag(cmat)
    denom = cmat.sum(axis=1) + cmat.sum(axis=0)
    return np.divide(2 * tp, denom, out=np.zeros_like(tp, dtype=np.float64), where=denom > 0)


def report(name, y_true, pred, show_confusion=False):
    cmat = confusion_matrix(y_true, pred)
    row_sums = cmat.sum(axis=1)
    recall = np.divide(
        np.diag(cmat), row_sums, out=np.zeros(N_CLASSES, dtype=np.float64), where=row_sums > 0
    )
    f1 = f1_per_class(cmat)
    print(f"\n{name}")
    print(f"  Accuracy {np.trace(cmat) / cmat.sum():.3f} | Balanced accuracy {recall.mean():.3f} | "
          f"Macro F1 {f1.mean():.3f}")
    print(f"  {'':<10}" + "".join(f"{f'class {c}':>9}" for c in range(N_CLASSES)))
    print(f"  {'Recall':<10}" + "".join(f"{r:>9.0%}" for r in recall))
    print(f"  {'F1':<10}" + "".join(f"{s:>9.3f}" for s in f1))
    if show_confusion:
        print("  Confusion matrix (rows = true class, columns = predicted class):")
        print(f"  {'':<10}" + "".join(f"{f'pred {c}':>9}" for c in range(N_CLASSES)))
        for c, row in enumerate(cmat):
            print(f"  {f'true {c}':<10}" + "".join(f"{n:>9}" for n in row))


# ----------------------------------------------------------------------------
# Validation results: baseline vs. quantum model
# ----------------------------------------------------------------------------
section("Validation results")
print(f"Settings: class weights {'on' if USE_CLASS_WEIGHTS else 'off'}, reservoirs '{best_name}', "
      f"weight decay quantum {q['wd']:g} / baseline {b['wd']:g}")
print(f"Reference: always predicting the most common class gives accuracy "
      f"{np.bincount(y_val).max() / len(y_val):.3f}")
print("Balanced accuracy = mean recall over the classes. Macro F1 = mean F1 over the classes.")
print("All tuning used training data only, so these validation scores are unbiased.")
report("Baseline: linear readout on the 16 scaled features", y_val, base_pred)
report(
    f"Quantum: 16 features + reservoirs '{best_name}' ({F_train.shape[1]} features) "
    f"-> linear readout",
    y_val,
    val_pred,
    show_confusion=True,
)

print("\nFirst 5 validation samples, quantum model:")
print(f"  {'sample_id':<20}{'true':>5}{'predicted':>11}" + "".join(f"{f'P({c})':>8}" for c in range(N_CLASSES)))
for sid, true, pred, probs in zip(
    validation["sample_id"][:5], y_val[:5], val_pred[:5], val_proba[:5]
):
    print(f"  {sid:<20}{true:>5}{pred:>11}" + "".join(f"{p:>8.3f}" for p in probs))

plt.figure(figsize=(6, 3))
plt.plot(history)
plt.xlabel("Training step")
plt.ylabel("Loss")
plt.title("Readout training")
plt.show()


# ----------------------------------------------------------------------------
# Predict both hidden sets with the final quantum readout and write the files
# ----------------------------------------------------------------------------
section("Hidden test predictions (quantum model, refit on train + validation)")
os.makedirs("predictions", exist_ok=True)
for name, df in [("hidden_a", hid_a), ("hidden_b", hid_b)]:
    feats = reservoir_features(prepare(df), circuits)
    proba = predict_proba(final_readout, transform_all(feats))
    pred = predict_with_offsets(proba, q["offsets"])
    pd.DataFrame({"sample_id": df["sample_id"], "label": pred}).to_csv(
        f"predictions/{name}.csv", index=False
    )
    print(f"Saved predictions/{name}.csv ({len(pred)} rows). Predicted {class_counts(pred)}")