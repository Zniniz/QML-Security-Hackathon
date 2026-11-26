"""Network traffic classifier - photonic reservoir vs. classical baselines.

Labels: 0 = benign, 1 = loud attack, 2 = stealth attack.

Pipeline:
    16 features -> log + robust scale + clip -> values in [0, 1], then:
    Quantum:  frozen photonic circuit (16 modes, 2 photons) -> 120 probabilities
              -> Linear(120 -> 3)
    Baseline: the same Linear(16 -> 3) with no circuit
    NumPy:    softmax regression from scratch, trained with momentum on minibatches
    The quantum model writes the hidden-test predictions. A Qiskit version of the
    encoder (RY angles + CNOT chain) is built but not run.
"""
import matplotlib.pyplot as plt
import merlin as ML
import numpy as np
import pandas as pd
import torch
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from sklearn.preprocessing import RobustScaler

SEED = 42
N_CLASSES = 3            # 0 = benign, 1 = loud attack, 2 = stealth attack
np.random.seed(SEED)
torch.manual_seed(SEED)

# Plot and print style: sharper figures with no top/right frame, arrays to 3 decimals
plt.rcParams.update({"figure.dpi": 105, "axes.spines.top": False, "axes.spines.right": False})
np.set_printoptions(precision=3, suppress=True)


# Console output helpers
def section(title):
    print(f"\n=== {title} ===")


def class_counts(labels):
    '''"class 0: 4803, class 1: 1827, class 2: 1370" for an array of labels'''
    counts = np.bincount(np.asarray(labels, dtype=int), minlength=N_CLASSES)
    return ", ".join(f"class {c}: {n}" for c, n in enumerate(counts))


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
train = pd.read_csv('data/train.csv')
validation = pd.read_csv('data/validation.csv')
hid_a = pd.read_csv('data/test_hidden_a.csv')
hid_b = pd.read_csv('data/test_hidden_b.csv')

section("Data")
print(f"Samples: train {len(train)} | validation {len(validation)} | "
      f"hidden_a {len(hid_a)} | hidden_b {len(hid_b)}")
print(f"Train labels:      {class_counts(train['label'])}")
print(f"Validation labels: {class_counts(validation['label'])}")

FEATURES = [c for c in train.columns if c not in ('sample_id', 'label')]

# Large, skewed columns: log1p squashes big values (log1p(x) = log(1 + x), safe at 0)
LOG_COLS = ['duration_s', 'packet_count', 'mean_packet_bytes', 'std_packet_bytes',
            'mean_interarrival_ms', 'out_in_byte_ratio', 'connections_per_min',
            'unique_destinations', 'port_diversity']

# Clean the data for training and validation
def select_feature(df):
    X = df[FEATURES].copy()
    for c in LOG_COLS:
        X[c] = np.log1p(X[c])
    return X.values

# Fit the scaler on training data only, then reuse it everywhere
scaler = RobustScaler().fit(select_feature(train))
def prepare(df):
    X = select_feature(df)
    X = scaler.transform(X)
    X = np.clip(X, -3, 3)  # cap outliers so (X + 3) / 6 stays within [0, 1]
    return (X + 3) / 6     # in [0, 1]; each encoder below turns this into an angle

X_train, y_train = prepare(train), train['label'].values
X_val, y_val = prepare(validation), validation['label'].values

# Class weights: class 0 is ~60% of the data, so an unweighted model learns to
# dump class 2 into class 0. Weighting each class by N / (C * count) makes every
# class count equally in the loss. Trade-off (0.5*pi, mean of 6 circuits):
#   unweighted: accuracy 0.796, balanced accuracy 0.654, class 2 recall 13%
#   weighted:   accuracy 0.691, balanced accuracy 0.698, class 2 recall 54%
# Set to False if plain accuracy is the metric that matters. Applies to every model.
USE_CLASS_WEIGHTS = True
CLASS_WEIGHTS_NP = len(y_train) / (N_CLASSES * np.bincount(y_train, minlength=N_CLASSES))

# plt.scatter(train.values.T.tolist()[1], train.values.T.tolist()[2])


# ----------------------------------------------------------------------------
# Qiskit encoder: one qubit per feature, RY angle encoding + a chain of CNOTs
# ----------------------------------------------------------------------------
def qml_encoder(data_matrix, entanglement=True):
    num_samples, num_features = data_matrix.shape
    # Scale the [0, 1] features to angles in [0, π] for RY rotations
    angles_matrix = data_matrix * np.pi
    # Template circuit with symbolic parameters (each sample's angles are bound in when run)
    X_params = ParameterVector("x", num_features)
    qc = QuantumCircuit(num_features)
    # Custom Gate Encoding: RY rotations with angles from the data
    for i in range(num_features):
        qc.ry(X_params[i], i)
    if entanglement:
        # CNOT from each qubit to the next entangles neighbouring qubits
        for i in range(num_features - 1):
            qc.cx(i, i + 1)
    return qc, angles_matrix

# Get the circuit and angles for train. feature_map = template circuit, X_train_angles = angles for every session
feature_map, X_train_angles = qml_encoder(X_train, entanglement=True)

n_qubits = feature_map.num_qubits
section("Qiskit encoder")
print(f"Feature map: {n_qubits} qubits, one per feature")
print(f"Training angles: {X_train_angles.shape[0]} samples x {X_train_angles.shape[1]} features "
      f"(RY angles in [0, π])")

from qiskit.quantum_info import SparsePauliOp

# One observable per qubit: Pauli Z on qubit i, identity on the rest.
# Qiskit reads Pauli strings right to left, so qubit i is the i-th character from the end.
# Nothing below measures these yet; an Estimator would give one <Z> value per qubit.
observables_list = []
for i in range(n_qubits):
    pauli_str = "I" * (n_qubits - 1 - i) + "Z" + "I" * i
    observable_single = SparsePauliOp(pauli_str)
    observables_list.append(observable_single)

print(f"Observables: {len(observables_list)}, Pauli Z on each qubit")


# ----------------------------------------------------------------------------
# Frozen photonic reservoir: 16 features in -> 120 probabilities out
# ----------------------------------------------------------------------------
N_MODES, N_PHOTONS = 16, 2   # one optical mode per feature
# Inputs are in [0, 1], so each feature becomes a phase in [0, ENCODING_SCALE].
# Keep this below 2*pi: at 2*pi the phase wraps around and the lowest and highest
# feature values look identical to the circuit. Over 6 random circuits, 0.25-0.5*pi
# did best (0.796 val acc), pi gave 0.790 and 2*pi gave 0.771.
ENCODING_SCALE = 0.5 * np.pi

def build_circuit():
    builder = ML.CircuitBuilder(n_modes=N_MODES)
    # Fixed mixing layer -> encoding (feature i sets the phase on mode i) -> fixed mixing layers
    builder.add_entangling_layer(trainable=False, name="pre")
    builder.add_angle_encoding(modes=list(range(N_MODES)), name="qrc_input", scale=ENCODING_SCALE)
    builder.add_entangling_layer(trainable=False, name="post_1")
    builder.add_entangling_layer(trainable=False, name="post_2")
    # builder.add_entangling_layer(trainable=False, name="post_3")

    layer = ML.QuantumLayer(
            input_size=N_MODES,
            builder=builder,
            n_photons=N_PHOTONS,
            # UNBUNCHED: at most one photon per mode -> C(16, 2) = 120 output probabilities
            measurement_strategy=ML.MeasurementStrategy.probs(
                computation_space=ML.ComputationSpace.UNBUNCHED
            ),
            dtype=torch.float64,
        )

    # Freeze every parameter: the circuit stays fixed, only the readout is trained
    for parameter in layer.parameters():
        parameter.requires_grad_(False)
    return layer.eval()

circuit = build_circuit()
CIRCUIT_DIM = circuit.output_size   # number of output probabilities (120)


# ----------------------------------------------------------------------------
# What the circuit does: sweep one input from 0 to 1, hold the other 15 at 0
# ----------------------------------------------------------------------------
sweep = np.linspace(0.0, 1.0, 80)  # the range prepare() maps real data into
all_probabilities = []
for i in range(N_MODES):
    batch = np.zeros((len(sweep), N_MODES))  # fresh batch so only mode i is swept
    batch[:, i] = sweep
    with torch.no_grad():
        all_probabilities.append(circuit(torch.as_tensor(batch, dtype=torch.float64)).cpu().numpy())

# One panel per mode: x = input value, y = output feature, colour = probability
vmax = max(p.max() for p in all_probabilities)  # shared colour scale across panels
fig, axes = plt.subplots(4, 4, figsize=(14, 12), sharex=True, sharey=True,
                         constrained_layout=True)
for i, (ax, probs) in enumerate(zip(axes.flat, all_probabilities)):
    img = ax.imshow(probs.T, aspect="auto", origin="lower",
                    extent=[0, 1, -0.5, CIRCUIT_DIM - 0.5], cmap="viridis",
                    vmin=0, vmax=vmax)
    ax.set_title(f"Sweep mode {i}", fontsize=9)
for ax in axes[-1, :]:
    ax.set_xlabel("Input value")
for ax in axes[:, 0]:
    ax.set_ylabel("Output feature")
fig.colorbar(img, ax=axes, label="Measurement probability", shrink=0.8)
fig.suptitle("A fixed photonic circuit transforms its inputs (other modes = 0)")
plt.show()


# ----------------------------------------------------------------------------
# Quantum model: frozen circuit features -> trained Linear(120 -> 3) readout
# ----------------------------------------------------------------------------
# Step 1: send every row through the frozen circuit -> 120 probabilities per row.
def reservoir_features(layer, X):
    with torch.no_grad():
        probs = layer(torch.as_tensor(X, dtype=torch.float64))
    return probs.cpu().numpy()

F_train = reservoir_features(circuit, X_train)   # shape (8000, 120)
F_val = reservoir_features(circuit, X_val)       # shape (1000, 120)

# The 120 probabilities are all tiny (about 1/120 each). Rescale each column
# to mean 0 / std 1 so the readout can learn from them easily.
# Fit on training data only, then apply the same scaling to validation.
feat_mean = F_train.mean(axis=0)
feat_std = F_train.std(axis=0) + 1e-12
def standardize(F):
    return (F - feat_mean) / feat_std


# Torch copy of the class weights for the loss; None = every row counts equally
CLASS_WEIGHTS = torch.as_tensor(CLASS_WEIGHTS_NP, dtype=torch.float64) if USE_CLASS_WEIGHTS else None


# Step 2: the trained classical readout, a single Linear layer 120 -> 3.
# Full-batch Adam: every step uses all training rows. Also returns the loss at each step.
def train_readout(features, labels, epochs=500, lr=0.05, weight_decay=1e-4, seed=SEED):
    X = torch.as_tensor(features, dtype=torch.float64)
    y = torch.as_tensor(labels, dtype=torch.long)
    torch.manual_seed(seed)
    model = torch.nn.Linear(X.shape[1], N_CLASSES, dtype=torch.float64)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    history = []
    for _ in range(epochs):
        optimizer.zero_grad()
        # softmax is built in here
        loss = torch.nn.functional.cross_entropy(model(X), y, weight=CLASS_WEIGHTS)
        loss.backward()
        optimizer.step()
        history.append(float(loss.item()))
    return model.eval(), history


# Class probabilities for each row: softmax of the readout's 3 raw scores
def predict_proba(model, features):
    with torch.no_grad():
        logits = model(torch.as_tensor(features, dtype=torch.float64))
        return torch.softmax(logits, dim=1).numpy()   # columns: P(0), P(1), P(2)


readout, history = train_readout(standardize(F_train), y_train)

val_proba = predict_proba(readout, standardize(F_val))
val_pred = val_proba.argmax(axis=1)   # predicted class = the most likely one
qrc_acc = (val_pred == y_val).mean()


# Step 3: baseline - the same Linear -> 3 model on the 16 raw (scaled) features,
# with no quantum circuit. The circuit is only worth it if it beats this.
# Standardized with training statistics only, like the circuit features.
base_mean, base_std = X_train.mean(axis=0), X_train.std(axis=0) + 1e-12
baseline, _ = train_readout((X_train - base_mean) / base_std, y_train)
base_pred = predict_proba(baseline, (X_val - base_mean) / base_std).argmax(axis=1)
base_acc = (base_pred == y_val).mean()


# ----------------------------------------------------------------------------
# Metrics: confusion matrix, per-class F1 and a printed report
# ----------------------------------------------------------------------------
# c_matrix[c1, c2] = number of rows with true class c1 and predicted class c2
def confusion_matrix(y, yh):
    n_classes = np.max(y) + 1
    c_matrix = np.zeros((n_classes, n_classes), dtype=int)
    for c1 in range(n_classes):
        for c2 in range(n_classes):
            #(y==c1)*(yh==c2) is 1 when both conditions are true or 0
            c_matrix[c1, c2] = np.sum((y==c1)*(yh==c2))
    return c_matrix


def f1_per_class(cmat):
    '''
    F1 for each class from a confusion matrix (rows = true, columns = predicted).
    F1_i = 2*TP / (2*TP + FP + FN), and 2*TP + FP + FN = row_i sum + column_i sum.
    '''
    tp = np.diag(cmat)
    return 2 * tp / (cmat.sum(axis=1) + cmat.sum(axis=0))


def report(name, y_true, pred, show_confusion=False):
    '''Print overall scores, a per-class recall/F1 table and optionally the confusion matrix.'''
    cmat = confusion_matrix(y_true, pred)
    recall = np.diag(cmat) / cmat.sum(axis=1)   # share of each true class predicted correctly
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
print(f"Settings: class weights {'on' if USE_CLASS_WEIGHTS else 'off'}, "
      f"encoding scale {ENCODING_SCALE / np.pi:.2f}π")
# The accuracy to beat: always guessing the most common class
print(f"Reference: always predicting the most common class gives accuracy "
      f"{np.bincount(y_val).max() / len(y_val):.3f}")
print("Balanced accuracy = mean recall over the classes. Macro F1 = mean F1 over the classes.")
report("Baseline: linear readout on the 16 scaled features", y_val, base_pred)
report(f"Quantum: 16 features -> photonic circuit -> {CIRCUIT_DIM} probabilities -> linear readout",
       y_val, val_pred)

# zip stops at its shortest input, so val_proba[:5] limits this to 5 rows
print("\nFirst 5 validation samples, quantum model:")
print(f"  {'sample_id':<20}{'true':>5}{'predicted':>11}" + "".join(f"{f'P({c})':>8}" for c in range(N_CLASSES)))
for sid, true, pred, probs in zip(validation['sample_id'], y_val, val_pred, val_proba[:5]):
    print(f"  {sid:<20}{true:>5}{pred:>11}" + "".join(f"{p:>8.3f}" for p in probs))

# Readout loss per training step: it should drop and then level off
plt.figure(figsize=(6, 3))
plt.plot(history)
plt.xlabel("Training step")
plt.ylabel("Loss")
plt.title("Readout training")
plt.show()


# ----------------------------------------------------------------------------
# NumPy softmax regression from scratch: momentum optimiser on minibatches
# ----------------------------------------------------------------------------
def softmax(z):
    '''
    calculate the softmax function along the last dimension
    '''
    z0 = np.max(z,-1, keepdims=True)
    z_exp = np.exp(z - z0) #to prevent underflow/overflow
    p = z_exp / np.sum(z_exp, -1, keepdims=True)
    return p

# Mean cross-entropy loss; only used by the commented-out progress print in fit()
def cost(
    x, # N x D
    yhot, # N x C: one hot coding of the labels
    w # D x C
        ):
    N, D = x.shape
    p = softmax(x @ w)
    ce_loss = -np.mean(np.sum(yhot * np.log(p), -1)) # the cost function L discussed above
    return ce_loss

def gradient(
    x, # NxD
    yhot, # NxC
    w, # DxC
    class_weights=None # C, optional: weight per class (None = every row counts equally)
        ):

    # Gradient of the mean cross-entropy: x^T (softmax(x w) - yhot) / N
    if class_weights is None:
        grad = (1/len(x))*x.T@(softmax(x@w)-yhot)
    else:
        # weighted mean over rows, matching torch's cross_entropy(weight=...)
        sw = yhot @ class_weights  # N: the weight of each row's true class
        grad = x.T@(sw[:, None]*(softmax(x@w)-yhot)) / sw.sum()

    return grad  # D x C matrix: the same shape as the weight matrix w

class MomentumOpt:
    '''
    Implements the method of momentum for linear regression and classification.
    Each step takes the gradient function and its arguments, including w_{t} as input,
    updates the exponential running average of gradients and returns w_{t+1}.
    '''
    def __init__(self, grad_fn, learning_rate=.4, beta = .95):
        self.grad_fn = grad_fn # the gradient function: grad_fn(x,y, w)
        self.learning_rate = learning_rate
        self.beta = beta
        self.m = None #the exponential running average of gradients

    def step(
        self,
        x, # N x D
        yhot, # N x C
        w # D+1 is the w at current time step. The first element w_0 is the bias and the rest are the weights
        ):

        # m = beta * m + (1 - beta) * grad, then step the weights against m
        if np.all(self.m==None):
            # first step: no running average yet, so start it from this gradient
            self.m = (1-self.beta)*self.grad_fn(x,yhot,w)
        else:
            self.m = self.beta*self.m+(1-self.beta)*self.grad_fn(x,yhot,w)
        wp = w-self.learning_rate*self.m

        return wp

# Labels -> one-hot rows, e.g. [0, 2] -> [[1, 0, 0], [0, 0, 1]]
def onehot(x, C=None):
    if C is None:
        C = np.max(x) + 1
    N = x.shape[0]
    x_hot = np.zeros((N,C))
    x_hot[np.arange(N), x] = 1 #use the value of x to index the last dimension
    return x_hot

class SoftmaxRegression:
    '''Multiclass logistic regression with a D x C weight matrix, trained on minibatches.'''

    def __init__(self, max_epochs = 1000, eps = 1e-2, minibatch=128):
        self.max_epochs = max_epochs
        self.eps = eps         # stop once an epoch moves the weights less than this
        self.b = minibatch     # rows per minibatch
        self.w = None # weights

    def fit(self, x, y, optimizer):
        yhot = onehot(y)
        N, D, C = x.shape[0], x.shape[1], yhot.shape[1]
        self.w = np.zeros((D,C))   # start from all-zero weights
        for epoch in range(self.max_epochs):
            w_old = self.w
            for i in range(N//self.b): # update the weights using i'th mini-batch

                # the last minibatch also takes the leftover rows, so none are skipped
                if i+1>=N//self.b:
                    x_new = x[i*self.b:]
                    yhot_new = yhot[i*self.b:]
                else:
                    x_new = x[i*self.b:(i+1)*self.b]
                    yhot_new = yhot[i*self.b:(i+1)*self.b]
                self.w = MomentumOpt.step(optimizer,x_new,yhot_new,self.w)

            if np.linalg.norm(self.w - w_old) < self.eps: # check for convergence
                print(f"Training converged after {epoch + 1} epochs (weight change < {self.eps})")
                break
        else:  # for/else: only runs if the epoch loop finished without a break
            print(f"Training stopped at the {self.max_epochs}-epoch limit without converging")
        return self

    def predict(self, x):
        p = softmax(x @ self.w)  #predict probabilities
        return p

def add_bias(x):
    '''prepend a column of ones so w[0] acts as the bias (intercept)'''
    return np.hstack([np.ones((len(x), 1)), x])


section("Numpy softmax regression (momentum, minibatch)")
# Pass the class weights through to the gradient when they're switched on
np_grad = (lambda x, yhot, w: gradient(x, yhot, w, CLASS_WEIGHTS_NP)) if USE_CLASS_WEIGHTS else gradient
fit = SoftmaxRegression().fit(add_bias(X_train), y_train, MomentumOpt(np_grad))  # train once

# Predicted class = the column with the highest probability
yh_test = fit.predict(add_bias(X_val))
new_yh_test = np.empty(len(yh_test))
for i in range(len(yh_test)):
    new_yh_test[i] = np.argmax(yh_test[i])
yh_train = fit.predict(add_bias(X_train))
new_yh_train = np.empty(len(yh_train))
for i in range(len(yh_train)):
    new_yh_train[i] = np.argmax(yh_train[i])

# Training vs. validation: a big gap between them would point to overfitting
report("Training set", y_train, new_yh_train, show_confusion=True)
report("Validation set", y_val, new_yh_test, show_confusion=True)


# ----------------------------------------------------------------------------
# Predict both hidden sets with the quantum model and write the prediction files
# ----------------------------------------------------------------------------
import os
section("Hidden test predictions (quantum model)")
os.makedirs("predictions", exist_ok=True)
for name, df in [("hidden_a", hid_a), ("hidden_b", hid_b)]:
    # Same steps as validation: prepare -> circuit -> standardize -> readout
    F = standardize(reservoir_features(circuit, prepare(df)))
    pred = predict_proba(readout, F).argmax(axis=1)
    pd.DataFrame({"sample_id": df["sample_id"], "label": pred}).to_csv(f"predictions/{name}.csv", index=False)
    print(f"Saved predictions/{name}.csv ({len(pred)} rows). Predicted {class_counts(pred)}")

from qiskit.primitives import StatevectorEstimator
import torch.nn as nn

# 1. Helper to run your frozen Qiskit encoder on a dataset -> returns (N_samples, 16) array
def extract_qiskit_features(data_matrix, circuit, observables, batch_size=256):
    estimator = StatevectorEstimator()
    angles = data_matrix * np.pi
    all_expectations = []

    # Batching keeps memory usage low across thousands of network traffic rows
    for start in range(0, len(angles), batch_size):
        batch_angles = angles[start:start + batch_size]
        # PUB format: (circuit, observables_column_vector, parameter_values_matrix)
        pub = (circuit, [[obs] for obs in observables], batch_angles)
        result = estimator.run([pub]).result()[0]
        # result.data.evs has shape (batch_size, 16)
        all_expectations.append(result.data.evs)

    return np.vstack(all_expectations)

# 2. Extract the 16 Pauli-Z expectation values (<Z_0> ... <Z_15>) for train and val
section("Running Qiskit Reservoir")
Z_train = extract_qiskit_features(X_train, feature_map, observables_list)
Z_val = extract_qiskit_features(X_val, feature_map, observables_list)
print(f"Extracted Qiskit features: train {Z_train.shape} | val {Z_val.shape}")

# 3. Train the exact same classical readout head: Linear(16 -> 3)
qiskit_head = nn.Linear(n_qubits, N_CLASSES, dtype=torch.float64)

weights_tensor = torch.tensor(CLASS_WEIGHTS_NP, dtype=torch.float64) if USE_CLASS_WEIGHTS else None
criterion = nn.CrossEntropyLoss(weight=weights_tensor)
optimizer = torch.optim.Adam(qiskit_head.parameters(), lr=0.05)

Z_train_t = torch.tensor(Z_train, dtype=torch.float64)
y_train_t = torch.tensor(y_train, dtype=torch.long)

for epoch in range(200):
    optimizer.zero_grad()
    logits = qiskit_head(Z_train_t)
    loss = criterion(logits, y_train_t)
    loss.backward()
    optimizer.step()
