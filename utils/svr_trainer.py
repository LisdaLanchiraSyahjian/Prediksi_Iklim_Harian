import numpy as np
from sklearn.svm import SVR

def _anova_kernel(X1, X2, gamma, degree):
    K = np.zeros((X1.shape[0], X2.shape[0]))
    for d in range(X1.shape[1]):
        diff = X1[:, d:d+1] - X2[:, d].reshape(1, -1)
        K += np.exp(-gamma * diff**2)
    return K ** degree

# Kombinasi tetap sesuai skenario penelitian
KOMBINASI_DEFAULT = {
    "Linear (C=1)":                    {"type": "linear",      "C": 1,  "epsilon": 0.1},
    "Linear (C=10)":                   {"type": "linear",      "C": 10, "epsilon": 0.1},
    "Polynomial (C=10, D=2)":          {"type": "poly",        "C": 10, "epsilon": 0.1, "degree": 2},
    "Polynomial (C=10, D=3)":          {"type": "poly",        "C": 10, "epsilon": 0.1, "degree": 3},
    "RBF (C=10, g=0.01)":             {"type": "rbf",         "C": 10, "epsilon": 0.1, "gamma": 0.01},
    "RBF (C=10, g=2)":                {"type": "rbf",         "C": 10, "epsilon": 0.1, "gamma": 2},
    "ANOVA RBF (C=10, g=0.01, D=2)":  {"type": "precomputed", "C": 10, "epsilon": 0.1, "gamma": 0.01, "degree": 2},
    "ANOVA RBF (C=10, g=0.01, D=3)":  {"type": "precomputed", "C": 10, "epsilon": 0.1, "gamma": 0.01, "degree": 3},
    "ANOVA RBF (C=10, g=2, D=2)":     {"type": "precomputed", "C": 10, "epsilon": 0.1, "gamma": 2,    "degree": 2},
    "ANOVA RBF (C=10, g=2, D=3)":     {"type": "precomputed", "C": 10, "epsilon": 0.1, "gamma": 2,    "degree": 3},
}

def train_semua(X_train, X_test, y_train, y_test, **kwargs):
    """Train semua 10 kombinasi kernel sesuai skenario penelitian. kwargs diabaikan."""
    X_tr = X_train.values if hasattr(X_train, 'values') else X_train
    X_te = X_test.values  if hasattr(X_test,  'values') else X_test
    y_tr = y_train.values if hasattr(y_train, 'values') else y_train

    hasil = {}
    for nama, params in KOMBINASI_DEFAULT.items():
        tipe = params["type"]

        if tipe == "precomputed":
            g, d = params["gamma"], params["degree"]
            K_tr = _anova_kernel(X_tr, X_tr, g, d)
            K_te = _anova_kernel(X_te, X_tr, g, d)
            model = SVR(kernel="precomputed", C=params["C"], epsilon=params["epsilon"])
            model.fit(K_tr, y_tr)
            pred_train = model.predict(K_tr)
            pred_test  = model.predict(K_te)

        elif tipe == "poly":
            model = SVR(kernel="poly", C=params["C"], epsilon=params["epsilon"],
                        degree=params["degree"])
            model.fit(X_tr, y_tr)
            pred_train = model.predict(X_tr)
            pred_test  = model.predict(X_te)

        elif tipe == "rbf":
            model = SVR(kernel="rbf", C=params["C"], epsilon=params["epsilon"],
                        gamma=params["gamma"])
            model.fit(X_tr, y_tr)
            pred_train = model.predict(X_tr)
            pred_test  = model.predict(X_te)

        elif tipe == "linear":
            model = SVR(kernel="linear", C=params["C"], epsilon=params["epsilon"])
            model.fit(X_tr, y_tr)
            pred_train = model.predict(X_tr)
            pred_test  = model.predict(X_te)

        hasil[nama] = {
            "model":      model,
            "pred_train": pred_train,
            "pred_test":  pred_test,
            "X_train":    X_tr,
            "params":     params,
        }

    return hasil