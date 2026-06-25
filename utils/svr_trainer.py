import numpy as np
from sklearn.svm import SVR
from sklearn.metrics.pairwise import rbf_kernel

# Kombinasi parameter default terbaik sesuai notebook
KOMBINASI_DEFAULT = {
    "Linear":                  {"kernel": "linear",      "C": 1,    "epsilon": 0.1},
    "Polynomial (degree=2)":   {"kernel": "poly",        "C": 1,    "epsilon": 0.1, "degree": 2, "coef0": 1},
    "Polynomial (degree=3)":   {"kernel": "poly",        "C": 1,    "epsilon": 0.1, "degree": 3, "coef0": 1},
    "RBF (gamma=1)":           {"kernel": "rbf",         "C": 1,    "epsilon": 0.1, "gamma": 1},
    "RBF (gamma=2)":           {"kernel": "rbf",         "C": 1,    "epsilon": 0.1, "gamma": 2},
    "ANOVA RBF (g=0.01,d=2)":  {"kernel": "precomputed", "C": 1,    "epsilon": 0.1, "gamma": 0.01, "degree": 2},
    "ANOVA RBF (g=0.01,d=4)":  {"kernel": "precomputed", "C": 1,    "epsilon": 0.1, "gamma": 0.01, "degree": 4},
    "ANOVA RBF (g=2,d=2)":     {"kernel": "precomputed", "C": 1,    "epsilon": 0.1, "gamma": 2,    "degree": 2},
    "ANOVA RBF (g=2,d=4)":     {"kernel": "precomputed", "C": 1,    "epsilon": 0.1, "gamma": 2,    "degree": 4},
}

def _anova_kernel(X1, X2, gamma=0.01, degree=2):
    K = np.zeros((X1.shape[0], X2.shape[0]))
    for d in range(X1.shape[1]):
        K += rbf_kernel(X1[:, d:d+1], X2[:, d:d+1], gamma=gamma)
    return K ** degree

def train_semua(X_train, X_test, y_train, y_test, C=1, epsilon=0.1,
                gamma_rbf=1, gamma_anova=0.01, degree_poly=2, degree_anova=2):
    """Train semua kombinasi kernel dengan parameter dari user."""
    X_tr = X_train.values if hasattr(X_train, 'values') else X_train
    X_te = X_test.values  if hasattr(X_test,  'values') else X_test
    y_tr = y_train.values if hasattr(y_train, 'values') else y_train
    y_te = y_test.values  if hasattr(y_test,  'values') else y_test

    kombinasi = {
        "Linear": {
            "type": "linear", "C": C, "epsilon": epsilon
        },
        f"Polynomial (degree={degree_poly})": {
            "type": "poly", "C": C, "epsilon": epsilon,
            "degree": degree_poly, "coef0": 1
        },
        f"RBF (gamma={gamma_rbf})": {
            "type": "rbf", "C": C, "epsilon": epsilon, "gamma": gamma_rbf
        },
        f"ANOVA RBF (g={gamma_anova},d={degree_anova})": {
            "type": "precomputed", "C": C, "epsilon": epsilon,
            "gamma": gamma_anova, "degree": degree_anova
        },
    }

    hasil = {}
    for nama, params in kombinasi.items():
        tipe = params["type"]
        if tipe == "precomputed":
            g, d = params["gamma"], params["degree"]
            K_tr = _anova_kernel(X_tr, X_tr, g, d)
            K_te = _anova_kernel(X_te, X_tr, g, d)
            model = SVR(kernel="precomputed", C=params["C"], epsilon=params["epsilon"])
            model.fit(K_tr, y_tr)
            pred_train = model.predict(K_tr)
            pred_test  = model.predict(K_te)
        else:
            kwargs = {"kernel": tipe, "C": params["C"], "epsilon": params["epsilon"]}
            if tipe == "poly":
                kwargs["degree"] = params["degree"]
                kwargs["coef0"]  = params["coef0"]
            if tipe == "rbf":
                kwargs["gamma"]  = params["gamma"]
            model = SVR(**kwargs)
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
