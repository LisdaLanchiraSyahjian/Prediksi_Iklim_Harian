def train_satu(X_train, X_test, y_train, y_test,
               C_linear=10, C_poly=10, degree_poly=2,
               C_rbf=10, gamma_rbf=0.01,
               C_anova=10, gamma_anova=0.01, degree_anova=2,
               epsilon=0.1):
    """Train 4 kernel dengan parameter pilihan user."""
    X_tr = X_train.values if hasattr(X_train, 'values') else X_train
    X_te = X_test.values  if hasattr(X_test,  'values') else X_test
    y_tr = y_train.values if hasattr(y_train, 'values') else y_train

    kombinasi = {
        f"Linear (C={C_linear})": {
            "type": "linear", "C": C_linear, "epsilon": epsilon
        },
        f"Polynomial (C={C_poly}, D={degree_poly})": {
            "type": "poly", "C": C_poly, "epsilon": epsilon, "degree": degree_poly
        },
        f"RBF (C={C_rbf}, g={gamma_rbf})": {
            "type": "rbf", "C": C_rbf, "epsilon": epsilon, "gamma": gamma_rbf
        },
        f"ANOVA RBF (C={C_anova}, g={gamma_anova}, D={degree_anova})": {
            "type": "precomputed", "C": C_anova, "epsilon": epsilon,
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