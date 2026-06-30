import numpy as np
import pandas as pd
import joblib

def _anova_kernel(X1, X2, gamma=0.01, degree=2):
    K = np.zeros((X1.shape[0], X2.shape[0]))
    for d in range(X1.shape[1]):
        diff = X1[:, d:d+1] - X2[:, d].reshape(1, -1)
        K += np.exp(-gamma * diff**2)
    return K ** degree

def load_model(path: str = "models/model_terbaik.joblib"):
    return joblib.load(path)

def prediksi_n_hari(model_dict, df_input, n_hari):
    from utils.preprocessing import FITUR, LAG

    model  = model_dict["model"]
    gamma  = model_dict.get("gamma", 0.01)
    degree = model_dict.get("degree", 2)
    min_f  = model_dict["min_fitur"]
    max_f  = model_dict["max_fitur"]
    min_c  = model_dict["min_curah"]
    max_c  = model_dict["max_curah"]

    X_train_lag = model_dict["X_train"]
    X_tr = X_train_lag.values if hasattr(X_train_lag, 'values') else X_train_lag

    # ── Normalisasi input ────────────────────────────────────────────────────
    df = df_input[FITUR].copy()
    df['IMERG_PRECTOT'] = np.log1p(df['IMERG_PRECTOT'])
    df = (df - min_f) / (max_f - min_f + 1e-8)
    window = df.values[-3:].tolist()
    hasil  = []

    kernel_type = model.kernel if hasattr(model, 'kernel') else 'precomputed'

    for _ in range(n_hari):
        # Susun lag features: tiap fitur t-3, t-2, t-1
        row = []
        for f_idx in range(len(FITUR)):
            for lag in range(3, 0, -1):
                row.append(window[-lag][f_idx])
        X_pred = np.array(row).reshape(1, -1)  # (1, 18)

        # Prediksi sesuai kernel
        if kernel_type == 'precomputed':
            # Pastikan X_tr juga 18 fitur
            if X_tr.shape[1] != 18:
                raise ValueError(f"Joblib lama tidak kompatibel (shape: {X_tr.shape}). Silakan retrain model.")
            K = _anova_kernel(X_pred, X_tr, gamma, degree)
            pred_norm = np.clip(model.predict(K)[0], 0, 1)
        else:
            # Linear, Poly, RBF — langsung predict
            pred_norm = np.clip(model.predict(X_pred)[0], 0, 1)

        # Denormalisasi
        pred_log = pred_norm * (max_c - min_c) + min_c
        pred_mm  = max(0.0, float(np.expm1(pred_log)))
        hasil.append(round(pred_mm, 2))

        # Geser window
        new_row    = list(window[-1])
        new_row[5] = pred_norm
        window.append(new_row)
        if len(window) > 3:
            window.pop(0)

    return hasil

def buat_df_hasil(prediksi, tanggal_mulai=None):
    if tanggal_mulai is None:
        tanggal_mulai = pd.Timestamp.today().normalize()
    tanggal = pd.date_range(tanggal_mulai, periods=len(prediksi), freq="D")
    return pd.DataFrame({
        "Hari ke-":                  range(1, len(prediksi) + 1),
        "Tanggal":                   tanggal.strftime("%Y-%m-%d"),
        "Prediksi Curah Hujan (mm)": prediksi,
    })