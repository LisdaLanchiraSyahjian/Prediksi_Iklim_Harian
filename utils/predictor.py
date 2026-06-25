import numpy as np
import pandas as pd
import joblib
from sklearn.metrics.pairwise import rbf_kernel

def _anova_kernel(X1, X2, gamma=0.01, degree=2):
    """X1: (n_pred, n_features), X2: (n_train, n_features) → output (n_pred, n_train)"""
    K = np.zeros((X1.shape[0], X2.shape[0]))
    for d in range(X1.shape[1]):
        K += rbf_kernel(X1[:, d:d+1], X2[:, d:d+1], gamma=gamma)
    return K ** degree

def load_model(path: str = "models/model_terbaik.joblib"):
    return joblib.load(path)

def prediksi_n_hari(model_dict, df_input, n_hari):
    from utils.preprocessing import FITUR, LAG

    model   = model_dict["model"]
    gamma   = model_dict["gamma"]
    degree  = model_dict["degree"]
    min_f   = model_dict["min_fitur"]
    max_f   = model_dict["max_fitur"]
    min_c   = model_dict["min_curah"]
    max_c   = model_dict["max_curah"]

    # X_train yang disimpan di model_dict adalah lag features (n_train × 18)
    X_train_lag = model_dict["X_train"]
    X_tr = X_train_lag.values if hasattr(X_train_lag, 'values') else X_train_lag
    # X_tr shape: (n_train_samples, 18) ← ini yang benar untuk kernel

    # ── Normalisasi 3 baris input manual ────────────────────────────────────
    df = df_input[FITUR].copy()
    df['IMERG_PRECTOT'] = np.log1p(df['IMERG_PRECTOT'])
    df = (df - min_f) / (max_f - min_f + 1e-8)
    # df shape: (3, 6) — 3 hari × 6 fitur

    # window = list of 3 rows, tiap row adalah list 6 nilai (normalized)
    window = df.values[-3:].tolist()   # [[val_t3...], [val_t2...], [val_t1...]]
    hasil  = []

    for _ in range(n_hari):
        # Susun 1 baris lag features: urutan t-3,t-2,t-1 untuk tiap fitur
        # Hasil: 6 fitur × 3 lag = 18 nilai
        row = []
        for f_idx in range(len(FITUR)):
            for lag in range(3, 0, -1):          # lag=3 → index -3, lag=2 → -2, lag=1 → -1
                row.append(window[-lag][f_idx])
        X_pred = np.array(row).reshape(1, -1)    # shape: (1, 18)

        # Kernel: (1, 18) vs (n_train, 18) → (1, n_train) ← benar untuk SVR precomputed
        K         = _anova_kernel(X_pred, X_tr, gamma, degree)   # (1, n_train)
        pred_norm = np.clip(model.predict(K)[0], 0, 1)

        # Denormalisasi + inverse log → mm asli
        pred_log = pred_norm * (max_c - min_c) + min_c
        pred_mm  = max(0.0, float(np.expm1(pred_log)))
        hasil.append(round(pred_mm, 2))

        # Geser window: tambah baris baru, buang yang paling lama
        new_row    = list(window[-1])             # copy hari terakhir
        new_row[5] = pred_norm                    # update IMERG_PRECTOT (index 5)
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
