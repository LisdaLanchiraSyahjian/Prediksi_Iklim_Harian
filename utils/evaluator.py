import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def denormalisasi(arr, min_val, max_val):
    return np.array(arr) * (max_val - min_val) + min_val

def hitung_metrik(y_true, y_pred, min_curah, max_curah):
    # Denormalisasi dulu, lalu balik dari log1p
    y_true = np.expm1(denormalisasi(y_true, min_curah, max_curah))
    y_pred = np.expm1(denormalisasi(y_pred, min_curah, max_curah))
    return {
        "MAE":  round(mean_absolute_error(y_true, y_pred), 3),
        "RMSE": round(np.sqrt(mean_squared_error(y_true, y_pred)), 3),
        "R²":   round(r2_score(y_true, y_pred), 3),
    }

def evaluasi_semua(hasil_training, y_train, y_test, min_curah, max_curah):
    rows = []
    for nama, data in hasil_training.items():
        tr = hitung_metrik(y_train, data["pred_train"], min_curah, max_curah)
        te = hitung_metrik(y_test,  data["pred_test"],  min_curah, max_curah)
        rows.append({
            "Kernel":     nama,
            "MAE Train":  tr["MAE"],  "RMSE Train": tr["RMSE"], "R² Train": tr["R²"],
            "MAE Test":   te["MAE"],  "RMSE Test":  te["RMSE"], "R² Test":  te["R²"],
        })
    return pd.DataFrame(rows)

def pilih_model_terbaik(df_eval):
    idx = df_eval["R² Test"].idxmax()
    return df_eval.loc[idx, "Kernel"]
