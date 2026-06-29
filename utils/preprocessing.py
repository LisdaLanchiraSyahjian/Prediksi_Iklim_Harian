import pandas as pd
import numpy as np

FITUR = ['ALLSKY_SFC_SW_DWN', 'T2M', 'WS2M', 'QV2M', 'PSC', 'IMERG_PRECTOT']
TARGET = 'IMERG_PRECTOT'
LAG = 3

LABEL_FITUR = {
    'ALLSKY_SFC_SW_DWN': 'Radiasi Matahari (kW-hr/m²/day)',
    'T2M':               'Suhu Udara (°C)',
    'WS2M':              'Kecepatan Angin (m/s)',
    'QV2M':              'Kelembaban Spesifik (g/kg)',
    'PSC':               'Tekanan Permukaan (kPa)',
    'IMERG_PRECTOT':     'Curah Hujan (mm/hari)',
}

def load_and_validate(file):
    df = pd.read_csv(file)
    df.columns = df.columns.str.strip()
    if 'Tanggal' in df.columns:
        df['Tanggal'] = pd.to_datetime(df['Tanggal'])
        df = df.sort_values('Tanggal').reset_index(drop=True)
    return df

def preprocess(df: pd.DataFrame):
    df = df.copy()

    # Log1p transform pada target dulu
    df[TARGET] = np.log1p(df[TARGET])

    # Simpan min/max curah SETELAH log1p tapi SEBELUM normalisasi
    min_curah = df[TARGET].min()
    max_curah = df[TARGET].max()

    # MinMax semua fitur
    min_fitur = df[FITUR].min()
    max_fitur = df[FITUR].max()
    for col in FITUR:
        df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

    return df, min_fitur, max_fitur, min_curah, max_curah

def buat_lag_features(df: pd.DataFrame):
    data_window = []

    for i in range(3, len(df)):
        row = []
        for kolom in FITUR:
            row.append(df[kolom].iloc[i-3])
            row.append(df[kolom].iloc[i-2])
            row.append(df[kolom].iloc[i-1])
        row.append(df[TARGET].iloc[i])
        data_window.append(row)

    columns = []
    for kolom in FITUR:
        columns.append(f"{kolom}_t-3")
        columns.append(f"{kolom}_t-2")
        columns.append(f"{kolom}_t-1")
    columns.append("Target_t")

    df_window = pd.DataFrame(data_window, columns=columns)
    X = df_window.drop(columns=["Target_t"])
    y = df_window["Target_t"]
    return X, y

def split_data(X, y, rasio: str = "80:20"):
    pct = int(rasio.split(":")[0]) / 100
    n = int(len(X) * pct)
    return X.iloc[:n], X.iloc[n:], y.iloc[:n], y.iloc[n:]