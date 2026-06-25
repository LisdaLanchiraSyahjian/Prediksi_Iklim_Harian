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
    """Log1p transform lalu MinMax normalize."""
    df = df.copy()
    # Log1p transform pada target
    df[TARGET] = np.log1p(df[TARGET])
    # MinMax semua fitur
    min_fitur = df[FITUR].min()
    max_fitur = df[FITUR].max()
    df[FITUR] = (df[FITUR] - min_fitur) / (max_fitur - min_fitur + 1e-8)
    min_curah = df[TARGET].min()
    max_curah = df[TARGET].max()
    return df, min_fitur, max_fitur, min_curah, max_curah

def buat_lag_features(df: pd.DataFrame):
    cols = {}
    for f in FITUR:
        for l in range(LAG, 0, -1):
            cols[f"{f}_t-{l}"] = df[f].shift(l)
    lag_df = pd.DataFrame(cols)
    lag_df[TARGET] = df[TARGET].values
    lag_df = lag_df.dropna().reset_index(drop=True)
    X = lag_df.drop(columns=[TARGET])
    y = lag_df[TARGET]
    return X, y

def split_data(X, y, rasio: str = "80:20"):
    pct = int(rasio.split(":")[0]) / 100
    n = int(len(X) * pct)
    return X.iloc[:n], X.iloc[n:], y.iloc[:n], y.iloc[n:]
