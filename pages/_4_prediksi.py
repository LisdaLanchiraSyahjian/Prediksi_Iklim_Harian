import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import io, os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "model_terbaik.joblib")

FITUR = ['ALLSKY_SFC_SW_DWN', 'T2M', 'WS2M', 'QV2M', 'PSC', 'IMERG_PRECTOT']

INFO_FITUR = {
    'ALLSKY_SFC_SW_DWN': ('☀️ Radiasi Matahari',    'kW-hr/m²/day', 0.5,  26.1, 15.0),
    'T2M':               ('🌡️ Suhu Udara',           '°C',           22.0, 33.0, 27.0),
    'WS2M':              ('💨 Kecepatan Angin',       'm/s',          0.3,  5.0,  1.5),
    'QV2M':              ('💧 Kelembaban Spesifik',   'g/kg',         10.0, 22.0, 16.0),
    'PSC':               ('🌀 Tekanan Permukaan',     'kPa',          99.0, 101.0, 100.2),
    'IMERG_PRECTOT':     ('🌧️ Curah Hujan',           'mm/hari',      0.0,  50.0, 2.0),
}

# ── Kategori curah hujan BMKG ─────────────────────────────────────────────────
def kategori_curah_hujan(mm):
    if mm < 5:
        return "☀️ Ringan"
    elif mm < 20:
        return "🌦️ Sedang"
    elif mm < 50:
        return "🌧️ Lebat"
    else:
        return "⛈️ Sangat Lebat"

def warna_bar_kategori(mm):
    if mm < 5:
        return "#FCD34D"
    elif mm < 20:
        return "#60A5FA"
    elif mm < 50:
        return "#34D399"
    else:
        return "#F87171"

def highlight_row(row):
    styles = {
        "☀️ Ringan":       "background-color:#FEF9C3; color:#854D0E;",
        "🌦️ Sedang":       "background-color:#DBEAFE; color:#1E40AF;",
        "🌧️ Lebat":        "background-color:#DCFCE7; color:#166534;",
        "⛈️ Sangat Lebat": "background-color:#FEE2E2; color:#991B1B;",
    }
    style = styles.get(row["Kategori"], "")
    return [style] * len(row)


def render():
    st.title("🔮 Prediksi Curah Hujan Harian")
    st.markdown("Masukkan data iklim **3 hari terakhir** untuk prediksi ke depan.")

    from utils.predictor import load_model, prediksi_n_hari, buat_df_hasil

    # ── Tentukan model yang dipakai ───────────────────────────────────────────
    use_trained = (
        st.session_state.get("model_terbaik") is not None
        and st.session_state.get("nama_terbaik") is not None
    )

    if use_trained:
        nama_model  = st.session_state["nama_terbaik"]
        ht          = st.session_state["model_terbaik"]
        X_train_lag = pd.DataFrame(ht["X_train"], columns=st.session_state["X_train"].columns)
        params      = ht.get("params", {})
        gamma       = params.get("gamma",  st.session_state.get("gamma_anova", 0.01))
        degree      = params.get("degree", st.session_state.get("degree_anova", 2))
        model_dict  = {
            "model":     ht["model"],
            "X_train":   X_train_lag,
            "gamma":     gamma,
            "degree":    degree,
            "min_fitur": st.session_state["min_fitur"],
            "max_fitur": st.session_state["max_fitur"],
            "min_curah": st.session_state["min_curah"],
            "max_curah": st.session_state["max_curah"],
        }
        st.info(f"✅ Model dari training: **{nama_model}**")
    else:
        st.info("ℹ️ Menggunakan model tersimpan: **model_terbaik.joblib (ANOVA RBF)**")
        try:
            model_dict = load_model(MODEL_PATH)
            nama_model = model_dict.get("nama_model", "ANOVA RBF")
        except Exception as e:
            st.error(f"Gagal load model: {e}")
            return

    st.divider()

    # ── Pilih jumlah hari ─────────────────────────────────────────────────────
    n_hari = st.selectbox(
        "⏱️ Jumlah Hari Prediksi ke Depan",
        [3, 7, 14, 21, 30],
        index=0,
    )

    st.divider()

    # ── Input manual 3 hari ───────────────────────────────────────────────────
    st.subheader("📋 Input Data Iklim 3 Hari Terakhir")
    st.caption("Isi nilai tiap parameter untuk **t-3** (3 hari lalu), **t-2** (2 hari lalu), **t-1** (kemarin).")

    input_data = {}
    for fitur in FITUR:
        label, satuan, vmin, vmax, vdef = INFO_FITUR[fitur]
        st.markdown(f"**{label}** — *{satuan}*")
        c1, c2, c3 = st.columns(3)
        with c1:
            v3 = st.number_input("t-3 (3 hari lalu)", min_value=float(vmin), max_value=float(vmax),
                                  value=float(vdef), step=0.01, format="%.2f", key=f"{fitur}_t3")
        with c2:
            v2 = st.number_input("t-2 (2 hari lalu)", min_value=float(vmin), max_value=float(vmax),
                                  value=float(vdef), step=0.01, format="%.2f", key=f"{fitur}_t2")
        with c3:
            v1 = st.number_input("t-1 (kemarin)",     min_value=float(vmin), max_value=float(vmax),
                                  value=float(vdef), step=0.01, format="%.2f", key=f"{fitur}_t1")
        input_data[fitur] = [v3, v2, v1]
        st.markdown("&nbsp;")

    st.divider()

    if st.button("🚀 Proses Prediksi", type="primary", use_container_width=True):
        with st.spinner("Menghitung prediksi..."):
            try:
                df_input = pd.DataFrame(
                    [{f: input_data[f][i] for f in FITUR} for i in range(3)]
                )
                pred     = prediksi_n_hari(model_dict, df_input, n_hari)
                df_hasil = buat_df_hasil(pred)
                st.session_state["hasil_prediksi"] = df_hasil
                st.session_state["step_selesai"]   = max(st.session_state["step_selesai"], 4)
            except Exception as e:
                st.error(f"Error prediksi: {e}")
                return

    # ── Tampilkan hasil ───────────────────────────────────────────────────────
    if st.session_state["hasil_prediksi"] is not None:
        df_hasil = st.session_state["hasil_prediksi"]

        st.divider()
        st.subheader(f"📊 Hasil Prediksi {len(df_hasil)} Hari ke Depan")

        # Tambah kolom kategori
        df_tampil = df_hasil.copy()
        df_tampil["Kategori"] = df_tampil["Prediksi Curah Hujan (mm)"].apply(kategori_curah_hujan)

        # Tabel dengan highlight warna per kategori
        st.dataframe(
            df_tampil.style.apply(highlight_row, axis=1).format({
                "Prediksi Curah Hujan (mm)": "{:.2f}"
            }),
            use_container_width=True, hide_index=True
        )

        st.markdown("&nbsp;")

        # ── Ringkasan kategori ────────────────────────────────────────────────
        st.subheader("📌 Ringkasan Kategori Curah Hujan")
        counts = df_tampil["Kategori"].value_counts()
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("☀️ Ringan",       counts.get("☀️ Ringan", 0),       "< 5 mm")
        with col2:
            st.metric("🌦️ Sedang",       counts.get("🌦️ Sedang", 0),       "5–20 mm")
        with col3:
            st.metric("🌧️ Lebat",        counts.get("🌧️ Lebat", 0),        "20–50 mm")
        with col4:
            st.metric("⛈️ Sangat Lebat", counts.get("⛈️ Sangat Lebat", 0), "> 50 mm")

        st.markdown("&nbsp;")

        # ── Grafik bar warna per kategori ─────────────────────────────────────
        warna_bar = [warna_bar_kategori(mm) for mm in df_tampil["Prediksi Curah Hujan (mm)"]]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=df_tampil["Tanggal"],
            y=df_tampil["Prediksi Curah Hujan (mm)"],
            marker_color=warna_bar,
            text=df_tampil["Kategori"],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>%{y:.2f} mm<br>%{text}<extra></extra>",
        ))
        fig.update_layout(
            title=f"Grafik Prediksi Curah Hujan — {len(df_hasil)} Hari ke Depan",
            xaxis_title="Tanggal",
            yaxis_title="Curah Hujan (mm)",
            height=430,
            margin=dict(l=0, r=0, t=50, b=0),
            hovermode="x unified",
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "🟡 **Ringan** < 5 mm &nbsp;|&nbsp; "
            "🔵 **Sedang** 5–20 mm &nbsp;|&nbsp; "
            "🟢 **Lebat** 20–50 mm &nbsp;|&nbsp; "
            "🔴 **Sangat Lebat** > 50 mm &nbsp;&nbsp;"
            "*(Kategori berdasarkan standar BMKG)*"
        )

        # ── Download Excel ────────────────────────────────────────────────────
        st.divider()
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
            df_tampil.to_excel(writer, index=False, sheet_name="Prediksi")
            if st.session_state.get("df_evaluasi") is not None:
                st.session_state["df_evaluasi"].to_excel(
                    writer, index=False, sheet_name="Evaluasi Model")
        buffer.seek(0)

        st.download_button(
            label="⬇️ Download Hasil Excel",
            data=buffer,
            file_name=f"prediksi_curah_hujan_{len(df_hasil)}hari.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
