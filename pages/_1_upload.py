import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def render():
    st.title("📂 Upload Data")
    st.markdown("Upload dataset curah hujan harian dalam format **CSV**.")

    uploaded = st.file_uploader("Drag & drop file CSV di sini", type=["csv"])

    if uploaded:
        try:
            from utils.preprocessing import load_and_validate, FITUR, preprocess
            df = load_and_validate(uploaded)
            missing = [k for k in FITUR if k not in df.columns]
            if missing:
                st.error(f"❌ Kolom tidak ditemukan: {missing}")
                return

            df_proc, min_f, max_f, min_c, max_c = preprocess(df)

            st.session_state["df_raw"]       = df
            st.session_state["df_processed"] = df_proc
            st.session_state["min_fitur"]    = min_f
            st.session_state["max_fitur"]    = max_f
            st.session_state["min_curah"]    = min_c
            st.session_state["max_curah"]    = max_c
            st.session_state["step_selesai"] = max(st.session_state["step_selesai"], 1)

            st.success(f"✅ **{uploaded.name}** berhasil diupload")

        except Exception as e:
            st.error(f"Gagal membaca file: {e}")
            return

    df = st.session_state.get("df_raw")
    if df is None:
        return

    from utils.preprocessing import FITUR, LABEL_FITUR

    # Info ringkas
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📋 Total Data", f"{len(df):,} baris")
    with col2:
        st.metric("📊 Jumlah Fitur", len(FITUR))
    with col3:
        if "Tanggal" in df.columns:
            st.metric("📅 Rentang Waktu",
                      f"{df['Tanggal'].min().strftime('%Y')} – {df['Tanggal'].max().strftime('%Y')}")

    st.divider()

    # Grafik semua fitur sekaligus
    st.subheader("📈 Grafik Semua Parameter")
    tgl = df["Tanggal"].astype(str).tolist() if "Tanggal" in df.columns else list(df.index)

    WARNA = ["#3B82F6", "#EF4444", "#22C55E", "#F59E0B", "#8B5CF6", "#EC4899"]
    fig = make_subplots(
        rows=len(FITUR), cols=1,
        shared_xaxes=True,
        subplot_titles=[LABEL_FITUR[f] for f in FITUR],
        vertical_spacing=0.04,
    )
    for i, fitur in enumerate(FITUR):
        fig.add_trace(
            go.Scatter(x=tgl, y=df[fitur].tolist(),
                       mode="lines", name=LABEL_FITUR[fitur],
                       line=dict(color=WARNA[i], width=1.2)),
            row=i+1, col=1,
        )
    fig.update_layout(
        height=200 * len(FITUR),
        margin=dict(l=0, r=0, t=30, b=0),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    if st.button("Lanjut ke Pembagian Data →", type="primary"):
        st.session_state["halaman"] = "Pembagian Data"
        st.rerun()
