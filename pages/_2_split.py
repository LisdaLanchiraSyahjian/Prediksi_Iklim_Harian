import streamlit as st

RASIO_OPTIONS = ["50:50", "60:40", "70:30", "80:20", "90:10"]

def render():
    st.title("✂️ Pembagian Data")
    st.markdown("Pilih rasio **Training : Testing** untuk pemodelan.")

    if st.session_state["df_processed"] is None:
        st.warning("⚠️ Belum ada data. Silakan upload CSV dulu.")
        return

    rasio = st.radio(
        "Rasio Pembagian Data",
        RASIO_OPTIONS,
        index=RASIO_OPTIONS.index(st.session_state["rasio_split"])
              if st.session_state["rasio_split"] in RASIO_OPTIONS else 3,
        horizontal=True,
    )

    from utils.preprocessing import buat_lag_features, split_data
    df = st.session_state["df_processed"]
    X, y = buat_lag_features(df)
    X_train, X_test, y_train, y_test = split_data(X, y, rasio)

    pct_train = int(rasio.split(":")[0])
    pct_test  = 100 - pct_train

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.metric("🟦 Data Training", f"{len(X_train):,} baris", delta=f"{pct_train}%")
    with col2:
        st.metric("🟩 Data Testing", f"{len(X_test):,} baris", delta=f"{pct_test}%")

    bar_html = f"""
    <div style='border-radius:8px; overflow:hidden; height:36px; display:flex; margin-top:12px;'>
        <div style='width:{pct_train}%; background:#2563EB; display:flex; align-items:center;
                    justify-content:center; color:white; font-weight:700;'>Training {pct_train}%</div>
        <div style='width:{pct_test}%; background:#86EFAC; display:flex; align-items:center;
                    justify-content:center; color:#065F46; font-weight:700;'>Testing {pct_test}%</div>
    </div>"""
    st.markdown(bar_html, unsafe_allow_html=True)

    st.markdown("&nbsp;")
    if st.button("Lanjut ke Kernel & Training →", type="primary"):
        st.session_state["rasio_split"] = rasio
        st.session_state["X_train"]     = X_train
        st.session_state["X_test"]      = X_test
        st.session_state["y_train"]     = y_train
        st.session_state["y_test"]      = y_test
        st.session_state["hasil_training"] = None   # reset kalau ganti rasio
        st.session_state["df_evaluasi"]    = None
        st.session_state["step_selesai"]   = max(st.session_state["step_selesai"], 2)
        st.session_state["halaman"]        = "Kernel & Training"
        st.rerun()
