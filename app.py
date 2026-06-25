import streamlit as st

st.set_page_config(
    page_title="Prediksi Iklim Harian",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
[data-testid="stSidebarNav"] { display: none; }
section[data-testid="stSidebar"] { min-width: 230px; max-width: 230px; }
</style>
""", unsafe_allow_html=True)

# --- Session state defaults ---
defaults = {
    "df_raw":         None,
    "df_processed":   None,
    "min_fitur":      None,
    "max_fitur":      None,
    "min_curah":      0.0,
    "max_curah":      1.0,
    "rasio_split":    "80:20",
    "X_train": None, "X_test": None,
    "y_train": None, "y_test": None,
    "C":              1.0,
    "epsilon":        0.1,
    "gamma_rbf":      1,
    "gamma_anova":    0.01,
    "degree_poly":    2,
    "degree_anova":   2,
    "hasil_training": None,
    "df_evaluasi":    None,
    "model_terbaik":  None,
    "nama_terbaik":   None,
    "hasil_prediksi": None,
    "halaman":        "Upload Data",
    "step_selesai":   0,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

STEPS = [
    ("📂", "Upload Data"),
    ("✂️",  "Pembagian Data"),
    ("⚙️", "Kernel & Training"),
    ("🔮", "Prediksi"),
]

with st.sidebar:
    st.markdown("## 🌧️ Prediksi Iklim Harian")
    st.markdown("**Perbandingan Kernel SVR**")
    st.divider()
    for i, (icon, nama) in enumerate(STEPS):
        aktif  = st.session_state["halaman"] == nama
        selesai = st.session_state["step_selesai"] >= i
        tipe   = "primary" if aktif else "secondary"
        label  = f"{icon} {nama}" + (" ✓" if selesai and not aktif else "")
        if st.button(label, key=f"nav_{i}", use_container_width=True, type=tipe):
            st.session_state["halaman"] = nama
            st.rerun()
    st.divider()
    st.caption("Prediksi Curah Hujan\nJawa Timur")

halaman = st.session_state["halaman"]
if halaman == "Upload Data":
    from pages._1_upload   import render; render()
elif halaman == "Pembagian Data":
    from pages._2_split    import render; render()
elif halaman == "Kernel & Training":
    from pages._3_kernel   import render; render()
elif halaman == "Prediksi":
    from pages._4_prediksi import render; render()
