import streamlit as st
import plotly.graph_objects as go

WARNA = {
    "Linear":              "#3B82F6",
    "Polynomial":          "#F59E0B",
    "RBF":                 "#22C55E",
    "ANOVA RBF":           "#EF4444",
}

def warna_kernel(nama):
    for k, w in WARNA.items():
        if k in nama:
            return w
    return "#888"

def render():
    st.title("⚙️ Kernel & Training SVR")
    st.markdown("Atur parameter dan latih semua kombinasi kernel SVR.")

    if st.session_state["X_train"] is None:
        st.warning("⚠️ Pembagian data belum selesai.")
        return

    # ── Parameter ────────────────────────────────────────────────────────────
    st.subheader("🔧 Parameter SVR")
    col1, col2 = st.columns(2)
    with col1:
        C = st.number_input("C (Regularisasi)", min_value=0.01, max_value=100.0,
                            value=float(st.session_state["C"]), step=0.1, format="%.2f")
        epsilon = st.number_input("Epsilon", min_value=0.001, max_value=1.0,
                                  value=float(st.session_state["epsilon"]), step=0.01, format="%.3f")
    with col2:
        gamma_rbf   = st.number_input("Gamma RBF",        min_value=0.001, max_value=10.0,
                                      value=float(st.session_state["gamma_rbf"]),   step=0.1,   format="%.3f")
        gamma_anova = st.number_input("Gamma ANOVA RBF",  min_value=0.0001, max_value=10.0,
                                      value=float(st.session_state["gamma_anova"]), step=0.001, format="%.4f")

    col3, col4 = st.columns(2)
    with col3:
        degree_poly = st.number_input("Degree Polynomial", min_value=1, max_value=5,
                                      value=int(st.session_state["degree_poly"]), step=1)
    with col4:
        degree_anova = st.number_input("Degree ANOVA RBF", min_value=1, max_value=5,
                                       value=int(st.session_state["degree_anova"]), step=1)

    st.info(f"""
    **Kombinasi default terbaik dari penelitian:**
    Linear (C=1) · Polynomial (C=1, degree=2 & 3) ·
    RBF (C=1, gamma=1 & 2) · ANOVA RBF (C=1, gamma=0.01/2, degree=2/4)
    """)

    run = st.button("🚀 Mulai Training Semua Kernel", type="primary", use_container_width=True)

    # Simpan parameter ke session
    st.session_state.update({
        "C": C, "epsilon": epsilon,
        "gamma_rbf": gamma_rbf, "gamma_anova": gamma_anova,
        "degree_poly": degree_poly, "degree_anova": degree_anova,
    })

    if run:
        st.session_state["hasil_training"] = None
        st.session_state["df_evaluasi"]    = None

    if st.session_state["hasil_training"] is None and run:
        with st.spinner("⏳ Melatih semua kernel... Harap tunggu."):
            from utils.svr_trainer import train_semua
            hasil = train_semua(
                st.session_state["X_train"], st.session_state["X_test"],
                st.session_state["y_train"], st.session_state["y_test"],
                C=C, epsilon=epsilon,
                gamma_rbf=gamma_rbf, gamma_anova=gamma_anova,
                degree_poly=degree_poly, degree_anova=degree_anova,
            )
            from utils.evaluator import evaluasi_semua, pilih_model_terbaik
            df_eval = evaluasi_semua(
                hasil,
                st.session_state["y_train"], st.session_state["y_test"],
                st.session_state["min_curah"], st.session_state["max_curah"],
            )
            nama_terbaik = pilih_model_terbaik(df_eval)
            st.session_state["hasil_training"] = hasil
            st.session_state["df_evaluasi"]    = df_eval
            st.session_state["nama_terbaik"]   = nama_terbaik
            st.session_state["model_terbaik"]  = hasil[nama_terbaik]
            st.session_state["step_selesai"]   = max(st.session_state["step_selesai"], 3)

    # ── Tampilkan hasil ───────────────────────────────────────────────────────
    if st.session_state["df_evaluasi"] is not None:
        df_eval      = st.session_state["df_evaluasi"]
        nama_terbaik = st.session_state["nama_terbaik"]

        st.divider()
        st.subheader(f"📊 Hasil Evaluasi — Split {st.session_state['rasio_split']}")

        def highlight(row):
            bg = "background-color:#FEF9C3;" if row["Kernel"] == nama_terbaik else ""
            return [bg] * len(row)

        st.dataframe(
            df_eval.style.apply(highlight, axis=1).format({
                c: "{:.3f}" for c in df_eval.columns if c != "Kernel"
            }),
            use_container_width=True, hide_index=True,
        )

        # Bar chart 3 metrik
        st.subheader("📉 Perbandingan Metrik (Test Set)")
        kernels = df_eval["Kernel"].tolist()
        warna_bar = [warna_kernel(k) for k in kernels]

        col1, col2, col3 = st.columns(3)
        for col, metrik, judul, kecil in [
            (col1, "MAE Test",  "MAE ↓",  True),
            (col2, "RMSE Test", "RMSE ↓", True),
            (col3, "R² Test",   "R² ↑",   False),
        ]:
            with col:
                vals = df_eval[metrik].tolist()
                fig = go.Figure(go.Bar(
                    x=kernels, y=vals, marker_color=warna_bar,
                    text=[f"{v:.3f}" for v in vals], textposition="outside",
                ))
                fig.update_layout(
                    title=judul, height=300,
                    margin=dict(l=0,r=0,t=40,b=60),
                    xaxis_tickangle=-30,
                    yaxis=dict(range=[0, max(vals)*1.3]),
                )
                st.plotly_chart(fig, use_container_width=True)

        # Model terbaik
        st.divider()
        row = df_eval[df_eval["Kernel"] == nama_terbaik].iloc[0]
        st.markdown(f"### 🏆 Model Terbaik: `{nama_terbaik}`")
        c1, c2, c3 = st.columns(3)
        c1.metric("MAE Test",  f"{row['MAE Test']:.3f}")
        c2.metric("RMSE Test", f"{row['RMSE Test']:.3f}")
        c3.metric("R² Test",   f"{row['R² Test']:.3f}")

        st.markdown("&nbsp;")
        if st.button("Lanjut ke Prediksi →", type="primary"):
            st.session_state["halaman"] = "Prediksi"
            st.rerun()
