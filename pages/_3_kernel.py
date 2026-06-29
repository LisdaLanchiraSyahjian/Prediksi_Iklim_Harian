import streamlit as st
import plotly.graph_objects as go

WARNA = {
    "Linear":     "#3B82F6",
    "Polynomial": "#F59E0B",
    "RBF":        "#22C55E",
    "ANOVA RBF":  "#EF4444",
}

def warna_kernel(nama):
    for k, w in WARNA.items():
        if k in nama:
            return w
    return "#888"

def render():
    st.title("⚙️ Kernel & Training SVR")
    st.markdown("Pilih parameter dan latih kombinasi kernel SVR.")

    if st.session_state["X_train"] is None:
        st.warning("⚠️ Pembagian data belum selesai.")
        return

    # ── Parameter ─────────────────────────────────────────────────────────────
    st.subheader("🔧 Parameter SVR")
    st.caption("Parameter default adalah kombinasi terbaik dari hasil evaluasi penelitian.")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**🔵 Linear**")
        C_linear = st.selectbox("C — Linear", [1, 10], index=1)

        st.markdown("**🟡 Polynomial** *(C=10 fixed)*")
        degree_poly = st.selectbox("Degree — Polynomial", [2, 3], index=0)

    with col2:
        st.markdown("**🟢 RBF** *(C=10 fixed)*")
        gamma_rbf = st.selectbox("Gamma — RBF", [0.01, 2], index=0)

        st.markdown("**🔴 ANOVA RBF** *(C=10 fixed)*")
        gamma_anova  = st.selectbox("Gamma — ANOVA RBF", [0.01, 2], index=0)
        degree_anova = st.selectbox("Degree — ANOVA RBF", [2, 3], index=0)

    st.info("""
    **💡 Rekomendasi parameter terbaik dari evaluasi penelitian:**
    Linear C=10 · Polynomial C=10, D=2 · RBF C=10, γ=0.01 · ANOVA RBF C=10, γ=0.01, D=2
    """)

    run = st.button("🚀 Mulai Training", type="primary", use_container_width=True)

    if run:
        st.session_state["hasil_training"] = None
        st.session_state["df_evaluasi"]    = None

    if st.session_state["hasil_training"] is None and run:
        with st.spinner("⏳ Melatih kernel... Harap tunggu."):
            from utils.svr_trainer import train_satu
            hasil = train_satu(
                st.session_state["X_train"], st.session_state["X_test"],
                st.session_state["y_train"], st.session_state["y_test"],
                C_linear=C_linear, degree_poly=degree_poly,
                gamma_rbf=gamma_rbf,
                gamma_anova=gamma_anova, degree_anova=degree_anova,
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

        st.subheader("📉 Perbandingan Metrik (Test Set)")
        kernels   = df_eval["Kernel"].tolist()
        warna_bar = [warna_kernel(k) for k in kernels]

        col1, col2, col3 = st.columns(3)
        for col, metrik, judul in [
            (col1, "MAE Test",  "MAE ↓"),
            (col2, "RMSE Test", "RMSE ↓"),
            (col3, "R² Test",   "R² ↑"),
        ]:
            with col:
                vals = df_eval[metrik].tolist()
                fig = go.Figure(go.Bar(
                    x=kernels, y=vals, marker_color=warna_bar,
                    text=[f"{v:.3f}" for v in vals], textposition="outside",
                ))
                fig.update_layout(
                    title=judul, height=300,
                    margin=dict(l=0, r=0, t=40, b=60),
                    xaxis_tickangle=-30,
                    yaxis=dict(range=[0, max(vals)*1.3]),
                )
                st.plotly_chart(fig, use_container_width=True)

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