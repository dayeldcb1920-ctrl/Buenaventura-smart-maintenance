import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from datetime import datetime

st.set_page_config(
    page_title="Buenaventura | Smart Maintenance",
    page_icon="⛏️",
    layout="wide"
)

st.title("⛏️ Buenaventura | Smart Maintenance")
st.caption("Prototipo académico de mantenimiento predictivo para camiones mineros")

st.warning(
    "DEMOSTRACIÓN ACADÉMICA: utiliza datos sintéticos, "
    "no datos reales de Buenaventura. No usar para decisiones operativas reales."
)


@st.cache_resource
def train_model():
    rng = np.random.default_rng(42)
    n = 3000

    t = rng.normal(82, 12, n).clip(45, 125)
    v = rng.normal(5, 2, n).clip(0.5, 14)
    h = rng.integers(100, 18000, n)
    s = rng.integers(0, 1200, n)

    score = (
        0.035 * (t - 75)
        + 0.48 * (v - 4)
        + 0.00010 * (h - 6000)
        + 0.0012 * (s - 300)
        + rng.normal(0, 1, n)
    )

    y = (score > 2).astype(int)

    X = pd.DataFrame({
        "Temperatura (°C)": t,
        "Vibración (mm/s)": v,
        "Horas de operación": h,
        "Horas desde último mantenimiento": s
    })

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=180,
        max_depth=8,
        min_samples_leaf=4,
        random_state=42,
        class_weight="balanced"
    )

    model.fit(Xtr, ytr)
    return model


model = train_model()

if "history" not in st.session_state:
    st.session_state.history = []


def assess(t, v, h, s):
    x = pd.DataFrame([{
        "Temperatura (°C)": t,
        "Vibración (mm/s)": v,
        "Horas de operación": h,
        "Horas desde último mantenimiento": s
    }])

    prob = float(model.predict_proba(x)[0][1])

    if prob < 0.30:
        risk = "BAJO"
        advice = "Mantener el monitoreo periódico y cumplir el plan preventivo."
    elif prob < 0.65:
        risk = "MODERADO"
        advice = "Programar una inspección técnica y revisar las tendencias."
    else:
        risk = "ELEVADO"
        advice = (
            "Priorizar una inspección técnica y verificar motor, "
            "componentes móviles e historial."
        )

    return prob, risk, advice


fleet = pd.DataFrame([
    ["Camión BM-01", 82, 5.0, 8500, 350],
    ["Camión BM-02", 110, 12.0, 15000, 1000],
    ["Camión BM-03", 91, 7.1, 11200, 620],
    ["Camión BM-04", 73, 3.2, 6200, 180],
    ["Camión BM-05", 99, 9.0, 13200, 800]
], columns=[
    "Equipo",
    "Temperatura (°C)",
    "Vibración (mm/s)",
    "Horas de operación",
    "Horas desde último mantenimiento"
])


tab1, tab2, tab3 = st.tabs([
    "📊 Panel de flota",
    "🔍 Analizar camión",
    "🗂️ Historial"
])


with tab1:
    st.subheader("Panel general de camiones mineros")

    st.caption(
        "Flota ficticia de demostración; los valores no corresponden a equipos reales."
    )

    res = [
        assess(r.iloc[1], r.iloc[2], r.iloc[3], r.iloc[4])
        for _, r in fleet.iterrows()
    ]

    fleet["Probabilidad estimada (%)"] = [
        round(x[0] * 100, 1) for x in res
    ]

    fleet["Riesgo"] = [x[1] for x in res]

    a, b, c = st.columns(3)

    a.metric("Equipos monitoreados", len(fleet))
    b.metric("Riesgo elevado", sum(x[1] == "ELEVADO" for x in res))
    c.metric("Riesgo moderado", sum(x[1] == "MODERADO" for x in res))

    st.dataframe(fleet, use_container_width=True, hide_index=True)

    st.subheader("Probabilidad estimada por equipo")

    st.bar_chart(
        fleet.set_index("Equipo")[["Probabilidad estimada (%)"]]
    )

    st.info(
        "Panel demostrativo para visualizar y priorizar inspecciones. "
        "No representa el estado real de ninguna unidad minera."
    )


with tab2:
    st.subheader("Registrar condiciones del camión")

    a, b = st.columns(2)

    with a:
        equipo = st.text_input("Código o nombre del equipo", "Camión BM-06")
        temp = st.slider("Temperatura del motor (°C)", 40, 125, 82)
        vib = st.slider("Vibración (mm/s)", 0.5, 14.0, 5.0, 0.1)

    with b:
        horas = st.number_input(
            "Horas acumuladas de operación", 0, 50000, 8500, 100
        )

        servicio = st.number_input(
            "Horas desde el último mantenimiento", 0, 3000, 350, 25
        )

    if st.button(
        "Analizar condición del equipo",
        type="primary",
        use_container_width=True
    ):
        prob, risk, advice = assess(temp, vib, horas, servicio)

        result = {
            "Fecha y hora": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Equipo": equipo,
            "Temperatura (°C)": temp,
            "Vibración (mm/s)": vib,
            "Horas de operación": horas,
            "Horas desde último mantenimiento": servicio,
            "Probabilidad estimada (%)": round(prob * 100, 1),
            "Riesgo": risk,
            "Recomendación": advice
        }

        st.session_state.last = result
        st.session_state.history.append(result)

    if "last" in st.session_state:
        r = st.session_state.last

        a, b = st.columns(2)

        a.metric("Nivel estimado de riesgo", r["Riesgo"])
        a.metric(
            "Probabilidad estimada",
            f'{r["Probabilidad estimada (%)"]:.1f}%'
        )

        emoji = (
            "🟢" if r["Riesgo"] == "BAJO"
            else "🟠" if r["Riesgo"] == "MODERADO"
            else "🔴"
        )

        b.markdown("## " + emoji + " " + r["Riesgo"])
        b.write(r["Recomendación"])

        st.info(
            "Probabilidad y umbrales demostrativos. Un uso real requiere "
            "entrenamiento, calibración y validación con datos históricos y sensores."
        )


with tab3:
    st.subheader("Historial de análisis de esta sesión")

    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)

        st.dataframe(df, use_container_width=True, hide_index=True)

        st.download_button(
            "Descargar historial en CSV",
            df.to_csv(index=False).encode("utf-8-sig"),
            file_name="historial_mantenimiento_demo.csv",
            mime="text/csv"
        )

        st.subheader("Cantidad de análisis por nivel de riesgo")

        st.bar_chart(
            df["Riesgo"].value_counts().reindex(
                ["BAJO", "MODERADO", "ELEVADO"],
                fill_value=0
            )
        )

    else:
        st.info(
            "Todavía no hay análisis registrados. "
            "Ve a «Analizar camión» y ejecuta una predicción."
        )


with st.expander("¿Cómo funciona el prototipo?"):
    st.write(
        "El sistema entrena un modelo Random Forest con datos sintéticos "
        "y estima una probabilidad según temperatura, vibración, "
        "horas de operación y tiempo desde el último mantenimiento."
    )

    st.write(
        "En una implementación real podrían evaluarse datos de sensores, "
        "SCADA o PI System y registros de mantenimiento del ERP, "
        "previa validación de disponibilidad y calidad."
    )

st.caption("Prototipo académico | Datos simulados | Versión 2.0")
