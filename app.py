import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Buenaventura | Smart Maintenance", page_icon="⛏️", layout="wide")
st.title("⛏️ Buenaventura | Smart Maintenance")
st.caption("Prototipo académico de mantenimiento predictivo para camiones mineros")
st.warning("DEMO: utiliza datos sintéticos, no datos reales de Buenaventura. No usar para decisiones operativas reales.")

@st.cache_resource
def train_model():
    rng = np.random.default_rng(42)
    n = 2500
    temp = rng.normal(82, 12, n).clip(45, 125)
    vib = rng.normal(5, 2, n).clip(.5, 14)
    hours = rng.integers(100, 18000, n)
    since = rng.integers(0, 1200, n)
    score = .035*(temp-75) + .48*(vib-4) + .00010*(hours-6000) + .0012*(since-300) + rng.normal(0, 1, n)
    y = (score > 2).astype(int)
    X = pd.DataFrame({"Temperatura (°C)":temp, "Vibración (mm/s)":vib,
                      "Horas de operación":hours, "Horas desde último mantenimiento":since})
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.25, random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=180, max_depth=8, min_samples_leaf=4,
                                   random_state=42, class_weight="balanced")
    model.fit(Xtr, ytr)
    return model, accuracy_score(yte, model.predict(Xte)), list(X.columns)

model, demo_accuracy, features = train_model()
st.subheader("1. Registrar condiciones del camión")
c1, c2 = st.columns(2)
with c1:
    temp = st.slider("Temperatura del motor (°C)", 40, 125, 82)
    vib = st.slider("Vibración (mm/s)", .5, 14.0, 5.0, .1)
with c2:
    hours = st.number_input("Horas acumuladas de operación", 0, 50000, 8500, 100)
    since = st.number_input("Horas desde el último mantenimiento", 0, 3000, 350, 25)

if st.button("Analizar condición del equipo", type="primary", use_container_width=True):
    row = pd.DataFrame([{"Temperatura (°C)":temp, "Vibración (mm/s)":vib,
                         "Horas de operación":hours, "Horas desde último mantenimiento":since}])
    prob = float(model.predict_proba(row)[0][1])
    if prob < .30:
        level, icon, advice = "BAJO", "🟢", "Mantener el monitoreo periódico y cumplir el plan preventivo."
    elif prob < .65:
        level, icon, advice = "MODERADO", "🟠", "Programar una inspección técnica y revisar tendencias."
    else:
        level, icon, advice = "ELEVADO", "🔴", "Priorizar una inspección técnica y verificar motor, componentes móviles e historial."
    st.divider()
    st.subheader("2. Resultado")
    a,b = st.columns(2)
    a.metric("Nivel estimado", level)
    a.metric("Probabilidad estimada", f"{prob:.1%}")
    b.markdown(f"## {icon} {level}")
    b.write(advice)
    st.info("Probabilidad y umbrales demostrativos. En un uso real, el modelo debe entrenarse, calibrarse y validarse con datos históricos y sensores de la operación.")

with st.expander("¿Cómo funciona?"):
    st.write("Se genera un conjunto de datos sintéticos y se entrena un modelo Random Forest. En una implementación real, podrían integrarse datos de sensores/SCADA o PI System y registros de mantenimiento del ERP, previa validación.")
    st.caption(f"Exactitud en datos sintéticos de prueba: {demo_accuracy:.1%}. No representa desempeño real en minería.")

st.subheader("3. Variables consideradas")
st.dataframe(pd.DataFrame({"Variable":features, "Descripción":["Temperatura del motor","Nivel de vibración","Horas acumuladas de operación","Tiempo desde el último mantenimiento"]}), use_container_width=True, hide_index=True)
st.caption("Prototipo académico | Datos simulados | Versión 1.0")
