from pathlib import Path
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Equipment Diagnostics ML",
    page_icon="⚙️",
    layout="wide"
)

st.title("ML-система диагностики электронного оборудования")

st.markdown("""
Проект диагностирует состояние электронного оборудования по температурным и вибрационным временным рядам.

Pipeline проекта:
""")

st.code("""
raw sensor time series
        ↓
windowing
        ↓
feature extraction
        ↓
model training
        ↓
diagnostic application
""", language="text")

classes = pd.DataFrame({
    "Класс": [
        "normal",
        "overheating",
        "cooling_degradation",
        "vibration_fault",
        "imbalance",
        "bearing_wear",
        "combined_fault"
    ],
    "Описание": [
        "Штатный режим",
        "Перегрев",
        "Деградация охлаждения",
        "Повышенная вибрация",
        "Дисбаланс",
        "Износ механического узла",
        "Комбинированная неисправность"
    ]
})

st.subheader("Классы")
st.dataframe(classes, use_container_width=True)
