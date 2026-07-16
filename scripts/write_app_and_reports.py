from pathlib import Path


files = {
    "app/streamlit_app.py": '''from pathlib import Path
import sys

import pandas as pd
import streamlit as st

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

st.set_page_config(
    page_title="Equipment Diagnostics ML",
    page_icon="⚙️",
    layout="wide"
)

st.title("ML-система диагностики электронного оборудования")

st.markdown("""
Проект решает задачу диагностики технического состояния электронного модуля
по температурным и вибрационным временным рядам.
""")

st.subheader("Pipeline проекта")

st.code("""
raw sensor time series
        ↓
windowing
        ↓
time-domain features + frequency-domain features
        ↓
ML classification
        ↓
diagnostic report
""", language="text")

st.subheader("Классы технического состояния")

classes = pd.DataFrame(
    {
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
            "Штатный режим работы",
            "Перегрев силовых компонентов",
            "Постепенное ухудшение охлаждения",
            "Повышенный общий уровень вибрации",
            "Дисбаланс вращающегося узла",
            "Износ механического узла с импульсными вибрациями",
            "Комбинированная тепловая и вибрационная неисправность"
        ]
    }
)

st.dataframe(classes, use_container_width=True)

st.subheader("Результаты pipeline")

st.code("""
data/raw/sensor_timeseries.csv
data/interim/windows.csv
data/processed/features.csv

models/baseline_model.pkl
models/best_model.pkl
models/label_encoder.pkl

reports/metrics/baseline_metrics.csv
reports/metrics/classic_metrics.csv
reports/metrics/best_model_report.txt
reports/metrics/best_model_confusion_matrix.csv
""", language="text")

st.info("Для анализа данных, диагностики и интерпретации модели используй страницы слева.")
''',

    "app/pages/1_data_overview.py": '''from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

project_root = Path(__file__).resolve().parents[2]

st.set_page_config(page_title="Data overview", layout="wide")

st.title("Обзор данных")

raw_path = project_root / "data" / "raw" / "sensor_timeseries.csv"
features_path = project_root / "data" / "processed" / "features.csv"

if not raw_path.exists() or not features_path.exists():
    st.error("Данные не найдены. Сначала запусти python run_pipeline.py")
    st.stop()

features = pd.read_csv(features_path)

st.subheader("Распределение классов")

class_counts = features["state"].value_counts().reset_index()
class_counts.columns = ["state", "count"]

fig = px.bar(
    class_counts,
    x="state",
    y="count",
    title="Количество окон по классам"
)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Сводка по ключевым признакам")

metric_columns = [
    "temperature_chip_mean",
    "temperature_chip_max",
    "temperature_delta_mean",
    "vibration_magnitude_rms",
    "vibration_magnitude_max",
    "vibration_magnitude_dominant_frequency"
]

existing_columns = [column for column in metric_columns if column in features.columns]

summary = features.groupby("state")[existing_columns].mean().round(3)
st.dataframe(summary, use_container_width=True)

st.subheader("Температурные признаки")

fig = px.box(
    features,
    x="state",
    y="temperature_chip_mean",
    title="Средняя температура кристалла по классам"
)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Вибрационные признаки")

fig = px.box(
    features,
    x="state",
    y="vibration_magnitude_rms",
    title="RMS вибрации по классам"
)
st.plotly_chart(fig, use_container_width=True)

if "vibration_magnitude_dominant_frequency" in features.columns:
    st.subheader("Частотный признак вибрации")

    fig = px.box(
        features,
        x="state",
        y="vibration_magnitude_dominant_frequency",
        title="Доминирующая частота вибрации по классам"
    )
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Пример временного ряда")

raw_sample = pd.read_csv(raw_path, nrows=60000)
states = raw_sample["state"].unique().tolist()

selected_state = st.selectbox("Выбери класс", states)

experiment_ids = raw_sample[raw_sample["state"] == selected_state]["experiment_id"].unique().tolist()
selected_experiment = st.selectbox("Выбери experiment_id", experiment_ids)

experiment = raw_sample[raw_sample["experiment_id"] == selected_experiment]

fig = px.line(
    experiment,
    x="timestamp",
    y=["temperature_chip", "temperature_case"],
    title="Температурные сигналы"
)
st.plotly_chart(fig, use_container_width=True)

fig = px.line(
    experiment,
    x="timestamp",
    y=["vibration_x", "vibration_y", "vibration_z"],
    title="Вибрационные сигналы"
)
st.plotly_chart(fig, use_container_width=True)
''',

    "app/pages/2_prediction.py": '''from pathlib import Path
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.models.predict import predict_from_features

st.set_page_config(page_title="Prediction", layout="wide")

st.title("Диагностика состояния оборудования")

features_path = project_root / "data" / "processed" / "features.csv"
model_path = project_root / "models" / "best_model.pkl"

if not features_path.exists() or not model_path.exists():
    st.error("Модель или признаки не найдены. Сначала запусти python run_pipeline.py")
    st.stop()

st.markdown("""
На этой странице можно проверить работу модели на подготовленных признаках.
Каждая строка соответствует одному окну временного ряда.
""")

uploaded_file = st.file_uploader(
    "Загрузи CSV с признаками или используй демонстрационный датасет",
    type=["csv"]
)

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
else:
    full_data = pd.read_csv(features_path)
    data = full_data.sample(n=min(200, len(full_data)), random_state=42).reset_index(drop=True)

st.subheader("Входные данные")
st.dataframe(data.head(20), use_container_width=True)

if st.button("Выполнить диагностику"):
    predictions = predict_from_features(data)

    st.subheader("Результаты диагностики")

    columns = ["predicted_state"]

    if "state" in predictions.columns:
        columns = ["state", "predicted_state"]

    if "prediction_confidence" in predictions.columns:
        columns.append("prediction_confidence")

    st.dataframe(predictions[columns].head(50), use_container_width=True)

    result_counts = predictions["predicted_state"].value_counts().reset_index()
    result_counts.columns = ["predicted_state", "count"]

    fig = px.bar(
        result_counts,
        x="predicted_state",
        y="count",
        title="Распределение предсказанных состояний"
    )
    st.plotly_chart(fig, use_container_width=True)

    if "prediction_confidence" in predictions.columns:
        st.metric(
            "Средняя уверенность модели",
            f"{predictions['prediction_confidence'].mean():.3f}"
        )

    st.subheader("Инженерная интерпретация")

    main_state = predictions["predicted_state"].mode()[0]

    explanations = {
        "normal": "Сигналы соответствуют штатному режиму: температура и вибрация находятся в допустимом диапазоне.",
        "overheating": "Вероятен перегрев: температурные признаки имеют повышенные значения.",
        "cooling_degradation": "Вероятно ухудшение охлаждения: температура растёт при сохранении рабочих режимов.",
        "vibration_fault": "Обнаружен повышенный уровень вибрации, возможна механическая неисправность.",
        "imbalance": "Сигнал похож на дисбаланс: выражена периодическая вибрационная составляющая.",
        "bearing_wear": "Сигнал похож на износ механического узла: возможны импульсные вибрационные компоненты.",
        "combined_fault": "Обнаружены признаки тепловой и вибрационной неисправности одновременно."
    }

    st.info(explanations.get(main_state, "Для данного состояния описание не задано."))
''',

    "app/pages/3_model_explanation.py": '''from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

project_root = Path(__file__).resolve().parents[2]

st.set_page_config(page_title="Model explanation", layout="wide")

st.title("Интерпретация модели")

model_path = project_root / "models" / "best_model.pkl"
features_path = project_root / "data" / "processed" / "features.csv"
metrics_path = project_root / "reports" / "metrics" / "classic_metrics.csv"
report_path = project_root / "reports" / "metrics" / "best_model_report.txt"
matrix_path = project_root / "reports" / "metrics" / "best_model_confusion_matrix.csv"

if not model_path.exists() or not features_path.exists():
    st.error("Модель или признаки не найдены. Сначала запусти python run_pipeline.py")
    st.stop()

model = joblib.load(model_path)
features = pd.read_csv(features_path)

drop_columns = [
    "state",
    "global_window_id",
    "experiment_id",
    "window_start",
    "window_end"
]

feature_columns = [column for column in features.columns if column not in drop_columns]

st.subheader("Сравнение моделей")

if metrics_path.exists():
    metrics = pd.read_csv(metrics_path)
    st.dataframe(metrics, use_container_width=True)

    fig = px.bar(
        metrics,
        x="model",
        y="f1_macro",
        title="F1-macro по моделям"
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("Файл с метриками моделей не найден.")

st.subheader("Classification report")

if report_path.exists():
    st.text(report_path.read_text(encoding="utf-8"))
else:
    st.warning("Classification report не найден.")

st.subheader("Confusion matrix")

if matrix_path.exists():
    matrix = pd.read_csv(matrix_path, index_col=0)

    fig = px.imshow(
        matrix,
        text_auto=True,
        title="Матрица ошибок"
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("Матрица ошибок не найдена.")

st.subheader("Важность признаков")

if hasattr(model, "feature_importances_"):
    importance = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance": model.feature_importances_
        }
    ).sort_values("importance", ascending=False).head(25)

    fig = px.bar(
        importance.sort_values("importance"),
        x="importance",
        y="feature",
        orientation="h",
        title="Топ-25 признаков по важности"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(importance, use_container_width=True)
else:
    st.info("Текущая модель не поддерживает feature_importances_.")

st.subheader("Как читать результат")

st.markdown("""
Важные признаки показывают, какие свойства окна сигнала сильнее всего используются моделью.

1. Высокие температурные признаки указывают на перегрев или деградацию охлаждения.
2. RMS и max вибрации указывают на общий уровень механических колебаний.
3. Доминирующая частота помогает отличать дисбаланс от других вибрационных проблем.
4. Энергия в диапазонах частот показывает, где сосредоточена вибрационная активность.
""")
''',

    "src/visualization/plots.py": '''import plotly.express as px


def plot_class_distribution(data):
    counts = data["state"].value_counts().reset_index()
    counts.columns = ["state", "count"]

    return px.bar(
        counts,
        x="state",
        y="count",
        title="Class distribution"
    )


def plot_signal(data, x_column, y_columns, title):
    return px.line(
        data,
        x=x_column,
        y=y_columns,
        title=title
    )


def plot_feature_box(data, feature_column):
    return px.box(
        data,
        x="state",
        y=feature_column,
        title=f"{feature_column} by state"
    )
''',

    "src/visualization/dashboard_utils.py": '''def get_state_description(state):
    descriptions = {
        "normal": "Штатный режим работы.",
        "overheating": "Повышенная температура силовых компонентов.",
        "cooling_degradation": "Постепенное ухудшение эффективности охлаждения.",
        "vibration_fault": "Повышенный общий уровень вибрации.",
        "imbalance": "Периодическая вибрация, похожая на дисбаланс.",
        "bearing_wear": "Импульсные вибрационные компоненты, похожие на износ механического узла.",
        "combined_fault": "Одновременные признаки тепловой и вибрационной неисправности."
    }

    return descriptions.get(state, "Описание состояния не задано.")


def get_risk_level(state, confidence):
    if state == "normal":
        return "low"

    if confidence >= 0.8:
        return "high"

    if confidence >= 0.5:
        return "medium"

    return "uncertain"
''',

    "docs/project_description.md": '''# Project description

Проект посвящён разработке ML-системы диагностики технического состояния электронного оборудования по температурным и вибрационным временным рядам.

## Цель

Разработать воспроизводимый pipeline, который по данным датчиков определяет состояние оборудования и помогает инженеру быстрее обнаруживать потенциальные неисправности.

## Входные данные

Используются временные ряды датчиков:

1. temperature_chip
2. temperature_case
3. vibration_x
4. vibration_y
5. vibration_z
6. current
7. voltage
8. load
9. fan_speed

## Классы

1. normal
2. overheating
3. cooling_degradation
4. vibration_fault
5. imbalance
6. bearing_wear
7. combined_fault

## Основная идея

Классифицируется не отдельное измерение, а окно временного ряда. Для каждого окна извлекаются временные и частотные признаки, после чего ML-модель определяет техническое состояние оборудования.
''',

    "docs/system_design.md": '''# System design

## Pipeline

```text
raw sensor signals
        ↓
windowing
        ↓
feature extraction
        ↓
model training
        ↓
evaluation
        ↓
diagnostic application