from pathlib import Path
import sys

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.models.predict import predict_from_features
from src.product.operator_dashboard import create_operator_snapshot
from src.product.alert_generation import generate_alerts
from src.product.maintenance_plan import create_maintenance_plan


st.set_page_config(
    page_title="Диагностика СВЧ-модулей",
    page_icon="⚙️",
    layout="wide"
)


STATE_NAMES = {
    "normal": "Норма",
    "overheating": "Перегрев",
    "cooling_degradation": "Ухудшение охлаждения",
    "vibration_fault": "Повышенная вибрация",
    "imbalance": "Дисбаланс",
    "bearing_wear": "Износ механического узла",
    "combined_fault": "Комбинированная неисправность"
}


FEATURE_DISPLAY_NAMES = {
    "temperature_chip_mean": "Средняя температура кристалла",
    "temperature_chip_max": "Максимальная температура кристалла",
    "temperature_chip_trend": "Тренд температуры кристалла",
    "temperature_case_mean": "Средняя температура корпуса",
    "temperature_delta_mean": "Средняя разность температур кристалл-корпус",
    "temperature_delta_max": "Максимальная разность температур кристалл-корпус",
    "temperature_chip_above_70_ratio": "Доля времени выше 70 °C",
    "temperature_chip_above_80_ratio": "Доля времени выше 80 °C",
    "vibration_magnitude_rms": "Среднеквадратичный уровень вибрации",
    "vibration_magnitude_max": "Максимальный уровень вибрации",
    "vibration_magnitude_crest_factor": "Пик-фактор вибрации",
    "vibration_magnitude_dominant_frequency": "Доминирующая частота вибрации",
    "vibration_magnitude_spectral_centroid": "Спектральный центр вибрации",
    "vibration_magnitude_energy_0_10_hz": "Энергия вибрации 0–10 Гц",
    "vibration_magnitude_energy_10_25_hz": "Энергия вибрации 10–25 Гц",
    "vibration_magnitude_energy_25_50_hz": "Энергия вибрации 25–50 Гц",
    "vibration_magnitude_high_frequency_energy_ratio": "Доля высокочастотной энергии",
    "load_mean": "Средняя нагрузка",
    "current_mean": "Средний ток",
    "voltage_mean": "Среднее напряжение",
    "fan_speed_mean": "Средняя скорость вентилятора"
}


def load_features():
    features_path = project_root / "data" / "processed" / "features.csv"

    if not features_path.exists():
        st.error("Файл с признаками не найден. Сначала запусти pipeline.")
        st.stop()

    return pd.read_csv(features_path)


def load_predictions(sample_size=1500, random_state=42):
    features = load_features()
    data = features.sample(n=min(sample_size, len(features)), random_state=random_state).reset_index(drop=True)
    predictions = predict_from_features(data)

    return data, predictions


def show_header(role):
    st.title("Система предиктивной диагностики СВЧ-модулей")
    st.caption(f"Текущая роль: {role}")


def show_operator_view():
    show_header("Оператор")

    st.markdown("""
    Рабочий экран оператора показывает состояние СВЧ-модулей на испытательных стендах.
    Основная задача оператора — быстро увидеть опасные состояния и принять решение:
    продолжить наблюдение, проверить модуль или остановить испытание.
    """)

    _, predictions = load_predictions(sample_size=1500, random_state=42)
    snapshot = create_operator_snapshot(predictions)

    visible = snapshot[
        [
            "Модуль",
            "Стенд",
            "Состояние",
            "Уровень риска",
            "Рекомендация"
        ]
    ].copy()

    critical_count = visible["Уровень риска"].str.contains("Критический").sum()
    high_count = visible["Уровень риска"].str.contains("Высокий").sum()
    normal_count = visible["Уровень риска"].str.contains("Низкий").sum()

    col1, col2, col3 = st.columns(3)

    col1.metric("Критический риск", int(critical_count))
    col2.metric("Высокий риск", int(high_count))
    col3.metric("Норма", int(normal_count))

    st.subheader("Состояние модулей")
    st.dataframe(visible, use_container_width=True, hide_index=True)

    st.subheader("Предупреждения")

    alerts = generate_alerts(predictions)

    if alerts.empty:
        st.success("Активных предупреждений нет.")
    else:
        alert_visible = alerts[
            [
                "Модуль",
                "Стенд",
                "Состояние",
                "Уровень риска",
                "Предупреждение",
                "Рекомендация"
            ]
        ].copy()

        st.warning(f"Найдено предупреждений: {len(alert_visible)}")
        st.dataframe(alert_visible, use_container_width=True, hide_index=True)

        critical_alerts = alert_visible[alert_visible["Уровень риска"].str.contains("Критический")]

        for _, row in critical_alerts.head(5).iterrows():
            st.error(
                f"{row['Модуль']} на {row['Стенд']}: {row['Состояние']}. "
                f"Действие: {row['Рекомендация']}."
            )


def show_technician_view():
    show_header("Инженер-техник")

    st.markdown("""
    Экран инженера-техника показывает очередь проверки оборудования.
    В план попадают модули с высоким или критическим риском.
    """)

    _, predictions = load_predictions(sample_size=1500, random_state=11)
    plan = create_maintenance_plan(predictions)

    if plan.empty:
        st.success("План проверки пуст: модулей с высоким риском не обнаружено.")
        return

    visible = plan[
        [
            "Приоритет",
            "Модуль",
            "Стенд",
            "Состояние",
            "Уровень риска",
            "Задача для техника"
        ]
    ].copy()

    st.subheader("План проверки")
    st.dataframe(visible, use_container_width=True, hide_index=True)

    st.subheader("Ближайшие задачи")

    for _, row in visible.head(7).iterrows():
        st.markdown(
            f"**Приоритет {int(row['Приоритет'])}. {row['Модуль']}** — "
            f"{row['Состояние']}. {row['Задача для техника']}."
        )


def show_ml_pipeline_tab():
    st.subheader("ML-пайплайн проекта")

    st.markdown("""
    В этом проекте модель не классифицирует отдельную строку датчиков.
    Объект классификации — **окно временного ряда** длительностью 5 секунд.
    """)

    st.code("""
Исходные временные ряды датчиков
        ↓
Нарезка сигнала на окна по 5 секунд
        ↓
Извлечение температурных и вибрационных признаков
        ↓
Обучение baseline-модели
        ↓
Обучение более сильных моделей
        ↓
Сравнение моделей по метрикам
        ↓
Выбор лучшей модели
        ↓
Применение модели в системе диагностики
""", language="text")

    stages = pd.DataFrame(
        {
            "Этап": [
                "Временные ряды",
                "Оконная обработка",
                "Извлечение признаков",
                "Baseline",
                "Классические модели",
                "Оценка качества",
                "Продуктовый слой"
            ],
            "Что происходит": [
                "Система получает температуру, вибрацию, ток, напряжение, нагрузку и скорость вентилятора",
                "Непрерывный сигнал разбивается на короткие интервалы фиксированной длины",
                "Из каждого окна рассчитываются статистические и частотные признаки",
                "Обучается простая модель для точки отсчёта",
                "Обучаются Случайный лес и Градиентный бустинг",
                "Модели сравниваются по точности, полноте и F1-мере",
                "Предсказание модели переводится в риск, предупреждение и задачу для техника"
            ]
        }
    )

    st.dataframe(stages, use_container_width=True, hide_index=True)


def show_features_tab():
    st.subheader("Признаки, используемые моделью")

    st.markdown("""
    Модель получает не сырые сигналы напрямую, а признаки, рассчитанные по каждому окну временного ряда.
    Это делает задачу похожей на промышленную диагностику по сигналам.
    """)

    feature_groups = pd.DataFrame(
        {
            "Группа признаков": [
                "Температурные признаки",
                "Температурная динамика",
                "Вибрационные признаки во временной области",
                "Частотные признаки вибрации",
                "Электрические и режимные признаки"
            ],
            "Примеры": [
                "средняя температура, максимум, минимум, диапазон",
                "тренд температуры, доля времени выше порога, разность кристалл-корпус",
                "RMS, максимум вибрации, пик-фактор",
                "доминирующая частота, энергия по диапазонам, спектральный центр",
                "ток, напряжение, нагрузка, скорость вентилятора"
            ],
            "Зачем нужны": [
                "Для поиска перегрева",
                "Для выявления деградации охлаждения",
                "Для фиксации общего роста вибрации",
                "Для различения дисбаланса, износа и вибрационных неисправностей",
                "Для учёта режима работы оборудования"
            ]
        }
    )

    st.dataframe(feature_groups, use_container_width=True, hide_index=True)

    features = load_features()
    numeric_features = features.drop(
        columns=[
            "state",
            "global_window_id",
            "experiment_id",
            "window_start",
            "window_end"
        ],
        errors="ignore"
    )

    st.metric("Количество объектов для обучения", len(features))
    st.metric("Количество признаков", len(numeric_features.columns))


def show_data_tab():
    features = load_features()
    features["Состояние"] = features["state"].map(STATE_NAMES)

    st.subheader("Данные после обработки")

    col1, col2, col3 = st.columns(3)

    col1.metric("Окон сигнала", len(features))
    col2.metric("Классов состояния", features["state"].nunique())
    col3.metric("Экспериментов", features["experiment_id"].nunique())

    st.subheader("Распределение состояний")

    counts = features["Состояние"].value_counts().reset_index()
    counts.columns = ["Состояние", "Количество"]

    fig = px.bar(counts, x="Состояние", y="Количество")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Температурные признаки")

    fig = px.box(
        features,
        x="Состояние",
        y="temperature_chip_mean",
        title="Средняя температура кристалла"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Вибрационные признаки")

    fig = px.box(
        features,
        x="Состояние",
        y="vibration_magnitude_rms",
        title="Среднеквадратичный уровень вибрации"
    )
    st.plotly_chart(fig, use_container_width=True)


def show_models_tab():
    st.subheader("Сравнение ML-моделей")

    st.markdown("""
    В проекте используется несколько моделей. Простая модель нужна как baseline,
    а более сильные модели используются для итогового качества.
    """)

    model_table = pd.DataFrame(
        {
            "Модель": [
                "Логистическая регрессия",
                "Случайный лес",
                "Градиентный бустинг"
            ],
            "Роль в проекте": [
                "Baseline: простая точка отсчёта",
                "Основная модель для табличных признаков",
                "Альтернативная сильная модель для сравнения"
            ],
            "Почему используется": [
                "Проверяет, достаточно ли признаки линейно разделяют классы",
                "Хорошо работает с нелинейными зависимостями и важностью признаков",
                "Позволяет сравнить качество с ансамблевой моделью другого типа"
            ]
        }
    )

    st.dataframe(model_table, use_container_width=True, hide_index=True)

    metrics_path = project_root / "reports" / "metrics" / "classic_metrics.csv"

    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path)

        model_names = {
            "random_forest": "Случайный лес",
            "gradient_boosting": "Градиентный бустинг"
        }

        metrics["Модель"] = metrics["model"].map(model_names).fillna(metrics["model"])

        visible_metrics = metrics.rename(
            columns={
                "accuracy": "Точность",
                "precision_macro": "Средняя точность по классам",
                "recall_macro": "Средняя полнота по классам",
                "f1_macro": "F1-мера"
            }
        )

        st.subheader("Метрики качества")

        st.dataframe(
            visible_metrics[
                [
                    "Модель",
                    "Точность",
                    "Средняя точность по классам",
                    "Средняя полнота по классам",
                    "F1-мера"
                ]
            ],
            use_container_width=True,
            hide_index=True
        )

        fig = px.bar(
            visible_metrics,
            x="Модель",
            y="F1-мера",
            title="Сравнение моделей по F1-мере"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Файл с метриками не найден.")


def show_error_analysis_tab():
    st.subheader("Анализ ошибок")

    matrix_path = project_root / "reports" / "metrics" / "best_model_confusion_matrix.csv"
    report_path = project_root / "reports" / "metrics" / "best_model_report.txt"

    if report_path.exists():
        st.subheader("Отчёт классификации")
        st.text(report_path.read_text(encoding="utf-8"))

    if matrix_path.exists():
        st.subheader("Матрица ошибок")

        matrix = pd.read_csv(matrix_path, index_col=0)
        matrix.index = [STATE_NAMES.get(index, index) for index in matrix.index]
        matrix.columns = [STATE_NAMES.get(column, column) for column in matrix.columns]

        fig = px.imshow(matrix, text_auto=True, title="Матрица ошибок")
        st.plotly_chart(fig, use_container_width=True)

    st.warning("""
    Сейчас модель показывает почти идеальные метрики. Это означает, что текущая синтетическая генерация
    слишком сильно разделяет классы. Для конкурсной версии данные нужно сделать сложнее.
    """)

    limitations = pd.DataFrame(
        {
            "Проблема текущей версии": [
                "Классы слишком хорошо разделяются",
                "Мало переходных режимов",
                "Не хватает слабых неисправностей",
                "Нет различий между реальными экземплярами модулей",
                "Нет неизвестных аномалий"
            ],
            "Как исправить": [
                "Сделать пересечение диапазонов температуры и вибрации",
                "Добавить режимы начала перегрева и начала вибрационной неисправности",
                "Добавить слабые дефекты, похожие на норму",
                "Добавить индивидуальные параметры модулей и стендов",
                "Добавить отдельный блок anomaly detection"
            ]
        }
    )

    st.dataframe(limitations, use_container_width=True, hide_index=True)


def show_interpretation_tab():
    st.subheader("Интерпретация модели")

    model_path = project_root / "models" / "best_model.pkl"

    if not model_path.exists():
        st.warning("Модель не найдена.")
        return

    model = joblib.load(model_path)
    features = load_features()

    drop_columns = [
        "state",
        "global_window_id",
        "experiment_id",
        "window_start",
        "window_end"
    ]

    feature_columns = [column for column in features.columns if column not in drop_columns]

    if hasattr(model, "feature_importances_"):
        importance = pd.DataFrame(
            {
                "Признак": feature_columns,
                "Важность": model.feature_importances_
            }
        ).sort_values("Важность", ascending=False).head(25)

        importance["Признак"] = importance["Признак"].map(FEATURE_DISPLAY_NAMES).fillna(importance["Признак"])

        fig = px.bar(
            importance.sort_values("Важность"),
            x="Важность",
            y="Признак",
            orientation="h",
            title="Наиболее важные признаки"
        )

        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(importance, use_container_width=True, hide_index=True)
    else:
        st.info("Текущая модель не поддерживает оценку важности признаков.")

    st.subheader("Инженерный смысл интерпретации")

    interpretation = pd.DataFrame(
        {
            "Признак": [
                "Средняя температура кристалла",
                "Тренд температуры",
                "Разность температур кристалл-корпус",
                "Среднеквадратичный уровень вибрации",
                "Доминирующая частота вибрации",
                "Энергия высоких частот"
            ],
            "Что может означать": [
                "Перегрев силовых компонентов",
                "Начало деградации охлаждения",
                "Проблема теплопередачи от кристалла к корпусу",
                "Общий рост механических колебаний",
                "Дисбаланс или периодическая механическая неисправность",
                "Износ механического узла или импульсные дефекты"
            ]
        }
    )

    st.dataframe(interpretation, use_container_width=True, hide_index=True)


def show_ml_engineer_view():
    show_header("ML-инженер")

    st.markdown("""
    Раздел ML-инженера показывает машинно-обучающую часть проекта:
    данные, признаки, модели, метрики, ошибки и интерпретацию.
    """)

    tabs = st.tabs(
        [
            "ML-пайплайн",
            "Данные",
            "Признаки",
            "Модели",
            "Ошибки",
            "Интерпретация"
        ]
    )

    with tabs[0]:
        show_ml_pipeline_tab()

    with tabs[1]:
        show_data_tab()

    with tabs[2]:
        show_features_tab()

    with tabs[3]:
        show_models_tab()

    with tabs[4]:
        show_error_analysis_tab()

    with tabs[5]:
        show_interpretation_tab()


st.sidebar.title("Доступ к системе")

role = st.sidebar.selectbox(
    "Выберите роль",
    [
        "Оператор",
        "Инженер-техник",
        "ML-инженер"
    ]
)

st.sidebar.markdown("---")

st.sidebar.markdown("""
**Разграничение доступа**

Оператор видит предупреждения и состояние модулей.

Инженер-техник видит план проверки.

ML-инженер видит ML-пайплайн, признаки, модели, ошибки и интерпретацию.
""")

if role == "Оператор":
    show_operator_view()
elif role == "Инженер-техник":
    show_technician_view()
else:
    show_ml_engineer_view()
