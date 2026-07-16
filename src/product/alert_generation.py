from src.product.risk_engine import enrich_with_risk
from src.utils.display_names import format_module_id
from src.utils.display_names import format_stand_id


def generate_alerts(predictions):
    data = enrich_with_risk(predictions)

    data["Модуль"] = data["experiment_id"].apply(format_module_id)
    data["Стенд"] = data["experiment_id"].apply(format_stand_id)

    alerts = data[data["risk_level"].isin(["high", "critical"])].copy()

    if alerts.empty:
        return alerts

    alerts = (
        alerts.sort_values("prediction_confidence", ascending=False)
        .groupby(["Модуль", "Стенд"], as_index=False)
        .first()
    )

    alerts["Предупреждение"] = (
        alerts["Модуль"]
        + " — "
        + alerts["Состояние"]
        + ". "
        + alerts["Уровень риска"]
    )

    return alerts[
        [
            "Модуль",
            "Стенд",
            "Состояние",
            "Уровень риска",
            "prediction_confidence",
            "Предупреждение",
            "Рекомендация",
            "risk_level"
        ]
    ].rename(columns={"prediction_confidence": "Уверенность модели"})
