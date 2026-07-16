import pandas as pd

from src.product.risk_engine import enrich_with_risk
from src.utils.display_names import format_module_id
from src.utils.display_names import format_stand_id


def create_operator_snapshot(predictions):
    data = enrich_with_risk(predictions)

    data["Модуль"] = data["experiment_id"].apply(format_module_id)
    data["Стенд"] = data["experiment_id"].apply(format_stand_id)

    priority = {
        "critical": 4,
        "high": 3,
        "medium": 2,
        "low": 1
    }

    data["priority"] = data["risk_level"].map(priority)

    snapshot = (
        data.sort_values("priority", ascending=False)
        .groupby(["Модуль", "Стенд"], as_index=False)
        .first()
    )

    result = snapshot[
        [
            "Модуль",
            "Стенд",
            "Состояние",
            "Уровень риска",
            "prediction_confidence",
            "Рекомендация",
            "risk_level",
            "priority"
        ]
    ].copy()

    result = result.rename(columns={"prediction_confidence": "Уверенность модели"})
    result = result.sort_values(["priority", "Уверенность модели"], ascending=False)

    return result


def create_summary_cards(snapshot):
    total_modules = snapshot["Модуль"].nunique()
    critical_count = (snapshot["risk_level"] == "critical").sum()
    high_count = (snapshot["risk_level"] == "high").sum()
    medium_count = (snapshot["risk_level"] == "medium").sum()
    normal_count = (snapshot["risk_level"] == "low").sum()

    return {
        "Всего модулей": total_modules,
        "Критический риск": int(critical_count),
        "Высокий риск": int(high_count),
        "Средний риск": int(medium_count),
        "Норма": int(normal_count)
    }
