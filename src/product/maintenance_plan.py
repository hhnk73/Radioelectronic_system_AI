from src.product.alert_generation import generate_alerts


def create_maintenance_plan(predictions):
    alerts = generate_alerts(predictions)

    if alerts.empty:
        return alerts

    priority_order = {
        "critical": 1,
        "high": 2
    }

    plan = alerts.copy()
    plan["Приоритет"] = plan["risk_level"].map(priority_order)
    plan["Задача для техника"] = plan["Рекомендация"]

    plan = plan.sort_values(["Приоритет", "Уверенность модели"], ascending=[True, False])

    return plan[
        [
            "Приоритет",
            "Модуль",
            "Стенд",
            "Состояние",
            "Уровень риска",
            "Задача для техника",
            "Уверенность модели"
        ]
    ]
