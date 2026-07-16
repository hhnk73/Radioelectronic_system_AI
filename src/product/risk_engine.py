from src.utils.display_names import get_action_name
from src.utils.display_names import get_risk_icon
from src.utils.display_names import get_risk_name
from src.utils.display_names import get_state_name


def get_risk_level(state, confidence):
    if state == "normal":
        return "low"

    if state in ["combined_fault", "overheating"] and confidence >= 0.75:
        return "critical"

    if state in ["cooling_degradation", "bearing_wear", "vibration_fault"] and confidence >= 0.75:
        return "high"

    if state in ["imbalance", "vibration_fault", "cooling_degradation"]:
        return "medium"

    return "medium"


def get_recommended_action(state, risk_level):
    if state == "normal":
        return "none"

    if risk_level == "critical":
        return "stop_test"

    if state in ["overheating", "cooling_degradation"]:
        return "check_cooling"

    if state in ["vibration_fault", "imbalance"]:
        return "check_mounting"

    if state == "bearing_wear":
        return "check_mechanics"

    if state == "combined_fault":
        return "stop_test"

    return "monitor"


def build_diagnostic_message(state, risk_level, confidence):
    state_name = get_state_name(state)
    risk_name = get_risk_name(risk_level)
    icon = get_risk_icon(risk_level)

    return f"{icon} Состояние: {state_name}. Уровень риска: {risk_name}. Уверенность модели: {confidence:.2f}."


def enrich_with_risk(data):
    data = data.copy()

    if "prediction_confidence" not in data.columns:
        data["prediction_confidence"] = 1.0

    data["risk_level"] = [
        get_risk_level(state, confidence)
        for state, confidence in zip(data["predicted_state"], data["prediction_confidence"])
    ]

    data["recommended_action"] = [
        get_recommended_action(state, risk)
        for state, risk in zip(data["predicted_state"], data["risk_level"])
    ]

    data["Состояние"] = data["predicted_state"].apply(get_state_name)
    data["Уровень риска"] = data["risk_level"].apply(lambda risk: f"{get_risk_icon(risk)} {get_risk_name(risk)}")
    data["Рекомендация"] = data["recommended_action"].apply(get_action_name)

    return data
