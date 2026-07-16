STATE_DISPLAY_NAMES = {
    "normal": "Норма",
    "overheating": "Перегрев",
    "cooling_degradation": "Ухудшение охлаждения",
    "vibration_fault": "Повышенная вибрация",
    "imbalance": "Дисбаланс",
    "bearing_wear": "Износ механического узла",
    "combined_fault": "Комбинированная неисправность"
}

RISK_DISPLAY_NAMES = {
    "low": "Низкий",
    "medium": "Средний",
    "high": "Высокий",
    "critical": "Критический"
}

RISK_ICONS = {
    "low": "🟢",
    "medium": "🟡",
    "high": "🟠",
    "critical": "🔴"
}

ACTION_DISPLAY_NAMES = {
    "none": "Действия не требуются",
    "monitor": "Продолжить наблюдение",
    "check_cooling": "Проверить радиатор, термопрокладку и вентиляционные отверстия",
    "check_mounting": "Проверить крепление модуля и виброизоляцию",
    "check_mechanics": "Проверить механический узел и источник вибрации",
    "stop_test": "Остановить испытание и провести внеплановую проверку"
}


def get_state_name(state):
    return STATE_DISPLAY_NAMES.get(state, state)


def get_risk_name(risk):
    return RISK_DISPLAY_NAMES.get(risk, risk)


def get_risk_icon(risk):
    return RISK_ICONS.get(risk, "⚪")


def get_action_name(action):
    return ACTION_DISPLAY_NAMES.get(action, action)


def format_module_id(experiment_id):
    module_number = int(experiment_id) % 80 + 1
    return f"СВЧ-модуль №{module_number:03d}"


def format_stand_id(experiment_id):
    stand_number = int(experiment_id) % 8 + 1
    return f"Испытательный стенд №{stand_number:02d}"
