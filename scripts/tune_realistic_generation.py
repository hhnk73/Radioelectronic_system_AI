from pathlib import Path


path = Path("src/data/generate_signals.py")
text = path.read_text(encoding="utf-8")


old_choose_effective_state = '''def choose_effective_state(state, rng):
    if rng.random() > 0.16:
        return state

    transitions = {
        "normal": ["cooling_degradation", "imbalance"],
        "overheating": ["cooling_degradation", "combined_fault"],
        "cooling_degradation": ["normal", "overheating"],
        "vibration_fault": ["imbalance", "bearing_wear"],
        "imbalance": ["vibration_fault", "normal"],
        "bearing_wear": ["vibration_fault", "combined_fault"],
        "combined_fault": ["overheating", "vibration_fault"]
    }

    return rng.choice(transitions[state])
'''


new_choose_effective_state = '''def choose_effective_state(state, rng):
    if rng.random() > 0.07:
        return state

    transitions = {
        "normal": ["cooling_degradation"],
        "overheating": ["cooling_degradation"],
        "cooling_degradation": ["normal", "overheating"],
        "vibration_fault": ["imbalance"],
        "imbalance": ["vibration_fault"],
        "bearing_wear": ["vibration_fault"],
        "combined_fault": ["overheating", "vibration_fault"]
    }

    return rng.choice(transitions[state])
'''


old_choose_severity = '''def choose_severity(state, rng):
    if state == "normal":
        return rng.uniform(0.0, 0.35)

    if rng.random() < 0.35:
        return rng.uniform(0.25, 0.55)

    if rng.random() < 0.25:
        return rng.uniform(0.55, 0.75)

    return rng.uniform(0.75, 1.0)
'''


new_choose_severity = '''def choose_severity(state, rng):
    if state == "normal":
        return rng.uniform(0.0, 0.25)

    random_value = rng.random()

    if random_value < 0.18:
        return rng.uniform(0.35, 0.55)

    if random_value < 0.50:
        return rng.uniform(0.55, 0.75)

    return rng.uniform(0.75, 1.0)
'''


text = text.replace(old_choose_effective_state, new_choose_effective_state)
text = text.replace(old_choose_severity, new_choose_severity)

path.write_text(text, encoding="utf-8")

print("Realistic data generation tuned successfully")
