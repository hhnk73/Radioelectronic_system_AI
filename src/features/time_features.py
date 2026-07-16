import numpy as np


def calculate_trend(values):
    x = np.arange(len(values))

    if len(values) < 2:
        return 0.0

    return float(np.polyfit(x, values, 1)[0])


def calculate_acceleration(values):
    values = np.asarray(values)

    if len(values) < 3:
        return 0.0

    first_diff = np.diff(values)
    second_diff = np.diff(first_diff)

    return float(np.mean(second_diff))


def calculate_basic_stats(values, prefix):
    values = np.asarray(values)

    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_std": float(np.std(values)),
        f"{prefix}_min": float(np.min(values)),
        f"{prefix}_max": float(np.max(values)),
        f"{prefix}_range": float(np.max(values) - np.min(values)),
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_q25": float(np.quantile(values, 0.25)),
        f"{prefix}_q75": float(np.quantile(values, 0.75)),
        f"{prefix}_iqr": float(np.quantile(values, 0.75) - np.quantile(values, 0.25)),
        f"{prefix}_trend": calculate_trend(values),
        f"{prefix}_acceleration": calculate_acceleration(values)
    }


def calculate_rms(values):
    values = np.asarray(values)

    return float(np.sqrt(np.mean(values ** 2)))


def calculate_crest_factor(values):
    values = np.asarray(values)

    rms = calculate_rms(values)

    if rms == 0:
        return 0.0

    return float(np.max(np.abs(values)) / rms)


def calculate_zero_crossing_rate(values):
    values = np.asarray(values)

    if len(values) < 2:
        return 0.0

    centered = values - np.mean(values)
    crossings = np.where(np.diff(np.signbit(centered)))[0]

    return float(len(crossings) / len(values))


def calculate_vibration_magnitude(window):
    vibration_x = window["vibration_x"].values
    vibration_y = window["vibration_y"].values
    vibration_z = window["vibration_z"].values

    return np.sqrt(
        vibration_x ** 2
        + vibration_y ** 2
        + vibration_z ** 2
    )


def safe_ratio(numerator, denominator):
    denominator = np.asarray(denominator)

    return np.where(np.abs(denominator) < 1e-6, 0.0, numerator / denominator)


def extract_time_features(window):
    features = {}

    sensor_columns = [
        "temperature_chip",
        "temperature_case",
        "current",
        "voltage",
        "load",
        "fan_speed",
        "vibration_x",
        "vibration_y",
        "vibration_z"
    ]

    for column in sensor_columns:
        features.update(
            calculate_basic_stats(
                values=window[column].values,
                prefix=column
            )
        )

    vibration_magnitude = calculate_vibration_magnitude(window)

    features["vibration_magnitude_mean"] = float(np.mean(vibration_magnitude))
    features["vibration_magnitude_std"] = float(np.std(vibration_magnitude))
    features["vibration_magnitude_min"] = float(np.min(vibration_magnitude))
    features["vibration_magnitude_max"] = float(np.max(vibration_magnitude))
    features["vibration_magnitude_range"] = float(np.max(vibration_magnitude) - np.min(vibration_magnitude))
    features["vibration_magnitude_rms"] = calculate_rms(vibration_magnitude)
    features["vibration_magnitude_crest_factor"] = calculate_crest_factor(vibration_magnitude)
    features["vibration_magnitude_zero_crossing_rate"] = calculate_zero_crossing_rate(vibration_magnitude)
    features["vibration_magnitude_trend"] = calculate_trend(vibration_magnitude)

    temperature_chip = window["temperature_chip"].values
    temperature_case = window["temperature_case"].values
    load = window["load"].values
    fan_speed = window["fan_speed"].values
    current = window["current"].values

    temperature_delta = temperature_chip - temperature_case

    features["temperature_delta_mean"] = float(np.mean(temperature_delta))
    features["temperature_delta_max"] = float(np.max(temperature_delta))
    features["temperature_delta_trend"] = calculate_trend(temperature_delta)
    features["temperature_chip_above_60_ratio"] = float(np.mean(temperature_chip > 60))
    features["temperature_chip_above_70_ratio"] = float(np.mean(temperature_chip > 70))
    features["temperature_chip_above_80_ratio"] = float(np.mean(temperature_chip > 80))

    temp_to_load = safe_ratio(temperature_chip, load)
    temp_to_fan = safe_ratio(temperature_chip, fan_speed)
    current_to_load = safe_ratio(current, load)

    features["temperature_to_load_mean"] = float(np.mean(temp_to_load))
    features["temperature_to_load_trend"] = calculate_trend(temp_to_load)
    features["temperature_to_fan_mean"] = float(np.mean(temp_to_fan))
    features["current_to_load_mean"] = float(np.mean(current_to_load))

    features["thermal_stress_index"] = float(
        np.mean(temperature_chip)
        + 0.4 * np.max(temperature_chip)
        + 10 * calculate_trend(temperature_chip)
        - 0.002 * np.mean(fan_speed)
    )

    features["vibration_stress_index"] = float(
        calculate_rms(vibration_magnitude)
        + 0.5 * np.max(vibration_magnitude)
        + 0.2 * calculate_crest_factor(vibration_magnitude)
    )

    features["combined_stress_index"] = float(
        features["thermal_stress_index"] * 0.03
        + features["vibration_stress_index"] * 4.0
    )

    return features
