from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def create_time_axis(duration_seconds, sampling_rate):
    n_points = duration_seconds * sampling_rate
    return np.arange(n_points) / sampling_rate


def create_module_profile(rng):
    return {
        "thermal_offset": rng.normal(0.0, 3.5),
        "case_offset": rng.normal(0.0, 2.0),
        "cooling_efficiency": rng.normal(1.0, 0.18),
        "vibration_offset": rng.normal(1.0, 0.22),
        "sensor_noise": rng.uniform(0.8, 1.6),
        "fan_offset": rng.normal(0.0, 220.0),
        "current_offset": rng.normal(0.0, 0.015),
        "resonance_shift": rng.normal(0.0, 2.5)
    }


def smooth_random_signal(n_points, rng, base, scale, clip_min, clip_max):
    control_points = max(8, n_points // 500)
    values = rng.normal(base, scale, size=control_points)
    x_old = np.linspace(0, n_points - 1, control_points)
    x_new = np.arange(n_points)
    signal = np.interp(x_new, x_old, values)
    signal += rng.normal(0.0, scale * 0.25, size=n_points)
    return np.clip(signal, clip_min, clip_max)


def add_sensor_artifacts(signal, rng, noise_scale):
    signal = signal.copy()
    n_points = len(signal)

    signal += rng.normal(0.0, noise_scale, size=n_points)

    if rng.random() < 0.25:
        drift = np.linspace(0, rng.normal(0.0, noise_scale * 4), n_points)
        signal += drift

    if rng.random() < 0.20:
        spike_count = rng.integers(2, 8)
        positions = rng.choice(n_points, size=spike_count, replace=False)
        signal[positions] += rng.normal(0.0, noise_scale * 8, size=spike_count)

    if rng.random() < 0.12:
        dropout_start = rng.integers(0, max(1, n_points - 100))
        dropout_len = rng.integers(20, 120)
        dropout_end = min(n_points, dropout_start + dropout_len)
        signal[dropout_start:dropout_end] = signal[max(0, dropout_start - 1)]

    return signal


def generate_base_signals(time, state, severity, profile, rng):
    n_points = len(time)

    if state in ["overheating", "combined_fault"]:
        base_load = rng.uniform(0.62, 0.88)
    elif state in ["cooling_degradation", "bearing_wear"]:
        base_load = rng.uniform(0.50, 0.78)
    else:
        base_load = rng.uniform(0.42, 0.75)

    load = smooth_random_signal(
        n_points=n_points,
        rng=rng,
        base=base_load,
        scale=0.09,
        clip_min=0.15,
        clip_max=1.0
    )

    if rng.random() < 0.35:
        event_start = rng.integers(0, n_points // 2)
        event_end = min(n_points, event_start + rng.integers(n_points // 8, n_points // 3))
        load[event_start:event_end] += rng.uniform(0.05, 0.18)

    load = np.clip(load, 0.1, 1.0)

    voltage = rng.normal(27.0, 0.35, size=n_points)
    voltage += 0.25 * np.sin(2 * np.pi * rng.uniform(0.05, 0.25) * time)
    voltage = add_sensor_artifacts(voltage, rng, noise_scale=0.05)

    current = 0.11 + 0.13 * load + profile["current_offset"]
    current += rng.normal(0.0, 0.012 * profile["sensor_noise"], size=n_points)

    fan_speed = smooth_random_signal(
        n_points=n_points,
        rng=rng,
        base=3250 + profile["fan_offset"],
        scale=210,
        clip_min=1700,
        clip_max=4300
    )

    if state == "cooling_degradation":
        degradation = np.linspace(0, 850 * severity, n_points)
        fan_speed -= degradation

    if state in ["overheating", "combined_fault"] and rng.random() < 0.45:
        fan_speed -= rng.uniform(150, 450) * severity

    fan_speed = np.clip(fan_speed, 1200, 4300)

    return load, voltage, current, fan_speed


def generate_temperature(time, load, fan_speed, state, severity, profile, rng):
    n_points = len(time)

    ambient = smooth_random_signal(
        n_points=n_points,
        rng=rng,
        base=rng.uniform(23.0, 31.0),
        scale=0.8,
        clip_min=18.0,
        clip_max=36.0
    )

    load_heating = (16.0 + rng.normal(0.0, 2.0)) * load
    cooling = (fan_speed - 2400) / 390 * profile["cooling_efficiency"]

    chip = ambient + load_heating - cooling + profile["thermal_offset"]
    case = ambient + 0.58 * load_heating - 0.65 * cooling + profile["case_offset"]

    if state == "normal":
        if rng.random() < 0.18:
            chip += np.linspace(0, rng.uniform(3, 8), n_points)
            case += np.linspace(0, rng.uniform(2, 5), n_points)

    elif state == "overheating":
        trend = np.linspace(0, rng.uniform(8, 24) * severity, n_points)
        chip += rng.uniform(5, 15) * severity + trend
        case += rng.uniform(3, 10) * severity + 0.65 * trend

    elif state == "cooling_degradation":
        trend = np.linspace(0, rng.uniform(7, 20) * severity, n_points)
        chip += rng.uniform(3, 10) * severity + trend
        case += rng.uniform(3, 9) * severity + 0.85 * trend

    elif state == "vibration_fault":
        chip += rng.uniform(0, 5) * severity
        case += rng.uniform(0, 4) * severity

    elif state == "imbalance":
        chip += rng.uniform(0, 4) * severity
        case += rng.uniform(0, 3) * severity

    elif state == "bearing_wear":
        trend = np.linspace(0, rng.uniform(1, 8) * severity, n_points)
        chip += trend + rng.uniform(0, 4) * severity
        case += 0.7 * trend + rng.uniform(0, 3) * severity

    elif state == "combined_fault":
        trend = np.linspace(0, rng.uniform(8, 22) * severity, n_points)
        chip += rng.uniform(7, 17) * severity + trend
        case += rng.uniform(4, 12) * severity + 0.72 * trend

    chip = add_sensor_artifacts(chip, rng, noise_scale=0.65 * profile["sensor_noise"])
    case = add_sensor_artifacts(case, rng, noise_scale=0.45 * profile["sensor_noise"])

    return chip, case


def add_impulses(signal, rng, severity):
    signal = signal.copy()
    n_points = len(signal)

    impulse_count = rng.integers(3, 18)
    impulse_count = max(1, int(impulse_count * severity))

    positions = rng.choice(n_points, size=impulse_count, replace=False)

    for position in positions:
        length = rng.integers(6, 25)
        end = min(position + length, n_points)
        actual_length = end - position
        amplitude = rng.uniform(0.12, 0.55) * severity
        impulse = np.exp(-np.linspace(0, 4, actual_length)) * amplitude
        signal[position:end] += impulse

    return signal


def generate_vibration_axis(time, state, severity, profile, rng, axis_shift):
    n_points = len(time)

    resonance_shift = profile["resonance_shift"]
    vibration_scale = profile["vibration_offset"]

    signal = rng.normal(0.0, 0.035 * profile["sensor_noise"], size=n_points)
    signal += 0.035 * np.sin(2 * np.pi * (3 + resonance_shift * 0.05) * time + axis_shift)

    normal_background = 0.035 * np.sin(2 * np.pi * (12 + resonance_shift) * time + axis_shift)
    signal += normal_background * rng.uniform(0.7, 1.4)

    if state == "normal":
        if rng.random() < 0.18:
            signal += rng.uniform(0.03, 0.10) * np.sin(2 * np.pi * rng.uniform(15, 30) * time + axis_shift)

    elif state == "overheating":
        signal += rng.uniform(0.02, 0.08) * np.sin(2 * np.pi * rng.uniform(10, 18) * time + axis_shift)

    elif state == "cooling_degradation":
        signal += rng.uniform(0.03, 0.10) * np.sin(2 * np.pi * rng.uniform(9, 18) * time + axis_shift)

    elif state == "vibration_fault":
        signal += rng.uniform(0.08, 0.24) * severity * np.sin(2 * np.pi * rng.uniform(22, 32) * time + axis_shift)
        signal += rng.uniform(0.04, 0.14) * severity * np.sin(2 * np.pi * rng.uniform(35, 49) * time + axis_shift)
        signal += rng.normal(0.0, 0.05 * severity, size=n_points)

    elif state == "imbalance":
        signal += rng.uniform(0.12, 0.32) * severity * np.sin(2 * np.pi * rng.uniform(13, 19) * time + axis_shift)
        signal += rng.uniform(0.02, 0.10) * severity * np.sin(2 * np.pi * rng.uniform(26, 38) * time + axis_shift)

    elif state == "bearing_wear":
        signal += rng.uniform(0.04, 0.14) * severity * np.sin(2 * np.pi * rng.uniform(30, 42) * time + axis_shift)
        signal += rng.normal(0.0, 0.055 * severity, size=n_points)
        signal = add_impulses(signal, rng, severity)

    elif state == "combined_fault":
        signal += rng.uniform(0.08, 0.25) * severity * np.sin(2 * np.pi * rng.uniform(18, 26) * time + axis_shift)
        signal += rng.uniform(0.05, 0.18) * severity * np.sin(2 * np.pi * rng.uniform(35, 49) * time + axis_shift)
        signal = add_impulses(signal, rng, severity * 0.7)

    signal *= vibration_scale
    signal = add_sensor_artifacts(signal, rng, noise_scale=0.018 * profile["sensor_noise"])

    return signal


def generate_vibration(time, state, severity, profile, rng):
    vibration_x = generate_vibration_axis(time, state, severity, profile, rng, axis_shift=0.0)
    vibration_y = generate_vibration_axis(time, state, severity, profile, rng, axis_shift=0.8)
    vibration_z = generate_vibration_axis(time, state, severity, profile, rng, axis_shift=1.6)

    return vibration_x, vibration_y, vibration_z


def choose_effective_state(state, rng):
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


def choose_severity(state, rng):
    if state == "normal":
        return rng.uniform(0.0, 0.25)

    random_value = rng.random()

    if random_value < 0.18:
        return rng.uniform(0.35, 0.55)

    if random_value < 0.50:
        return rng.uniform(0.55, 0.75)

    return rng.uniform(0.75, 1.0)


def generate_experiment(experiment_id, state, config, rng):
    sampling_rate = config["simulation"]["sampling_rate"]
    duration_seconds = config["simulation"]["duration_seconds"]

    time = create_time_axis(duration_seconds, sampling_rate)

    profile = create_module_profile(rng)
    severity = choose_severity(state, rng)
    effective_state = choose_effective_state(state, rng)

    load, voltage, current, fan_speed = generate_base_signals(
        time=time,
        state=effective_state,
        severity=severity,
        profile=profile,
        rng=rng
    )

    temperature_chip, temperature_case = generate_temperature(
        time=time,
        load=load,
        fan_speed=fan_speed,
        state=effective_state,
        severity=severity,
        profile=profile,
        rng=rng
    )

    vibration_x, vibration_y, vibration_z = generate_vibration(
        time=time,
        state=effective_state,
        severity=severity,
        profile=profile,
        rng=rng
    )

    return pd.DataFrame({
        "experiment_id": experiment_id,
        "timestamp": time,
        "temperature_chip": temperature_chip,
        "temperature_case": temperature_case,
        "vibration_x": vibration_x,
        "vibration_y": vibration_y,
        "vibration_z": vibration_z,
        "current": current,
        "voltage": voltage,
        "load": load,
        "fan_speed": fan_speed,
        "state": state
    })


def generate_dataset(config):
    rng = np.random.default_rng(config["random_seed"])
    classes = config["classes"]
    experiments_per_class = config["simulation"]["experiments_per_class"]

    experiments = []
    experiment_id = 0

    for state in classes:
        for _ in range(experiments_per_class):
            experiment = generate_experiment(
                experiment_id=experiment_id,
                state=state,
                config=config,
                rng=rng
            )
            experiments.append(experiment)
            experiment_id += 1

    return pd.concat(experiments, ignore_index=True)


def save_dataset(data, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=False)


def main():
    project_root = Path(__file__).resolve().parents[2]
    config_path = project_root / "configs" / "data_config.yaml"

    config = load_config(config_path)
    data = generate_dataset(config)

    output_path = project_root / config["output"]["raw_data_path"]
    save_dataset(data, output_path)

    print(f"Saved raw time series to {output_path}")
    print(f"Rows: {len(data)}")
    print(f"Experiments: {data['experiment_id'].nunique()}")
    print(data["state"].value_counts())


if __name__ == "__main__":
    main()
