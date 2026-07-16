from pathlib import Path

files = {
    "requirements.txt": r'''numpy
pandas
matplotlib
scikit-learn
scipy
pyyaml
joblib
streamlit
plotly
pytest
jupyter
''',

    ".gitignore": r'''.venv/
__pycache__/
*.pyc

data/raw/
data/interim/
data/processed/
data/external/

models/*.pkl
models/*.joblib

reports/figures/
reports/metrics/

.ipynb_checkpoints/
.idea/
.env
.DS_Store
''',

    "configs/data_config.yaml": r'''random_seed: 42

simulation:
  sampling_rate: 100
  duration_seconds: 60
  experiments_per_class: 80

classes:
  - normal
  - overheating
  - cooling_degradation
  - vibration_fault
  - imbalance
  - bearing_wear
  - combined_fault

sensors:
  temperature_chip: true
  temperature_case: true
  vibration_x: true
  vibration_y: true
  vibration_z: true
  current: true
  voltage: true
  load: true
  fan_speed: true

windowing:
  window_size_seconds: 5
  step_size_seconds: 2.5

output:
  raw_data_path: data/raw/sensor_timeseries.csv
  windows_path: data/interim/windows.csv
  features_path: data/processed/features.csv
''',

    "configs/model_config.yaml": r'''random_seed: 42

target_column: state

test_size: 0.2

models:
  baseline:
    name: logistic_regression

  classic:
    - random_forest
    - gradient_boosting

metrics:
  - accuracy
  - precision_macro
  - recall_macro
  - f1_macro

main_metric: f1_macro

output:
  baseline_model_path: models/baseline_model.pkl
  best_model_path: models/best_model.pkl
  label_encoder_path: models/label_encoder.pkl
  baseline_metrics_path: reports/metrics/baseline_metrics.csv
  classic_metrics_path: reports/metrics/classic_metrics.csv
  best_model_report_path: reports/metrics/best_model_report.txt
  best_model_confusion_matrix_path: reports/metrics/best_model_confusion_matrix.csv
''',

    "src/__init__.py": "",
    "src/data/__init__.py": "",
    "src/features/__init__.py": "",
    "src/models/__init__.py": "",
    "src/utils/__init__.py": "",
    "src/visualization/__init__.py": "",

    "src/utils/paths.py": r'''from pathlib import Path


def get_project_root():
    return Path(__file__).resolve().parents[2]


def resolve_project_path(relative_path):
    return get_project_root() / relative_path
''',

    "src/utils/io.py": r'''from pathlib import Path

import joblib
import pandas as pd
import yaml


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def save_dataframe(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, index=False)


def load_dataframe(path):
    return pd.read_csv(path)


def save_joblib(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)


def load_joblib(path):
    return joblib.load(path)


def save_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        file.write(text)
''',

    "src/data/generate_signals.py": r'''from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def create_time_axis(duration_seconds, sampling_rate):
    n_points = duration_seconds * sampling_rate
    return np.arange(n_points) / sampling_rate


def generate_base_signals(time, rng):
    n_points = len(time)

    load = rng.normal(loc=0.55, scale=0.08, size=n_points)
    load = np.clip(load, 0.1, 1.0)

    voltage = rng.normal(loc=27.0, scale=0.4, size=n_points)
    current = 0.12 + 0.12 * load + rng.normal(loc=0.0, scale=0.01, size=n_points)

    fan_speed = rng.normal(loc=3200, scale=180, size=n_points)
    fan_speed = np.clip(fan_speed, 1800, 4200)

    return load, voltage, current, fan_speed


def generate_temperature(time, load, fan_speed, state, rng):
    n_points = len(time)

    ambient_temperature = rng.normal(loc=25.0, scale=0.7, size=n_points)
    heating_from_load = 18 * load
    cooling_effect = (fan_speed - 2500) / 350

    base_chip = ambient_temperature + heating_from_load - cooling_effect
    base_case = ambient_temperature + 0.55 * heating_from_load - 0.7 * cooling_effect

    temperature_chip = base_chip + rng.normal(loc=0.0, scale=0.6, size=n_points)
    temperature_case = base_case + rng.normal(loc=0.0, scale=0.4, size=n_points)

    if state == "normal":
        temperature_chip += rng.normal(loc=0.0, scale=0.8, size=n_points)
        temperature_case += rng.normal(loc=0.0, scale=0.5, size=n_points)

    elif state == "overheating":
        trend = np.linspace(0, 28, n_points)
        temperature_chip += 18 + trend
        temperature_case += 10 + 0.65 * trend

    elif state == "cooling_degradation":
        trend = np.linspace(0, 20, n_points)
        fan_penalty = np.linspace(0, 900, n_points)
        temperature_chip += 7 + trend + fan_penalty / 140
        temperature_case += 5 + 0.75 * trend + fan_penalty / 220

    elif state == "vibration_fault":
        temperature_chip += rng.normal(loc=3.0, scale=1.0, size=n_points)
        temperature_case += rng.normal(loc=2.0, scale=0.8, size=n_points)

    elif state == "imbalance":
        temperature_chip += rng.normal(loc=2.0, scale=1.0, size=n_points)
        temperature_case += rng.normal(loc=1.5, scale=0.7, size=n_points)

    elif state == "bearing_wear":
        temperature_chip += np.linspace(0, 8, n_points)
        temperature_case += np.linspace(0, 5, n_points)

    elif state == "combined_fault":
        trend = np.linspace(0, 26, n_points)
        temperature_chip += 17 + trend
        temperature_case += 10 + 0.7 * trend

    return temperature_chip, temperature_case


def generate_vibration_axis(time, state, rng, axis_shift):
    n_points = len(time)

    signal = rng.normal(loc=0.0, scale=0.03, size=n_points)
    signal += 0.04 * np.sin(2 * np.pi * 3 * time + axis_shift)

    if state == "normal":
        signal += 0.03 * np.sin(2 * np.pi * 12 * time + axis_shift)

    elif state == "overheating":
        signal += 0.04 * np.sin(2 * np.pi * 12 * time + axis_shift)
        signal += rng.normal(loc=0.0, scale=0.02, size=n_points)

    elif state == "cooling_degradation":
        signal += 0.05 * np.sin(2 * np.pi * 10 * time + axis_shift)
        signal += rng.normal(loc=0.0, scale=0.025, size=n_points)

    elif state == "vibration_fault":
        signal += 0.22 * np.sin(2 * np.pi * 26 * time + axis_shift)
        signal += 0.12 * np.sin(2 * np.pi * 48 * time + axis_shift)
        signal += rng.normal(loc=0.0, scale=0.08, size=n_points)

    elif state == "imbalance":
        signal += 0.35 * np.sin(2 * np.pi * 15 * time + axis_shift)
        signal += 0.08 * np.sin(2 * np.pi * 30 * time + axis_shift)
        signal += rng.normal(loc=0.0, scale=0.04, size=n_points)

    elif state == "bearing_wear":
        signal += 0.08 * np.sin(2 * np.pi * 35 * time + axis_shift)
        signal += rng.normal(loc=0.0, scale=0.09, size=n_points)

        impulse_positions = rng.choice(n_points, size=max(3, n_points // 300), replace=False)

        for position in impulse_positions:
            end = min(position + 12, n_points)
            length = end - position
            impulse = np.exp(-np.linspace(0, 3, length)) * rng.uniform(0.35, 0.7)
            signal[position:end] += impulse

    elif state == "combined_fault":
        signal += 0.25 * np.sin(2 * np.pi * 20 * time + axis_shift)
        signal += 0.16 * np.sin(2 * np.pi * 45 * time + axis_shift)
        signal += rng.normal(loc=0.0, scale=0.09, size=n_points)

    return signal


def generate_vibration(time, state, rng):
    vibration_x = generate_vibration_axis(time, state, rng, axis_shift=0.0)
    vibration_y = generate_vibration_axis(time, state, rng, axis_shift=0.8)
    vibration_z = generate_vibration_axis(time, state, rng, axis_shift=1.6)

    return vibration_x, vibration_y, vibration_z


def generate_experiment(experiment_id, state, config, rng):
    sampling_rate = config["simulation"]["sampling_rate"]
    duration_seconds = config["simulation"]["duration_seconds"]

    time = create_time_axis(duration_seconds, sampling_rate)

    load, voltage, current, fan_speed = generate_base_signals(time, rng)

    if state in ["overheating", "combined_fault"]:
        load = np.clip(load + rng.normal(loc=0.28, scale=0.04, size=len(time)), 0.1, 1.0)
        current = current + 0.08 * load

    if state == "cooling_degradation":
        fan_speed = np.clip(fan_speed - np.linspace(0, 1200, len(time)), 1200, 4200)

    temperature_chip, temperature_case = generate_temperature(
        time=time,
        load=load,
        fan_speed=fan_speed,
        state=state,
        rng=rng
    )

    vibration_x, vibration_y, vibration_z = generate_vibration(time, state, rng)

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
''',

    "src/data/windowing.py": r'''from pathlib import Path

import pandas as pd
import yaml


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def create_windows_for_experiment(experiment_data, window_size, step_size):
    windows = []

    experiment_data = experiment_data.sort_values("timestamp").reset_index(drop=True)
    n_rows = len(experiment_data)

    start = 0
    window_id = 0

    while start + window_size <= n_rows:
        end = start + window_size
        window = experiment_data.iloc[start:end].copy()

        window["window_id"] = window_id
        window["window_start"] = window["timestamp"].iloc[0]
        window["window_end"] = window["timestamp"].iloc[-1]

        windows.append(window)

        start += step_size
        window_id += 1

    return windows


def create_windows(data, config):
    sampling_rate = config["simulation"]["sampling_rate"]
    window_size_seconds = config["windowing"]["window_size_seconds"]
    step_size_seconds = config["windowing"]["step_size_seconds"]

    window_size = int(window_size_seconds * sampling_rate)
    step_size = int(step_size_seconds * sampling_rate)

    all_windows = []
    global_window_id = 0

    for experiment_id, experiment_data in data.groupby("experiment_id"):
        experiment_windows = create_windows_for_experiment(
            experiment_data=experiment_data,
            window_size=window_size,
            step_size=step_size
        )

        for window in experiment_windows:
            window["global_window_id"] = global_window_id
            all_windows.append(window)
            global_window_id += 1

    return pd.concat(all_windows, ignore_index=True)


def save_windows(windows, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    windows.to_csv(output_path, index=False)


def main():
    project_root = Path(__file__).resolve().parents[2]
    config_path = project_root / "configs" / "data_config.yaml"

    config = load_config(config_path)

    raw_data_path = project_root / config["output"]["raw_data_path"]
    windows_path = project_root / config["output"]["windows_path"]

    data = pd.read_csv(raw_data_path)

    windows = create_windows(data, config)
    save_windows(windows, windows_path)

    print(f"Saved windows to {windows_path}")
    print(f"Rows: {len(windows)}")
    print(f"Windows: {windows['global_window_id'].nunique()}")
    print(f"Experiments: {windows['experiment_id'].nunique()}")
    print(windows["state"].value_counts())


if __name__ == "__main__":
    main()
''',

    "src/features/time_features.py": r'''import numpy as np


def calculate_trend(values):
    x = np.arange(len(values))

    if len(values) < 2:
        return 0.0

    slope = np.polyfit(x, values, 1)[0]

    return float(slope)


def calculate_basic_stats(values, prefix):
    values = np.asarray(values)

    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_std": float(np.std(values)),
        f"{prefix}_min": float(np.min(values)),
        f"{prefix}_max": float(np.max(values)),
        f"{prefix}_range": float(np.max(values) - np.min(values)),
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_trend": calculate_trend(values)
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


def calculate_vibration_magnitude(window):
    vibration_x = window["vibration_x"].values
    vibration_y = window["vibration_y"].values
    vibration_z = window["vibration_z"].values

    magnitude = np.sqrt(
        vibration_x ** 2
        + vibration_y ** 2
        + vibration_z ** 2
    )

    return magnitude


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
    features["vibration_magnitude_max"] = float(np.max(vibration_magnitude))
    features["vibration_magnitude_rms"] = calculate_rms(vibration_magnitude)
    features["vibration_magnitude_crest_factor"] = calculate_crest_factor(vibration_magnitude)

    temperature_chip = window["temperature_chip"].values
    temperature_case = window["temperature_case"].values

    features["temperature_delta_mean"] = float(np.mean(temperature_chip - temperature_case))
    features["temperature_delta_max"] = float(np.max(temperature_chip - temperature_case))
    features["temperature_chip_above_70_ratio"] = float(np.mean(temperature_chip > 70))
    features["temperature_chip_above_80_ratio"] = float(np.mean(temperature_chip > 80))

    return features
''',

    "src/features/frequency_features.py": r'''import numpy as np


def calculate_fft(values, sampling_rate):
    values = np.asarray(values)
    values = values - np.mean(values)

    fft_values = np.fft.rfft(values)
    frequencies = np.fft.rfftfreq(len(values), d=1 / sampling_rate)

    amplitudes = np.abs(fft_values)

    return frequencies, amplitudes


def calculate_band_energy(frequencies, amplitudes, low, high):
    mask = (frequencies >= low) & (frequencies < high)

    if not np.any(mask):
        return 0.0

    return float(np.sum(amplitudes[mask] ** 2))


def calculate_dominant_frequency(frequencies, amplitudes):
    if len(amplitudes) <= 1:
        return 0.0

    amplitudes_without_dc = amplitudes.copy()
    amplitudes_without_dc[0] = 0

    dominant_index = np.argmax(amplitudes_without_dc)

    return float(frequencies[dominant_index])


def calculate_spectral_centroid(frequencies, amplitudes):
    amplitude_sum = np.sum(amplitudes)

    if amplitude_sum == 0:
        return 0.0

    return float(np.sum(frequencies * amplitudes) / amplitude_sum)


def extract_frequency_features_for_signal(values, sampling_rate, prefix):
    frequencies, amplitudes = calculate_fft(values, sampling_rate)

    features = {
        f"{prefix}_dominant_frequency": calculate_dominant_frequency(frequencies, amplitudes),
        f"{prefix}_spectral_centroid": calculate_spectral_centroid(frequencies, amplitudes),
        f"{prefix}_energy_0_10_hz": calculate_band_energy(frequencies, amplitudes, 0, 10),
        f"{prefix}_energy_10_25_hz": calculate_band_energy(frequencies, amplitudes, 10, 25),
        f"{prefix}_energy_25_50_hz": calculate_band_energy(frequencies, amplitudes, 25, 50),
        f"{prefix}_total_spectral_energy": float(np.sum(amplitudes ** 2))
    }

    total_energy = features[f"{prefix}_total_spectral_energy"]

    if total_energy == 0:
        features[f"{prefix}_high_frequency_energy_ratio"] = 0.0
    else:
        features[f"{prefix}_high_frequency_energy_ratio"] = features[f"{prefix}_energy_25_50_hz"] / total_energy

    return features


def extract_frequency_features(window, sampling_rate):
    features = {}

    vibration_columns = [
        "vibration_x",
        "vibration_y",
        "vibration_z"
    ]

    for column in vibration_columns:
        features.update(
            extract_frequency_features_for_signal(
                values=window[column].values,
                sampling_rate=sampling_rate,
                prefix=column
            )
        )

    vibration_magnitude = np.sqrt(
        window["vibration_x"].values ** 2
        + window["vibration_y"].values ** 2
        + window["vibration_z"].values ** 2
    )

    features.update(
        extract_frequency_features_for_signal(
            values=vibration_magnitude,
            sampling_rate=sampling_rate,
            prefix="vibration_magnitude"
        )
    )

    return features
''',

    "src/features/feature_pipeline.py": r'''from pathlib import Path

import pandas as pd
import yaml

from src.features.frequency_features import extract_frequency_features
from src.features.time_features import extract_time_features


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def extract_features_from_window(window, sampling_rate):
    features = {}

    first_row = window.iloc[0]

    features["global_window_id"] = first_row["global_window_id"]
    features["experiment_id"] = first_row["experiment_id"]
    features["window_start"] = first_row["window_start"]
    features["window_end"] = first_row["window_end"]
    features["state"] = first_row["state"]

    features.update(extract_time_features(window))
    features.update(extract_frequency_features(window, sampling_rate))

    return features


def build_feature_dataset(windows, config):
    sampling_rate = config["simulation"]["sampling_rate"]

    feature_rows = []

    grouped_windows = windows.groupby("global_window_id")

    for index, (_, window) in enumerate(grouped_windows, start=1):
        features = extract_features_from_window(window, sampling_rate)
        feature_rows.append(features)

        if index % 1000 == 0:
            print(f"Processed windows: {index}")

    return pd.DataFrame(feature_rows)


def save_features(features, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_path, index=False)


def main():
    project_root = Path(__file__).resolve().parents[2]
    config_path = project_root / "configs" / "data_config.yaml"

    config = load_config(config_path)

    windows_path = project_root / config["output"]["windows_path"]
    features_path = project_root / config["output"]["features_path"]

    windows = pd.read_csv(windows_path)

    features = build_feature_dataset(windows, config)

    save_features(features, features_path)

    print(f"Saved features to {features_path}")
    print(f"Rows: {len(features)}")
    print(f"Columns: {len(features.columns)}")
    print(features["state"].value_counts())


if __name__ == "__main__":
    main()
''',

    "src/models/splitting.py": r'''import numpy as np


def split_by_experiment(data, test_size, random_seed):
    rng = np.random.default_rng(random_seed)

    train_experiments = []
    test_experiments = []

    for _, class_data in data.groupby("state"):
        experiment_ids = class_data["experiment_id"].unique()
        rng.shuffle(experiment_ids)

        test_count = max(1, int(len(experiment_ids) * test_size))

        test_class_experiments = experiment_ids[:test_count]
        train_class_experiments = experiment_ids[test_count:]

        test_experiments.extend(test_class_experiments)
        train_experiments.extend(train_class_experiments)

    train_data = data[data["experiment_id"].isin(train_experiments)].copy()
    test_data = data[data["experiment_id"].isin(test_experiments)].copy()

    return train_data, test_data
''',

    "src/models/train_baseline.py": r'''from pathlib import Path

import joblib
import pandas as pd
import yaml

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from src.models.splitting import split_by_experiment


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def split_features_and_target(data, target_column):
    drop_columns = [
        "state",
        "global_window_id",
        "experiment_id",
        "window_start",
        "window_end"
    ]

    x = data.drop(columns=drop_columns)
    y = data[target_column]

    return x, y


def create_baseline_model(feature_columns):
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), feature_columns)
        ]
    )

    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )


def save_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        file.write(text)


def main():
    project_root = Path(__file__).resolve().parents[2]

    data_config = load_config(project_root / "configs" / "data_config.yaml")
    model_config = load_config(project_root / "configs" / "model_config.yaml")

    features_path = project_root / data_config["output"]["features_path"]
    model_path = project_root / model_config["output"]["baseline_model_path"]
    label_encoder_path = project_root / model_config["output"]["label_encoder_path"]
    metrics_path = project_root / model_config["output"]["baseline_metrics_path"]
    report_path = project_root / "reports" / "metrics" / "baseline_classification_report.txt"

    data = pd.read_csv(features_path)

    train_data, test_data = split_by_experiment(
        data=data,
        test_size=model_config["test_size"],
        random_seed=model_config["random_seed"]
    )

    x_train, y_train = split_features_and_target(train_data, model_config["target_column"])
    x_test, y_test = split_features_and_target(test_data, model_config["target_column"])

    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    model = create_baseline_model(feature_columns=x_train.columns.tolist())
    model.fit(x_train, y_train_encoded)

    y_pred = model.predict(x_test)

    metrics = {
        "model": "logistic_regression",
        "accuracy": accuracy_score(y_test_encoded, y_pred),
        "precision_macro": precision_score(y_test_encoded, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_test_encoded, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_test_encoded, y_pred, average="macro", zero_division=0)
    }

    report = classification_report(
        y_test_encoded,
        y_pred,
        target_names=label_encoder.classes_,
        zero_division=0
    )

    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_path)
    joblib.dump(label_encoder, label_encoder_path)
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
    save_text(report_path, report)

    print("Baseline model trained")
    print(pd.DataFrame([metrics]))
    print(report)


if __name__ == "__main__":
    main()
''',

    "src/models/train_classic_ml.py": r'''from pathlib import Path

import joblib
import pandas as pd
import yaml

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.preprocessing import LabelEncoder

from src.models.splitting import split_by_experiment


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def split_features_and_target(data, target_column):
    drop_columns = [
        "state",
        "global_window_id",
        "experiment_id",
        "window_start",
        "window_end"
    ]

    x = data.drop(columns=drop_columns)
    y = data[target_column]

    return x, y


def get_models(random_seed):
    return {
        "random_forest": RandomForestClassifier(
            n_estimators=180,
            max_depth=None,
            min_samples_split=4,
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=random_seed
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=160,
            learning_rate=0.06,
            max_depth=3,
            random_state=random_seed
        )
    }


def save_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        file.write(text)


def main():
    project_root = Path(__file__).resolve().parents[2]

    data_config = load_config(project_root / "configs" / "data_config.yaml")
    model_config = load_config(project_root / "configs" / "model_config.yaml")

    features_path = project_root / data_config["output"]["features_path"]
    best_model_path = project_root / model_config["output"]["best_model_path"]
    label_encoder_path = project_root / model_config["output"]["label_encoder_path"]
    metrics_path = project_root / model_config["output"]["classic_metrics_path"]
    report_path = project_root / model_config["output"]["best_model_report_path"]
    matrix_path = project_root / model_config["output"]["best_model_confusion_matrix_path"]

    data = pd.read_csv(features_path)

    train_data, test_data = split_by_experiment(
        data=data,
        test_size=model_config["test_size"],
        random_seed=model_config["random_seed"]
    )

    x_train, y_train = split_features_and_target(train_data, model_config["target_column"])
    x_test, y_test = split_features_and_target(test_data, model_config["target_column"])

    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    models = get_models(model_config["random_seed"])

    results = []
    best_model = None
    best_name = None
    best_score = -1
    best_predictions = None

    for name, model in models.items():
        print(f"Training {name}")
        model.fit(x_train, y_train_encoded)

        y_pred = model.predict(x_test)

        metrics = {
            "model": name,
            "accuracy": accuracy_score(y_test_encoded, y_pred),
            "precision_macro": precision_score(y_test_encoded, y_pred, average="macro", zero_division=0),
            "recall_macro": recall_score(y_test_encoded, y_pred, average="macro", zero_division=0),
            "f1_macro": f1_score(y_test_encoded, y_pred, average="macro", zero_division=0)
        }

        results.append(metrics)

        if metrics["f1_macro"] > best_score:
            best_score = metrics["f1_macro"]
            best_model = model
            best_name = name
            best_predictions = y_pred

    metrics_df = pd.DataFrame(results).sort_values("f1_macro", ascending=False)

    best_model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    matrix_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(best_model, best_model_path)
    joblib.dump(label_encoder, label_encoder_path)

    metrics_df.to_csv(metrics_path, index=False)

    report = classification_report(
        y_test_encoded,
        best_predictions,
        target_names=label_encoder.classes_,
        zero_division=0
    )

    matrix = confusion_matrix(y_test_encoded, best_predictions)

    pd.DataFrame(
        matrix,
        index=label_encoder.classes_,
        columns=label_encoder.classes_
    ).to_csv(matrix_path)

    save_text(report_path, f"Best model: {best_name}\n\n{report}")

    print("Classic ML training finished")
    print(metrics_df)
    print(f"Best model: {best_name}")
    print(report)


if __name__ == "__main__":
    main()
''',

    "src/models/evaluate.py": r'''from pathlib import Path

import joblib
import pandas as pd
import yaml

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from src.models.splitting import split_by_experiment


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def split_features_and_target(data, target_column):
    drop_columns = [
        "state",
        "global_window_id",
        "experiment_id",
        "window_start",
        "window_end"
    ]

    x = data.drop(columns=drop_columns)
    y = data[target_column]

    return x, y


def save_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        file.write(text)


def main():
    project_root = Path(__file__).resolve().parents[2]

    data_config = load_config(project_root / "configs" / "data_config.yaml")
    model_config = load_config(project_root / "configs" / "model_config.yaml")

    data = pd.read_csv(project_root / data_config["output"]["features_path"])

    _, test_data = split_by_experiment(
        data=data,
        test_size=model_config["test_size"],
        random_seed=model_config["random_seed"]
    )

    x_test, y_test = split_features_and_target(test_data, model_config["target_column"])

    model = joblib.load(project_root / model_config["output"]["best_model_path"])
    label_encoder = joblib.load(project_root / model_config["output"]["label_encoder_path"])

    y_test_encoded = label_encoder.transform(y_test)
    y_pred = model.predict(x_test)

    report = classification_report(
        y_test_encoded,
        y_pred,
        target_names=label_encoder.classes_,
        zero_division=0
    )

    matrix = confusion_matrix(y_test_encoded, y_pred)

    report_path = project_root / model_config["output"]["best_model_report_path"]
    matrix_path = project_root / model_config["output"]["best_model_confusion_matrix_path"]

    save_text(report_path, report)

    pd.DataFrame(
        matrix,
        index=label_encoder.classes_,
        columns=label_encoder.classes_
    ).to_csv(matrix_path)

    print(report)


if __name__ == "__main__":
    main()
''',

    "src/models/predict.py": r'''from pathlib import Path

import joblib
import pandas as pd
import yaml


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def prepare_features(data):
    drop_columns = [
        "state",
        "global_window_id",
        "experiment_id",
        "window_start",
        "window_end"
    ]

    existing_drop_columns = [column for column in drop_columns if column in data.columns]

    return data.drop(columns=existing_drop_columns)


def predict_from_features(features):
    project_root = Path(__file__).resolve().parents[2]

    model_config = load_config(project_root / "configs" / "model_config.yaml")

    model = joblib.load(project_root / model_config["output"]["best_model_path"])
    label_encoder = joblib.load(project_root / model_config["output"]["label_encoder_path"])

    x = prepare_features(features)

    predicted_encoded = model.predict(x)
    predicted_labels = label_encoder.inverse_transform(predicted_encoded)

    result = features.copy()
    result["predicted_state"] = predicted_labels

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(x)
        result["prediction_confidence"] = probabilities.max(axis=1)

    return result


def main():
    project_root = Path(__file__).resolve().parents[2]

    data_config = load_config(project_root / "configs" / "data_config.yaml")

    features_path = project_root / data_config["output"]["features_path"]
    features = pd.read_csv(features_path).head(10)

    predictions = predict_from_features(features)

    print(predictions[["state", "predicted_state"]].head(10))


if __name__ == "__main__":
    main()
''',

    "src/data/make_dataset.py": r'''from src.data.generate_signals import main as generate_signals_main
from src.data.windowing import main as windowing_main
from src.features.feature_pipeline import main as feature_pipeline_main


def main():
    generate_signals_main()
    windowing_main()
    feature_pipeline_main()


if __name__ == "__main__":
    main()
''',

    "run_pipeline.py": r'''import subprocess
import sys


def run_module(module_name):
    print()
    print(f"Running {module_name}")
    result = subprocess.run(
        [sys.executable, "-m", module_name],
        check=True
    )
    return result


def main():
    modules = [
        "src.data.generate_signals",
        "src.data.windowing",
        "src.features.feature_pipeline",
        "src.models.train_baseline",
        "src.models.train_classic_ml",
        "src.models.evaluate"
    ]

    for module_name in modules:
        run_module(module_name)

    print()
    print("Pipeline finished successfully")


if __name__ == "__main__":
    main()
'''
}


def main():
    project_root = Path.cwd()

    for relative_path, content in files.items():
        path = project_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    print("ML-core files written successfully")
    print(f"Project root: {project_root}")


if __name__ == "__main__":
    main()
