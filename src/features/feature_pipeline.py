from pathlib import Path

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
