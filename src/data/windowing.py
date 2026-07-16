from pathlib import Path

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
