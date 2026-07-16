from pathlib import Path

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
