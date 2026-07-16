from pathlib import Path

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
