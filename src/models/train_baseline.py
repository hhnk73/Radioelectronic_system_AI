from pathlib import Path

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
