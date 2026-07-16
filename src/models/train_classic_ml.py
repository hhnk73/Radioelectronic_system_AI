from pathlib import Path

import joblib
import pandas as pd
import yaml

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
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
            n_estimators=350,
            max_depth=None,
            min_samples_split=3,
            min_samples_leaf=1,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=random_seed
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features="sqrt",
            class_weight="balanced",
            n_jobs=-1,
            random_state=random_seed
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=240,
            learning_rate=0.06,
            max_leaf_nodes=31,
            l2_regularization=0.05,
            random_state=random_seed
        )
    }


def save_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        file.write(text)


def calculate_metrics(model_name, y_true, y_pred):
    return {
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0)
    }


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

        metrics = calculate_metrics(name, y_test_encoded, y_pred)
        results.append(metrics)

        print(pd.DataFrame([metrics]))

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
