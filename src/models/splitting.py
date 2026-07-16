import numpy as np


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
