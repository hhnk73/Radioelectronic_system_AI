import subprocess
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
