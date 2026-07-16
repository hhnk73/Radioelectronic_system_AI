from src.data.generate_signals import main as generate_signals_main
from src.data.windowing import main as windowing_main
from src.features.feature_pipeline import main as feature_pipeline_main


def main():
    generate_signals_main()
    windowing_main()
    feature_pipeline_main()


if __name__ == "__main__":
    main()
