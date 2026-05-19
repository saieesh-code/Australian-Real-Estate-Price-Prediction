from src.preprocessing import (
    load_data,
    handle_missing_values,
    remove_outliers
)

from src.feature_engineering import create_features

from src.eda import (
    plot_saleprice_distribution,
    plot_correlation_heatmap
)

from src.train_model import train_models


TRAIN_PATH = "data/raw/train.csv"
TEST_PATH = "data/raw/test.csv"


def main():

    print("Loading datasets...")

    train_df, test_df = load_data(
        TRAIN_PATH,
        TEST_PATH
    )

    print("Handling missing values...")

    train_df = handle_missing_values(train_df)
    test_df = handle_missing_values(test_df)

    print("Removing outliers...")

    train_df = remove_outliers(train_df)

    print("Creating features...")

    train_df = create_features(train_df)
    test_df = create_features(test_df)

    print("Generating EDA plots...")

    plot_saleprice_distribution(train_df)
    plot_correlation_heatmap(train_df)

    print("Training models...")

    train_models(train_df)

    print("Project completed successfully.")


if __name__ == "__main__":
    main()