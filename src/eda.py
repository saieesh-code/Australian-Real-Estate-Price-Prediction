import matplotlib.pyplot as plt
import seaborn as sns


def plot_saleprice_distribution(df):
    plt.figure(figsize=(10, 5))
    sns.histplot(df["SalePrice"], kde=True)
    plt.title("Sale Price Distribution")
    plt.savefig("outputs/plots/saleprice_distribution.png")
    plt.close()


def plot_correlation_heatmap(df):
    plt.figure(figsize=(15, 10))

    corr = df.corr(numeric_only=True)

    sns.heatmap(corr, cmap="coolwarm")

    plt.title("Correlation Heatmap")

    plt.savefig("outputs/plots/correlation_heatmap.png")

    plt.close()