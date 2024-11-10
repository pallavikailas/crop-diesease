import seaborn as sns
import matplotlib.pyplot as plt

def plot_correlation_matrix(df):
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
    plt.show()

def plot_scatter(df, x_col, y_col):
    sns.scatterplot(data=df, x=x_col, y=y_col)
    plt.show()
