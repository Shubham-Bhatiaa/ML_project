import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


wine_dataset = pd.read_csv("winequality-red.csv")

wine_dataset.describe()

sns.catplot(x="quality", data=wine_dataset, kind="count")

plot = plt.figure(figsize=(5, 5))
sns.barplot(x="quality", y="volatile acidity", data=wine_dataset)

plot = plt.figure(figsize=(5, 5))
sns.barplot(x="quality", y="citric acid", data=wine_dataset)

correlation = wine_dataset.corr()

plt.figure(figsize=(10, 10))
sns.heatmap(
    correlation,
    cbar=True,
    square=True,
    fmt=".1f",
    annot=True,
    annot_kws={"size": 8},
    cmap="Greens",
)
