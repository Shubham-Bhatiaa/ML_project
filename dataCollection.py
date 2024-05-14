import numpy as np
import pandas as pd

wine_dataset = pd.read_csv("winequality-red.csv")

wine_dataset.shape


wine_dataset.head()

wine_dataset.isnull().sum()


