import pandas as pd
import matplotlib.pyplot as plt
import seaborn

# Load the financial phrasebank dataset into a pandas dataframe.
dataframe = pd.read_csv("financial_phrasebank.csv")

# Plot the label distributions from the financial phrasebank dataset.
plt.figure(figsize = (8, 6))
seaborn.histplot(x = "sentiment", data = dataframe)
plt.savefig("fpb_distributions.png")

# Load the validation dataset into a pandas dataframe.
dataframe = pd.read_csv("yahoo_validation.csv")

# Plot the label distributions from the validations dataset.
plt.figure(figsize = (8, 6))
seaborn.histplot(x = "sentiment", data = dataframe)
plt.savefig("valid_distributions.png")
