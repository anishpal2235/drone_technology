import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
data = pd.read_excel("Data.xlsx")
variables = ["D", "P", "J", "N", "CT","CP"]
correlation_matrix = data[variables].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", square=True)
plt.title("Correlation Matrix")
plt.show()
