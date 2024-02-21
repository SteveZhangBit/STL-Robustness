
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data
data_path = 'lunar_lander_data.csv'
data = pd.read_csv(data_path)

# Preparing the data for heatmap
# Since d1 and d2 are continuous, we need to discretize them for the heatmap
data['d1_bins'] = pd.cut(data['d1'], bins=50, labels=False)
data['d2_bins'] = pd.cut(data['d2'], bins=50, labels=False)

# Create a pivot table for the heatmap
pivot_table = data.pivot_table(index='d1_bins', columns='d2_bins', values='Cost', aggfunc=np.mean)

# Plotting the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(pivot_table, cmap='viridis')
plt.title('Heatmap of Cost by d1 and d2')
plt.xlabel('d2')
plt.ylabel('d1')
plt.show()
