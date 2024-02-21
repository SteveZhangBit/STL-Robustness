
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from robustness.analysis.utils import L2Norm, normalize
from tqdm import tqdm
# # Function to normalize the values
# def normalize(value, bounds):
#     lower, upper = bounds
#     return (value - lower) / (upper - lower)

# Load the data
data_path = 'lunar_lander_data.csv'
data = pd.read_csv(data_path)

# Define the development bounds
dev_bounds = [(0.0, 20.0), (0.0, 2.0)]

# Plotting logic
plt.figure(figsize=(10, 8))

# Assuming 'samples' are the rows in the CSV, and based on the provided logic,
# we need to adjust for our context. This is an interpretation to fit the data structure.
for index, row in tqdm(data.iterrows(), total=data.shape[0]):
    # Extract d1, d2 values and consider them as X in the provided logic
    d1, d2 = row['d1'], row['d2']
    
    # Normalize d1 and d2
    normalized_d = normalize([d1, d2], np.asarray(dev_bounds))
    # normalized_d2 = normalize(d2, dev_bounds[1])
    
    # Determine the color and size based on Cost (assuming Cost as Y in the provided logic)
    if row['Cost'] >= 0.0:
        plt.scatter(normalized_d[0], normalized_d[1], c='gray', marker='x', alpha=0.90, s=40)
    else:
        plt.scatter(normalized_d[0], normalized_d[1], c='yellow', marker='x', s=80)

plt.xlabel('Normalized d1')
plt.ylabel('Normalized d2')
plt.title('Scatter Plot with Normalized d1 and d2')
plt.show()
