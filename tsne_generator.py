import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# Load the dataset (replace 'your_dataset.csv' with the actual file name)
data = pd.read_csv('traces/processed_data.csv')

# Separate input features (X) and output variable (y)
X = data.drop(['Rob', 'pred'], axis=1)  # Features
y = data['Rob']  # Target variable

# Standardize the input features (optional but often recommended for t-SNE)
scaler = StandardScaler()
X_standardized = scaler.fit_transform(X)

# Apply t-SNE for visualization
tsne = TSNE(n_components=3, random_state=42)
X_embedded = tsne.fit_transform(X_standardized)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_embedded[:, 0], X_embedded[:, 1], X_embedded[:, 2], c=y, cmap='viridis')

ax.set_title('t-SNE 3D Plot')
ax.set_xlabel('Dimension 1')
ax.set_ylabel('Dimension 2')
ax.set_zlabel('Dimension 3')
# Visualize the t-SNE embeddings
# plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y, cmap='viridis')
# plt.title('t-SNE Visualization')
# plt.xlabel('Dimension 1')
# plt.ylabel('Dimension 2')
# plt.colorbar()
#plt.savefig('tsne.png')
plt.show(block=False)
plt.savefig('tsne.png')