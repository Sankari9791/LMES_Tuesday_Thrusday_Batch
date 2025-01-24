import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


df = pd.read_csv("diamonds.csv")
print(df.head())

numeric_columns = ['carat', 'depth', 'table', 'price', 'x', 'y', 'z']
numeric_data = df[numeric_columns]

# Standardizing the numeric data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numeric_data)

# Applying PCA
pca = PCA()
pca_result = pca.fit_transform(scaled_data)

print(pca_result)

pca_columns = ["PC1", "PC2", "PC3", "PC4", "PC5", "PC6", "PC7"]
print(pca_columns)
pca_df = pd.DataFrame(pca_result, columns=pca_columns)
print(pca_df)