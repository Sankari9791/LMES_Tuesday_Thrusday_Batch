import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

df=pd.read_csv("diamonds.csv")
print(df.head())

df_column=df.columns
df.drop(columns="Unnamed: 0",inplace=True,axis=1)
df_column=df.columns

numeric_columns=[]
for x in df_column:
    if df[x].dtype!="object":
        numeric_columns.append(x)
print(numeric_columns)
#(or)
# numeric_columns = ['carat', 'depth', 'table', 'price', 'x', 'y', 'z']

#
#
# #standarize the scaler
scaler=StandardScaler()
scaled_data=scaler.fit_transform(df[numeric_columns])
print(scaled_data)
#
#  #apply pca
pca=PCA(n_components=5)
pca_result=pca.fit_transform(scaled_data)

print(pca_result)
# #
pca_columns = ["PC1", "PC2", "PC3", "PC4", "PC5"]
print(pca_columns)
pca_df = pd.DataFrame(pca_result, columns=pca_columns)
print(pca_df)