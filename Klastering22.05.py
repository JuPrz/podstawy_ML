import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.preprocessing import LabelEncoder 
import math
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans

#zaczytanie danych z pliku

df_train = pd.read_csv("podstawy_ML/countries of the world.csv" , sep = "," , decimal= ',', encoding = 'utf-8')
print(df_train)

# Data preprocessing - Data Cleaning
df_train = df_train.dropna()  # Remove rows with missing values
df_train = df_train.reset_index(drop=True)

reg_col = ["Region"]
df_train = df_train.drop("Country", axis=1)  # Drop the "Country" column
label_encoder = LabelEncoder()
df_train[reg_col] = df_train[reg_col].apply(label_encoder.fit_transform)

kmeans = KMeans(n_clusters=3)
kmeans.fit(df_train)
labels = kmeans.labels_ #wyniki który wiersz do którego klustra
print(labels)

model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
model.fit(df_train)


# Create a scatter plot
plt.figure(figsize=(12, 8))
plt.scatter(df_train.iloc[:, 0], df_train.iloc[:, 1], c=labels, cmap='viridis')
plt.xlabel("Region")
plt.ylabel("Population")
plt.title("Regions features")
plt.colorbar(label="Cluster")
plt.show()

# Step 5: Define the plot_dendrogram function
def plot_dendrogram(model, **kwargs):
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    dendrogram(linkage_matrix, **kwargs)

# Step 6: Plot the dendrogram
plt.figure(figsize=(10, 6))
plt.title("Hierarchical Clustering Dendrogram")
plot_dendrogram(model, truncate_mode="level", p=3)
plt.xlabel("Number of countries in node")
plt.ylabel("Distance")
plt.show()