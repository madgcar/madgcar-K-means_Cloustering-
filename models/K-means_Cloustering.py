#!pip install pandas
#!pip install numpy
#!pip install matplotlib
#!pip install seaborn
#!pip install sklearn
#!pip install shutil
#!pip install sqlalchemy
#!pip install os

# Importo las librerias

import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib as plt

houses = pd.read_csv('https://raw.githubusercontent.com/4GeeksAcademy/k-means-project-tutorial/main/housing.csv')

X = houses.loc[:, ["MedInc", "Latitude", "Longitude"]]
X.head()

from sklearn.cluster import KMeans

# Creo una nueva caracteristica
kmeans = KMeans(n_clusters=6)
X["Cluster"] = kmeans.fit_predict(X)
X["Cluster"] = X["Cluster"].astype("category")

X.head()

sns.relplot(
    x="Longitude", y="Latitude", hue="Cluster", data=X, height=6,
);

