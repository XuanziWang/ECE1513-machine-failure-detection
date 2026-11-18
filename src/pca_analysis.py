import pandas as pd
from sklearn.decomposition import PCA
import numpy as np

X_train_scaled = pd.read_csv("data/processed/X_train_scaled.csv")
numerical_cols = ['Air temperature [K]', 'Process temperature [K]',
       'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']

# Air temperature [K] and Process temperature [K] have pretty high correlation
# print(corr)

pca = PCA()
pca.fit(X_train_scaled)

var = np.cumsum(pca.explained_variance_ratio_)
print(var)

## As a result, using 5 components could preserve 95.5% of variance, which doesn't significantly
## decrease the number of features. So, we will not use PCA for this analysis. 