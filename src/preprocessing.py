import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('data/ai4i2020.csv')

# Check missing values
# The dataset has no missing value
# print(df.isnull().sum())

# Check distribution 
# print(df["Machine failure"].value_counts())

# creating features and targets
X = df.drop(columns=[
            "UDI", 
            "Product ID", 
            "Machine failure", 
            "TWF", 
            "HDF", 
            "PWF", 
            "OSF",
            "RNF"
    ])

y = df["Machine failure"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1513, stratify=y
)

# convert 'Type' to a one-hot vectors
X_train = pd.get_dummies(X_train, columns=['Type'], dtype=int)
X_test = pd.get_dummies(X_test, columns=['Type'], dtype=int)

# All other columns are numerical
## scale the numerical columns for logistic regression and SVM. Keep original for Random Forest
numerical_cols = ['Air temperature [K]', 'Process temperature [K]',
       'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test_scaled[numerical_cols] = scaler.transform(X_test[numerical_cols])

# print(X_train.head())
# print(X_test.head())

# Writing the processed data to csv
X_train.to_csv("data/processed/X_train.csv", index=False)
X_test.to_csv("data/processed/X_test.csv", index=False)

X_train_scaled.to_csv("data/processed/X_train_scaled.csv", index=False)
X_test_scaled.to_csv("data/processed/X_test_scaled.csv", index=False)

y_train.to_csv("data/processed/y_train.csv", index=False)
y_test.to_csv("data/processed/y_test.csv", index=False)
