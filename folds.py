import pandas as pd
from sklearn.model_selection import KFold

df = pd.read_csv('cleaned_dataset_Thyroid1.csv')
X = df.drop('binaryClass', axis=1)
y = df['binaryClass']

kf = KFold(n_splits=5, shuffle=True)
FOLDS = list(kf.split(X, y))
