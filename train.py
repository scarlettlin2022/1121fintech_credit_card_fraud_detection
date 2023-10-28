from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

df = pd.read_csv("preprocessed.csv")
X=df.drop(columns=['label', 'txkey'])
y=df['label']
print(df.columns) 

X, y = make_classification(n_samples=100, random_state=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,
                                                    random_state=1)
clf = MLPClassifier(random_state=1, max_iter=300).fit(X_train, y_train)
#print(clf.predict_proba(X_test[:1]))
#clf.predict(X_test[:5, :])
print(clf.score(X_test, y_test))

df_test = pd.read_csv("public_processed.csv")
df_name = pd.DataFrame(df_test['txkey'].values, columns=['txkey'])
df_test = df_test.drop(columns=['txkey', 'cano', 'acqic', 'chid', 'csmam', 'csmcu', 'insfg', 'etymd', 'hcefg', 'mchno', 'ovrlt', 'scity']).fillna(0)


pred=clf.predict(df_test)
print(df_test.shape[0])
df_name.insert(1, "pred", pred)
df_name.to_csv("1.csv", index=False)
df_name.append(pd.DataFrame(pred))
