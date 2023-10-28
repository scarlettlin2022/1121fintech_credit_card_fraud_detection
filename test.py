import pandas as pd
import numpy as np

df = pd.read_csv("training.csv")

print(np.sum(df['label'].values))

df = df.drop(columns=['acqic', 'cano', 'chid', 'csmam', 'csmcu', 'insfg', 'etymd', 'hcefg', 'mchno', 'ovrlt', 'scity']).fillna(0)


df.to_csv("preprocessed.csv", index=False)