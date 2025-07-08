import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from deep_metabolitics.config import data_dir


# fpath = data_dir / "pathwayfluxminmax_10_folds/label_train_0.parquet.gzip"

# df = pd.read_parquet(fpath)

# print(f"{df.min().min() = }")
# print(f"{df.max().max() = }")

# standard_scaled_df = pd.DataFrame(StandardScaler().fit_transform(df), columns=df.columns)

# print(f"{standard_scaled_df.min().min() = }")
# print(f"{standard_scaled_df.max().max() = }")


# minmax_scaled_df = pd.DataFrame(MinMaxScaler().fit_transform(df), columns=df.columns)

# print(f"{minmax_scaled_df.min().min() = }")
# print(f"{minmax_scaled_df.max().max() = }")


fpath = data_dir / "pathwayfluxminmax_10_folds/metabolomics_train_0.parquet.gzip"
df = pd.read_parquet(fpath)

print("Row base null rate")
print(df.isna().mean(axis=1).describe())

print("Column base null rate")
print(df.isna().mean(axis=0).describe())


df = df.loc[:, df.isna().mean() < 0.9]

print(f"{df.shape = }")

print("Row base null rate")
print(df.isna().mean(axis=1).describe())

print("Column base null rate")
print(df.isna().mean(axis=0).describe())

df = df[df.isna().mean(axis=1) <= 0.75]

print(f"{df.shape = }")

print("Row base null rate")
print(df.isna().mean(axis=1).describe())

print("Column base null rate")
print(df.isna().mean(axis=0).describe())