from sklearn import preprocessing

# Encode text values to indexes(i.e. [1],[2],[3] for red,green,blue).
def encode_text_index(df, name):
    le = preprocessing.LabelEncoder()
    df[name] = le.fit_transform(df[name])
    return le.classes_

# Encode numeric columns in DataFrame as zscores
def encode_numeric_zscore(df, cols=None, exclude=[]):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    print(type(exclude))

    if cols == None:
        cols = df.columns
        cols = cols[df.dtypes.apply(lambda x: x.name).isin(numerics) == True]

    if isinstance(exclude, str):
        exclude = [exclude,]

    cols = set(cols) - set(exclude)

    for name in cols:
        df[name] = df[name].transform(lambda x: (x - x.mean()) / x.std())

    return list(cols)


# DEPRECATED
# # Encode a numeric column as zscores
# def encode_numeric_col_zscore(df, name, mean=None, sd=None):
#     # if mean is None:
#     #     mean = df[name].mean()
#     #
#     # if sd is None:
#     #     sd = df[name].std()
#     #
#     # df[name] = (df[name] - mean) / sd
#     df[name] = df[name].transform(lambda x: (x - x.mean()) / x.std())
