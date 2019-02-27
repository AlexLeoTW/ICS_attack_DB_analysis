import os
import pandas as pd
import numpy as np

# read CSVs in a directory into a single DataFrame
def batch_read_csv(dirpath='.'):
    files = os.listdir(dirpath)
    df = None

    # sort by number in filename
    files = list(filter(lambda name: name.endswith('.csv'), files))
    files.sort(key = lambda name: int(''.join(filter(str.isdigit, name))))

    for filename in files:
        filepath = os.path.abspath(os.path.join(dirpath, filename))
        print('reading "{}"...'.format(filename))

        if not isinstance(df, pd.DataFrame):
            df = pd.read_csv(filepath)
        else:
            df = df.append(pd.read_csv(filepath))

    return df, files

# dealing with: NaN, ∞, -∞
def cleanup(df):
    old_columns = df.columns

    def remove(df):
        df = df[df.isin([np.nan, np.inf, -np.inf]).any(axis=1) == False]

    def replace_with_high_std(df, std_increase=10):
        df.dropna(axis=1, how='any', inplace=True)

        cols_name_w_missing = df.columns[df.isin([np.nan, np.inf, -np.inf]).any(axis=0)]
        cols_w_missing = df[cols_name_w_missing]
        # print(cols_w_missing)
        wo_missing = cols_w_missing[cols_w_missing.abs() < np.inf]
        cols_max_std = (wo_missing - wo_missing.mean()).abs().max() / wo_missing.std()

        new_inf_val = wo_missing.mean() + wo_missing.std() * (cols_max_std + std_increase)
        new_neg_inf = wo_missing.mean() - wo_missing.std() * (cols_max_std + std_increase)

        df.replace(np.inf, new_inf_val, inplace=True)
        df.replace(-np.inf, new_neg_inf, inplace=True)

    def replace_with_type_min_max(df):
        df.dropna(axis=1, how='any', inplace=True)

        for col in df.columns:
            col_type = df[col].dtype
            print('preprocessing {}({})...'.format(col, col_type))
            if col_type in ['float16', 'float32', 'float64']:
                df[col].replace(np.inf, np.finfo(col_type).max, inplace=True)
                df[col].replace(-np.inf, np.finfo(col_type).min, inplace=True)
            if col_type in ['int16', 'int32', 'int64']:
                df[col].replace(np.inf, np.iinfo(col_type).max, inplace=True)
                df[col].replace(-np.inf, np.iinfo(col_type).min, inplace=True)

    remove(df)
    # replace_with_high_std(df)
    # replace_with_type_min_max(df)

    # return dropped columns in list
    return list(set(old_columns) - set(df.columns))

# split source DataFrame into "input data (x)" and "expected output (y)"
def to_xy(df, target):
    result = []
    for x in df.columns:
        if x != target:
            result.append(x)
    # find out the type of the target column.  Is it really this hard? :(
    target_type = df[target].dtypes
    target_type = target_type[0] if hasattr(target_type, '__iter__') else target_type
    # Encode to int for classification, float otherwise. TensorFlow likes 32 bits.
    if target_type in (np.int64, np.int32):
        # Classification
        dummies = pd.get_dummies(df[target])
        return df[result].astype(np.float32), dummies.astype(np.float32)
    else:
        # Regression
        return df[result].astype(np.float32), df[target].astype(np.float32)

# Create time sequences of x, y with time_step
def to_sequences(x, y, step_size):
    if x.shape[0] != y.shape[0]:
        return None, None

    step_count = x.shape[0] - step_size + 1
    x_seq = np.zeros((step_count, step_size, x.shape[1]))
    y_seq = np.zeros((step_count, step_size, y.shape[1]))
    for i in range(step_count):
        x_seq[i] = x[i:i+step_size]
        y_seq[i] = y[i:i+step_size]

    return x_seq, y_seq
