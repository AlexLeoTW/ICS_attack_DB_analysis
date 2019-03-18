import os
import pandas as pd
import numpy as np
from keras.models import Sequential
from sklearn import metrics

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

# exclude cols by regex (or name)
def dropp_columns_regex(df, regex):
    if regex == '' or regex == None:
        return

    drop_columns = set(df.filter(regex=regex, axis=1).columns)
    print('drop_columns = {}'.format(drop_columns))
    selected_columns = set(df.columns) - drop_columns
    df = df[selected_columns]

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

# Caculate: acc_score, reca_score, prec_score, f1_score
def measure_accuracy(model, x_test, y_test):
    pred = model.predict(x_test)
    pred = np.argmax(pred,axis=2)[:,-1]
    y_eval = np.argmax(y_test,axis=2)[:,-1]

    acc_score = metrics.accuracy_score(y_eval, pred)
    # print('Accuracy =', acc_score*100)

    reca_score = metrics.recall_score(y_eval, pred, average='macro')
    # print('Recall =', reca_score*100)

    prec_score = metrics.precision_score(y_eval, pred, average='macro')
    # print('Precision =', prec_score*100)

    f1_score = metrics.f1_score(y_eval, pred, average='macro')
    # print('F-score =', f1_score)

    return acc_score, reca_score, prec_score, f1_score

def dir_create_if_not_exist(file_path):
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))

def save_results(ann_name, options, model=None, validation_data=None, fit_result=None, train_time=np.nan):
    if not options.model_path == None:
        if not isinstance(model, Sequential):
            raise ValueError('"model" must be an instance of keras.model.Sequential')

        print('saving model "{}"...'.format(os.path.basename(options.model_path)))
        dir_create_if_not_exist(options.model_path)
        model.save(options.model_path)

    if not options.log_path == None:
        print('saving epochs log "{}"...'.format(os.path.basename(options.log_path)))
        dir_create_if_not_exist(options.log_path)
        log_csv = open(options.log_path, "w")
        log_csv.write(pd.DataFrame(fit_result.history).to_csv())

    if not options.statistics_path == None:
        if not isinstance(model, Sequential):
            raise ValueError('"model" must be an instance of keras.model.Sequential')
        if len(validation_data) < 2:
            raise ValueError('"validation_data" must contain x_test and y_test')

        x_test, y_test = validation_data
        acc_score, reca_score, prec_score, f1_score = measure_accuracy(model, x_test, y_test)
        epochs = len(fit_result.history['val_acc']) if fit_result != None else np.nan
        avg_epoch_time = train_time / epochs
        last_val_loss = fit_result.history['val_loss'][-1] if fit_result != None else np.nan
        last_val_acc = fit_result.history['val_acc'][-1] if fit_result != None else np.nan
        last_loss = fit_result.history['loss'][-1] if fit_result != None else np.nan
        last_acc = fit_result.history['acc'][-1] if fit_result != None else np.nan

        statistics_columns = ['ann_name', 'step_size', 'units',
            'acc_score', 'reca_score', 'prec_score', 'f1_score',
            'train_time', 'epochs', 'avg_epoch_time',
            'last_val_loss', 'last_val_acc', 'last_loss', 'last_acc']

        history_statistics = pd.DataFrame(columns=statistics_columns)
        if os.path.isfile(options.statistics_path):
            history_statistics = pd.read_csv(options.statistics_path)

        statistics = pd.DataFrame([[ann_name, options.step_size, options.units,
            acc_score, reca_score, prec_score, f1_score,
            train_time, epochs, avg_epoch_time,
            last_val_loss, last_val_acc, last_loss, last_acc]], columns=statistics_columns)

        history_statistics = history_statistics.append(statistics, ignore_index=True)
        if options.drop_duplicates:
            history_statistics.drop_duplicates(subset=['ann_name', 'step_size', 'units'], inplace=True, keep='last')

        print('saving history statistics "{}"...'.format(os.path.basename(options.statistics_path)))
        dir_create_if_not_exist(options.statistics_path)
        statistics_csv = open(options.statistics_path, "w")
        statistics_csv.write(history_statistics.to_csv(index=False))
