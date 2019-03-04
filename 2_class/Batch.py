DIR_PATH = '../../2_classes/'
STEP_SIZE = 20
VALIDATION_SPLIT = 0.2

import os
import time
import numpy as np
import pandas as pd
import encoder as enc
import common as common
import tensorflow as tf
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.recurrent import SimpleRNN, LSTM, GRU
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn import metrics

dirpath = os.path.abspath(os.path.dirname(__file__))
dirpath = os.path.abspath(os.path.join(dirpath, '../../2_classes/data1.csv'))

def exec_on_file(dirpath, callback):
    dirpath = os.path.abspath(os.path.dirname(__file__))
    dirpath = os.path.abspath(os.path.join(dirpath, DIR_PATH))
    files = os.listdir(dirpath)
    files = list(filter(lambda name: name.endswith('.csv'), files))
    files.sort(key = lambda name: int(''.join(filter(str.isdigit, name))))

    for filename in files:
        filepath = os.path.abspath(os.path.join(dirpath, filename))
        dataset_name = filename[:-4]
        print('reading "{}"...'.format(filename))

        df = pd.read_csv(filepath)
        callback(df, dataset_name)

def convert_df_single_file(df):
    # dealing with: NaN, ∞, -∞
    print('===== cleanup =====')
    dropped_columns = common.cleanup(df)
    print('dropped_columns: {}'.format(dropped_columns))

    # encode
    print('===== encode =====')
    def encode(df):
        columns = enc.encode_numeric_zscore(df)
        print('Z-scored columns: \n  {}'.format(columns))
        classes = enc.encode_text_index(df, 'marker')
        print('marker classes: \n  {}'.format(classes))
    encode(df)

    old_columns = df.columns
    df.dropna(inplace=True,axis=1)
    print('dropped_columns: \n\t{}'.format(list(set(old_columns) - set(df.columns))))

    # Convert a Pandas dataframe to the x,y inputs that TensorFlow needs
    print('===== to_xy =====')
    x, y = common.to_xy(df,'marker')
    print('x.shape: {}, {}'.format(x.shape, type(x)))
    print('y.shape: {}, {}'.format(y.shape, type(y)))

    return x, y

def train_recurrent(recurrent, x_train, x_test, y_train, y_test, step_size=20, units=50):
    monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5, verbose=1, mode='auto')

    model = Sequential()
    model.add(recurrent(units=units, input_shape=(step_size, x_train.shape[2]),
        return_sequences = True))
    model.add(Dense(units=2, kernel_initializer='normal', activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    start_time = time.time()
    result = model.fit(x_train, y_train,
        validation_data=(x_test,y_test),
        callbacks=[monitor],
        verbose=1, epochs=200)
    end_time = time.time()
    model.summary()

    return model, (end_time - start_time), result.history

def measure_accuracy(model, x_test, y_test):
    pred = model.predict(x_test)
    pred = np.argmax(pred,axis=2)[:,-1]
    y_eval = np.argmax(y_test,axis=2)[:,-1]

    auc_score = metrics.accuracy_score(y_eval, pred)
    print('Accuracy =', auc_score*100)

    reca_score = metrics.recall_score(y_eval, pred)
    print('Recall =', reca_score*100)

    prec_score = metrics.precision_score(y_eval, pred)
    print('Precision =', prec_score*100)

    f1_score = metrics.f1_score(y_eval, pred)
    print('F-score =', f1_score)

    return auc_score, reca_score, prec_score, f1_score

def test_single_file(df, dataset_name):
    STEP_SIZE_LIST = [1, 5, 10, 15, 20, 25, 30, 35, 40]
    RNN_UNIT_SIZE_LIST = [10, 20, 30, 40, 50, 60]
    RECURRENT = {
        'SimpleRNN': SimpleRNN,
        'LSTM': LSTM,
        'GRU': GRU
    }

    dirpath = os.path.abspath(os.path.dirname(__file__))
    dirpath = os.path.abspath(os.path.join(dirpath, DIR_PATH, dataset_name))

    x, y = convert_df_single_file(df)
    recurrent_acc = pd.DataFrame(columns=['recurrent_name', 'step_size', 'units', 'auc_score', 'reca_score', 'prec_score', 'f1_score', 'train_time'])

    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

    for step_size in STEP_SIZE_LIST:
        x_seq, y_seq = common.to_sequences(x, y, step_size=step_size)
        x_train, x_test, y_train, y_test =  train_test_split(x_seq, y_seq,
            test_size=VALIDATION_SPLIT, random_state=42)

        for recurrent_name in RECURRENT:
            for units in RNN_UNIT_SIZE_LIST:

                graph = tf.Graph()
                with tf.Session(graph=graph):
                    print(recurrent_name, units)
                    model, train_time, history = train_recurrent(RECURRENT[recurrent_name], x_train, x_test, y_train, y_test, step_size, units)

                    model_filename = '{}_S{}_U{}'.format(recurrent_name, step_size, units)
                    model_filepath = os.path.join(dirpath, model_filename)
                    print('saving model "{}"...'.format(model_filename))
                    model.save(model_filepath)

                    history_filename = '{}_S{}_U{}.csv'.format(recurrent_name, step_size, units)
                    history_filepath = os.path.join(dirpath, history_filename)
                    history_csv = open(history_filepath, "w")
                    history_csv.write(pd.DataFrame(history).to_csv())

                    auc_score, reca_score, prec_score, f1_score = measure_accuracy(model, x_test, y_test)
                    recurrent_acc = recurrent_acc.append(pd.DataFrame(
                            [[recurrent_name, step_size, units, auc_score, reca_score, prec_score, f1_score, train_time]],
                            columns=['recurrent_name', 'step_size', 'units', 'auc_score', 'reca_score', 'prec_score', 'f1_score', 'train_time']),
                        ignore_index=True)

    recurrent_acc_csv = open(os.path.join(dirpath, "recurrent_acc.csv"), "w")
    recurrent_acc_csv.write(recurrent_acc.to_csv())

exec_on_file(DIR_PATH, test_single_file)
