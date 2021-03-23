import tensorflow as tf
import numpy as np
from tensorflow import keras
import pandas as pd
import os
from sklearn.model_selection import StratifiedKFold
import gc
import random
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

SEED =1998
from tensorflow import random
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['PYTHONHASHSEED']=str(SEED)
np.random.seed(SEED)
random.set_seed(SEED)

data_dir = './data/'


kfcv_seed = 1998
kfold_func = StratifiedKFold
data_enhance_method = []
def My_acc_combo(y, y_pred, mode):

    if mode== 'behavior':
        # 将行为ID转为编码
        code_y, code_y_pred = mapping[y], mapping[y_pred]
        if code_y == code_y_pred: #编码完全相同得分1.0
            return 1.0
        elif code_y.split("_")[0] == code_y_pred.split("_")[0]: #编码仅字母部分相同得分1.0/7
            return 1.0/7
        elif code_y.split("_")[1] == code_y_pred.split("_")[1]: #编码仅数字部分相同得分1.0/3
            return 1.0/3
        else:
            return 0.0

def set_data_enhance(val):
    if not isinstance(val, list):
        val = [val]
    global data_enhance_method
    data_enhance_method = val

mapping = {0: 'A_0', 1: 'A_1', 2: 'A_2', 3: 'A_3', 
    4: 'D_4', 5: 'A_5', 6: 'B_1',7: 'B_5', 
    8: 'B_2', 9: 'B_3', 10: 'B_0', 11: 'A_6', 
    12: 'C_1', 13: 'C_3', 14: 'C_0', 15: 'B_6', 
    16: 'C_2', 17: 'C_5', 18: 'C_6'}

reversed_mapping = {value: key for key, value in mapping.items()}

def decode_label(label_code):
    str = mapping[label_code]
    scene_code = ord(str.split('_')[0]) - ord('A')
    action_code = ord(str.split('_')[1]) - ord('0')
    return scene_code, action_code

def kfcv_evaluate(model_name, x, y):
    kfold = kfold_func(n_splits=k, shuffle=True, random_state=kfcv_seed)
    evals = {'loss':0.0, 'accuracy':0.0}
    index = 0

    for train, val in kfold.split(x, np.argmax(y, axis=-1)):
        print('Processing fold: %d (%d, %d)' % (index, len(train), len(val)))
        
        model = keras.models.load_model('./models/%s/part_%d.h5' % (model_name, index))

        loss, acc = model.evaluate(x=x[val], y=y[val])
        evals['loss'] += loss / k
        evals['accuracy'] += acc / k
        index += 1
    return evals

def kfcv_predict(model_name, inputs):
    path = './models/' + model_name + '/'
    models = []
    for i in range(k):
        models.append(keras.models.load_model(path + 'part_%d.h5' % i))

    print('%s loaded.' % model_name)
    result = []
    for j in range(k):
        result.append(models[j].predict(inputs))

    print('result got')
    result = sum(result) / k
    return result

def kfcv_fit(builder, x, y, epochs, checkpoint_path, verbose=2, batch_size=64):
    kfold = kfold_func(n_splits=k, shuffle=True, random_state=kfcv_seed)
    histories = []
    evals = []

    if checkpoint_path[len(checkpoint_path) - 1] != '/':
        checkpoint_path += '/'

    for i in range(k):
        if os.path.exists(checkpoint_path + 'part_%d.h5' % i):
            os.remove(checkpoint_path + 'part_%d.h5' % i)

    for index, (train, val) in enumerate(kfold.split(x, np.argmax(y, axis=-1))):
        print('Processing fold: %d (%d, %d)' % (index, len(train), len(val)))
        model = builder()

        x_train = x[train]
        y_train = y[train]

        if len(data_enhance_method) > 0:
            x_train_copy = np.copy(x_train)
            y_train_copy = np.copy(y_train)
            for method in data_enhance_method:
                x_, y_ = data_enhance(method, x_train_copy, y_train_copy)
                x_train = np.r_[x_train, x_]
                y_train = np.r_[y_train, y_]
            x_train, y_train = shuffle(x_train, y_train)
            print('Data enhanced (%s) => %d' % (' '.join(data_enhance_method), len(x_train)))

        checkpoint = keras.callbacks.ModelCheckpoint(checkpoint_path + 'part_%d.h5' % index,
                                 monitor='val_accuracy',
                                 verbose=0,
                                 mode='max',
                                 save_best_only=True)

        h = model.fit(x=x_train, y=y_train,
                epochs=epochs,
                verbose=verbose,
                validation_data=(x[val], y[val]),
                callbacks=[checkpoint],
                batch_size=batch_size,
                shuffle=True
                )
        evals.append(model.evaluate(x=x[val], y=y[val]))
        histories.append(h)
        del model
        gc.collect()
    return histories, evals

def data_enhance(method, train_data, train_labels):
    if method == 'noise':
        noise = train_data + np.random.normal(0, 0.1, size=train_data.shape)
        return noise, train_labels
    
    elif method == 'mixup':
        index = [i for i in range(len(train_labels))]
        np.random.shuffle(index)

        x_mixup = np.zeros(train_data.shape)
        y_mixup = np.zeros(train_labels.shape)

        for i in range(len(train_labels)):
            x1 = train_data[i]
            x2 = train_data[index[i]]
            y1 = train_labels[i]
            y2 = train_labels[index[i]]

            factor = np.random.beta(0.2, 0.2)

            x_mixup[i] = x1 * factor + x2 * (1 - factor)
            y_mixup[i] = y1 * factor + y2 * (1 - factor)

        return x_mixup, y_mixup

def save_results(path, output):
    print('saving...')

    df_r = pd.DataFrame(columns=['fragment_id', 'behavior_id'])
    for i in range(len(output)):
        behavior_id = output[i]
        df_r = df_r.append(
            {'fragment_id': i, 'behavior_id': behavior_id}, ignore_index=True)
    df_r.to_csv(path, index=False)

def infer(model_name, inputs, csv_output):
    output = np.argmax(kfcv_predict(model_name, inputs), axis=-1)
    save_results(csv_output, output)
    print('- END -')
    print('Your file locates at %s' % csv_output)

def shuffle(data, labels, seed=None):
    index = [i for i in range(len(labels))]
    if seed != None:
        np.random.seed(seed)
    np.random.shuffle(index)
    return data[index], labels[index]

# 特征列名称
src_names = ['acc_x', 'acc_y', 'acc_z', 'acc_xg', 'acc_yg', 'acc_zg', 'acc', 'acc_g']

def handle_features(data):
    data.drop(columns=['time_point'], inplace=True)

    data['acc'] = (data.acc_x ** 2 + data.acc_y ** 2 + data.acc_z ** 2) ** 0.5
    data['acc_g'] = (data.acc_xg ** 2 + data.acc_yg ** 2 + data.acc_zg ** 2) ** 0.5

    return data

# 构造numpy特征矩阵
def handle_mats(grouped_data):
    mats = [i.values for i in grouped_data]
    # padding
    for i in range(len(mats)):
        padding_times = 61 - mats[i].shape[0]
        for j in range(padding_times):
            mats[i] = np.append(mats[i], [[0 for _ in range(mats[i].shape[1])]], axis=0)

    mats_padded = np.zeros([len(mats), 61, mats[0].shape[1]])
    for i in range(len(mats)):
        mats_padded[i] = mats[i]

    return mats_padded

def get_test_data(use_scaler=True):
    FILE_NAME = data_dir+"sensor_test.csv"
    FILE_NAME1 = data_dir+"sensor_train.csv"
    data = handle_features(pd.read_csv(FILE_NAME))
    train_ori = handle_features(pd.read_csv(FILE_NAME1)[['fragment_id','time_point','acc_x','acc_y','acc_z','acc_xg','acc_yg','acc_zg']])

    if use_scaler:
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        data[src_names] = scaler.transform(data[src_names].values)
        train_ori[src_names] = scaler.transform(train_ori[src_names].values)


    grouped_data = [i.drop(columns='fragment_id') for _, i in data.groupby('fragment_id')]

    grouped_data1 = [i.drop(columns='fragment_id') for _, i in train_ori.groupby('fragment_id')]

    return [handle_mats(grouped_data), handle_mats(grouped_data1)]

def get_train_data(use_scaler=True, shuffle=False, pseudo_labels_file=None):
    df = pd.read_csv(data_dir+"sensor_train.csv")

    # 简单拼接伪标签
    if pseudo_labels_file != None:
        df = df.append(pd.read_csv(pseudo_labels_file))
    data = handle_features(df)

    # 标准化，并将统计值保存
    if use_scaler:
        scaler = StandardScaler()
        scaler.fit(data[src_names].values)  
        with open('scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
        data[src_names] = scaler.transform(data[src_names].values)

    grouped_data = [i.drop(columns='fragment_id') for _, i in data.groupby('fragment_id')]
    train_labels = np.array([int(i.iloc[0]['behavior_id']) for i in grouped_data])
    for i in range(len(grouped_data)):
        grouped_data[i].drop(columns='behavior_id', inplace=True)
    train_data = handle_mats(grouped_data)
    
    if shuffle:
        index = [i for i in range(len(train_labels))]
        np.random.seed(2020)
        np.random.shuffle(index)

        train_data = train_data[index]
        train_labels = train_labels[index]

    return train_data, train_labels

def get_train_test_data(use_scaler=True, shuffle=True, pseudo_labels_file=None):
    train_data, train_lables = get_train_data(use_scaler, shuffle=False, pseudo_labels_file=pseudo_labels_file)
    test_data, train_ori = get_test_data(use_scaler)
    return train_data, train_lables, test_data, train_ori


def BLOCK(seq, filters, kernal_size):
    cnn = keras.layers.Conv1D(filters, 1, padding='SAME', activation='relu')(seq)
    cnn = keras.layers.LayerNormalization()(cnn)

    cnn = keras.layers.Conv1D(filters, kernal_size, padding='SAME', activation='relu')(cnn)
    cnn = keras.layers.LayerNormalization()(cnn)

    cnn = keras.layers.Conv1D(filters, 1, padding='SAME', activation='relu')(cnn)
    cnn = keras.layers.LayerNormalization()(cnn)

    seq = keras.layers.Conv1D(filters, 1)(seq)
    seq = keras.layers.Add()([seq, cnn])
    return seq

def BLOCK2(seq, filters=128, kernal_size=3):
    seq = BLOCK(seq, filters, kernal_size)
    seq = keras.layers.AveragePooling1D(2)(seq) 
    seq = keras.layers.SpatialDropout1D(0.3)(seq)
    seq = BLOCK(seq, filters//2, kernal_size)
    seq = keras.layers.GlobalAveragePooling1D()(seq)
    return seq

def ComplexConv1D(input_shape, num_classes):
    inputs = keras.layers.Input(shape=input_shape[1:])
    seq_3 = BLOCK2(inputs, kernal_size=3)
    seq_5 = BLOCK2(inputs, kernal_size=3)
    seq_7 = BLOCK2(inputs, kernal_size=3)
    seq = keras.layers.concatenate([seq_3, seq_5, seq_7])
    seq = keras.layers.Dense(512, activation='relu')(seq)
    seq = keras.layers.Dropout(0.3)(seq)
    seq = keras.layers.Dense(128, activation='relu')(seq)
    seq = keras.layers.Dropout(0.3)(seq)
    outputs = keras.layers.Dense(num_classes, activation='softmax')(seq)

    model = keras.models.Model(inputs=[inputs], outputs=[outputs])

    model.compile(optimizer=tf.optimizers.Adam(1e-3),
            loss=tf.losses.CategoricalCrossentropy(label_smoothing=0.1),           
            metrics=['accuracy'])

    return model
# 
# 导入精心挑选的pseudo labels
train_data, train_labels, test_data, train_ori = get_train_test_data(pseudo_labels_file='./pseudo_labels/pseudo_labels.csv')
# 设置数据增强方式 (noise, mixup or both)

# 

la = pd.read_csv(data_dir+"sensor_train.csv")
y_train = la.groupby('fragment_id')['behavior_id'].min()

histories = []
evals = []
checkpoint_path = './checkpoint'
if checkpoint_path[len(checkpoint_path) - 1] != '/':
    checkpoint_path += '/'
 
for i in range(5):
    if os.path.exists(checkpoint_path + 'part_%d.h5' % i):
        os.remove(checkpoint_path + 'part_%d.h5' % i)



x = train_data[:7292]
y = train_labels[:7292]

y = to_categorical(y, num_classes=19)

proba_t = np.zeros((7500, 19))
valid = np.zeros((7292, 19)) 


kfold = StratifiedKFold(n_splits=20, shuffle=True, random_state=2031) 

wei_train_x = train_data[7292:]
wei_label_y = train_labels[7292:]

wei_label_y = to_categorical(wei_label_y, num_classes=19)


## 为伪标签创建20折,然后训练时分配下去
valid_fold = []
kfold1 = StratifiedKFold(n_splits=20, shuffle=True, random_state=2031) 
for index, (train, val) in enumerate(kfold1.split(wei_train_x, np.argmax(wei_label_y,axis=1))):
    valid_fold.append(val)

for index, (train, val) in enumerate(kfold.split(x, np.argmax(y, axis=-1))):
    print('Processing fold: %d (%d, %d)' % (index, len(train), len(val)))
    model = ComplexConv1D(train_data.shape, 19)
    
    x_train = x[train]
    y_train = y[train]

    x_train = np.vstack([x_train, wei_train_x[valid_fold[index]]])
    y_train = np.vstack([y_train, wei_label_y[valid_fold[index]]])

    early_stopping = EarlyStopping(monitor='val_accuracy', verbose=0, mode='max', patience=30)
    checkpoint = keras.callbacks.ModelCheckpoint(checkpoint_path + 'part_%d.h5' % index,  monitor='val_accuracy',  verbose=0,  mode='max', save_best_only=True)
    plateau = ReduceLROnPlateau(monitor="val_accuracy", verbose=0, mode='max', factor=0.1, patience=8)
    h = model.fit(x=x_train, y=y_train,epochs=500, verbose=2,validation_data=(x[val], y[val]),
                  callbacks=[checkpoint,early_stopping,plateau], batch_size=64,shuffle=True)

    valid[val] += model.predict(x[val])
    val_ = model.predict(x[val])
    proba_t += model.predict(test_data)/20.

    print('准确率得分：', round(accuracy_score(np.argmax(y[val], axis=1), np.argmax(val_, axis=1)), 5))

    del model
    gc.collect()

np.save('./npy_file/oneD_CNN_valid', valid)
np.save('./npy_file/oneD_CNN_test', proba_t)