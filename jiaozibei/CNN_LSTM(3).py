import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm 
### BUG：混合了Tensorflow keras和keras API。优化器和模型应来自同一层定义。
from scipy.signal import resample
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import Input, Model, regularizers
from tensorflow.compat.v1.keras.layers import CuDNNLSTM, CuDNNGRU
# tensorflow 2.0没有加入对CuDNNLSTM的支持。但是，tensorflow的LSTM对GPU加速优化的稀烂，专门引入只能由GPU加速并行运算的CuDNNLSSTM（）
# 把LSTM改成CuDNNLSTM之后，训练速度至少提升了5倍以上
# CuDNNLSTM是为CUDA并行处理而设计的，如果没有GPU，它将无法运行。而LSTM是为普通CPU设计的。由于并行性，执行时间更快。
from tensorflow.keras.layers import GaussianNoise, LSTM, GRU, Bidirectional, Layer, Conv1D, BatchNormalization, MaxPooling1D, AveragePooling1D, Dropout, SpatialDropout1D, Dense, concatenate, Activation, Lambda, dot
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
### 注意：为了防止不必要的BUG，用keras的时候，必须要带上tensorflow，即tensorflow.keras，遇到的BUG：元组对象没有layer属性，
### 另外，只有keras可能导致调用的东西无效。比如，ReduceLROnPlateau在用keras.callbacks中import的时候，没有起作用，从tensorflow.keras.callbacks调用有作用。
from sklearn.model_selection import StratifiedKFold

### 执行下面语句能够尽量减少每次keras分数的不确定 ###
SEED = 42
import random
import os
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
######################################################

datasets_path = './data/'
models_path = './model/'
sub_path = '/content/drive/My Drive/kesci_JZB/Baseline/sub/'

def acc_combo(y, y_pred, mode):
    # 数值ID与行为编码的对应关系
  mapping = {0: 'A_0', 1: 'A_1', 2: 'A_2', 3: 'A_3', 4: 'D_4', 5: 'A_5', 6: 'B_1',7: 'B_5', 
          8: 'B_2', 9: 'B_3', 10: 'B_0', 11: 'A_6', 12: 'C_1', 13: 'C_3', 14: 'C_0', # 递手机
          15: 'B_6', 16: 'C_2', 17: 'C_5', 18: 'C_6' }
  if mode == 'behavior':  # 场景+动作
    code_y, code_y_pred = mapping[y], mapping[y_pred] 
    if code_y == code_y_pred: # 编码完全相同得分1.0 即 C_0 == C_0
      return 1.0
    elif code_y.split("_")[0] == code_y_pred.split("_")[0]: # 场景相同得 1.0/7 分
      return 1.0/7
    elif code_y.split("_")[1] == code_y_pred.split("_")[1]: # 动作相同得 1.0/3 分
      return 1.0/3
    else: # 都不对，不得分
      return 0.0 

  # if mode == 'scene':  # 最高得到7500/7 = 1071分  0.78左右
  #     mapping_scene = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
  #     code_y, code_y_pred = mapping_scene[y], mapping_scene[y_pred]
  #     if code_y == code_y_pred:
  #         return 1.0/7
  #     else: 
  #         return 0.0

  # if mode == 'action':  # 最高得到7500/3 = 2500分
  #     mapping_action = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6'}
  #     code_y, code_y_pred = mapping_action[y], mapping_action[y_pred]

  #     if code_y == code_y_pred: 
  #         return 1.0/3
  #     else:
  #         return 0.0




train = pd.read_csv(datasets_path+'sensor_train.csv')
test = pd.read_csv(datasets_path+'sensor_test.csv')
sub = pd.read_csv(datasets_path+'result.csv')
# train['train_scene'] = train['behavior_id'].apply(lambda x: 0 if x==0 or x==1 or x==2 or x==3 or x==5 or x==11 else 
#                                 1 if x==6 or x==7 or x==8 or x==9 or x==10 or x==15 else 
#                                 2 if x==12 or x==13 or x==14 or x==16 or x==17 or x==18 else 3)

# train['train_action'] = train['behavior_id'].apply(lambda x: 0 if x==0 or x==10 or x==14 else 1 if x==1 or x==6 or x==12 else 
#                                  2 if x==2 or x==8 or x==6 else 3 if x==3 or x==9 or x==13 else 
#                                  6 if x==11 or x==15 or x==18 else 5 if x==5 or x==7 or x==17 else 4)

# y_scene = train.groupby('fragment_id')['train_scene'].min()
# y_action = train.groupby('fragment_id')['train_action'].min()
y = train.groupby('fragment_id')['behavior_id'].min()

## 求加速度的模
train['mod'] = (train.acc_x ** 2 + train.acc_y ** 2 + train.acc_z ** 2) ** .5
train['modg'] = (train.acc_xg ** 2 + train.acc_yg ** 2 + train.acc_zg ** 2) ** .5
test['mod'] = (test.acc_x ** 2 + test.acc_y ** 2 + test.acc_z ** 2) ** .5
test['modg'] = (test.acc_xg ** 2 + test.acc_yg ** 2 + test.acc_zg ** 2) ** .5

## 8个分别一阶差分
train_diff1 = pd.DataFrame()
diff_fea = ['acc_x','acc_y','acc_z','acc_xg','acc_yg','acc_zg','mod','modg']
train_diff1 = train.groupby('fragment_id')[diff_fea].diff(1).fillna(0.) 
train_diff1.columns = ['x_diff_1','y_diff_1','z_diff_1','xg_diff_1','yg_diff_1','zg_diff_1','mod_diff_1','modg_diff_1']

test_diff1 = pd.DataFrame()
test_diff1 = test.groupby('fragment_id')[diff_fea].diff(1).fillna(0.)
test_diff1.columns = train_diff1.columns
## 8个分别二阶差分
train_diff2 = pd.DataFrame()
train_diff2 = train.groupby('fragment_id')[diff_fea].diff(2).fillna(0.) 
train_diff2.columns = ['x_diff_2','y_diff_2','z_diff_2','xg_diff_2','yg_diff_2','zg_diff_2','mod_diff_2','modg_diff_2']

test_diff2 = pd.DataFrame()
test_diff2 = test.groupby('fragment_id')[diff_fea].diff(2).fillna(0.)
test_diff2.columns = train_diff2.columns

## 融合
train = pd.concat([train, train_diff1, train_diff2], axis = 1)
test = pd.concat([test, test_diff1, test_diff2], axis = 1)

No_train_fea = ['fragment_id', 'time_point', 'behavior_id', 'train_scene', 'train_action']
train_fea = [fea for fea in train.columns if fea not in No_train_fea]
fea_num = len(train_fea)
## 归一化
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
train[train_fea] = pd.DataFrame(scaler.fit_transform(train[train_fea]), columns = train_fea)
test[train_fea] = pd.DataFrame(scaler.fit_transform(test[train_fea]), columns = train_fea)
'''
分析：
如果某个特征的方差远大于其它特征的方差，那么它将会在算法学习中占据主导位置，导致我们的学习器不能像我们期望的那样，去学习其他的特征，
这将导致最后的模型收敛速度慢甚至不收敛，因此需要对这样的特征数据进行标准化/归一化。转化函数为：x =(x - 𝜇)/𝜎
'''



## 原
x = np.zeros((7292, 60, fea_num)) 
t = np.zeros((7500, 60, fea_num))
## 先采集原来的数据集
for i in tqdm(range(7292)):
  tmp = train[train.fragment_id == i][:60].reset_index(drop = True) 
  x[i, :, :] = resample(tmp[train_fea], 60, np.array(tmp.time_point))[0]
for i in tqdm(range(7500)):
  tmp = test[test.fragment_id == i][:60].reset_index(drop = True)
  t[i, :, :] = resample(tmp[train_fea], 60, np.array(tmp.time_point))[0]


# 自定义评估函数：get_acc_combo()
def get_acc_combo():
    def combo(y, y_pred):
        # 数值ID与行为编码的对应关系
        mapping = {0: 'A_0', 1: 'A_1', 2: 'A_2', 3: 'A_3',
                4: 'D_4', 5: 'A_5', 6: 'B_1',7: 'B_5',
                8: 'B_2', 9: 'B_3', 10: 'B_0', 11: 'A_6',
                12: 'C_1', 13: 'C_3', 14: 'C_0', 15: 'B_6',
                16: 'C_2', 17: 'C_5', 18: 'C_6'}
        # 将行为ID转为编码

        code_y, code_y_pred = mapping[int(y)], mapping[int(y_pred)]
        if code_y == code_y_pred: #编码完全相同得分1.0
            return 1.0
        elif code_y.split("_")[0] == code_y_pred.split("_")[0]: #编码仅字母部分相同得分1.0/7
            return 1.0/7
        elif code_y.split("_")[1] == code_y_pred.split("_")[1]: #编码仅数字部分相同得分1.0/3
            return 1.0/3
        else:
            return 0.0
    confusionMatrix=np.zeros((19,19))
    for i in range(19):
      for j in range(19):
        confusionMatrix[i,j]=combo(i,j)
    confusionMatrix=tf.convert_to_tensor(confusionMatrix)

    def acc_combo(y, y_pred):
      y=tf.argmax(y,axis=1)
      y_pred = tf.argmax(y_pred, axis=1)
      indices=tf.stack([y,y_pred],axis=1)
      scores=tf.gather_nd(confusionMatrix,tf.cast(indices,tf.int32))
      return tf.reduce_mean(scores)
    return acc_combo



epochs = 1
batch_size = 120
# 尝试：以128为分界线，向下（*0.5）和向上（*2）训练后比较测试结果，若向下更好则再*0.5，直接结果不再提升
# batchsize设置：通常10到100，一般设置为2的n次方。原因：计算机的gpu和cpu的memory都是2进制方式存储的，设置2的n次方可以加快计算速度。
kernel_size = 3 
pool_size = 2
dropout_rate = 0.4 # 防止过拟合
n_classes = 19
# n_action = 7
# n_scene = 3
act_swish = lambda x:x * tf.nn.sigmoid(x)
# Swish 是一种新型激活函数，公式为： f(x) = x · sigmoid(x)。Swish 具备无上界有下界、平滑、非单调的特性
proba_tA = np.zeros((7500, n_classes)) 
valid = np.zeros((7292, n_classes))
y_ = to_categorical(y, n_classes) ### Q：训练好的模型预测的分数特别低 A：这里的y用的是y_scene（场景标签）而不是y（行为标签）
## loss平滑标签
# CC_Ls = tf.keras.losses.CategoricalCrossentropy(label_smoothing = 0.1) 加了标签平滑分数下降了

# xyz、mod、modg 一、二阶差分 + 三层单向LSTM + StandardScaler 线下 0.82134 线上 0.7437936507936508
def Net(): 
  input = Input(shape=(60, fea_num))
  # model = GaussianNoise(0.1)(input) # 先给数据加入高斯噪声进行数据增强，防止过拟合  为数据施加0均值，标准差为stddev的加性高斯噪声
  model = Conv1D(1024, kernel_size, input_shape=(60, fea_num), activation=act_swish, padding='same', kernel_regularizer = regularizers.l2(0.01))(input) 
  model = BatchNormalization()(model) 
  model = AveragePooling1D(pool_size=pool_size)(model) 
  model = Dropout(dropout_rate)(model) # 一般来说，Dropout仅在池化层后使用
  
  model = Conv1D(512, kernel_size, activation=act_swish, padding='same')(model) 
  model = BatchNormalization()(model) 
  model = AveragePooling1D(pool_size=pool_size)(model)  
  model = Dropout(dropout_rate)(model)

  model = Conv1D(256, kernel_size, activation=act_swish, padding='same')(model) 
  model = BatchNormalization()(model)
  model = AveragePooling1D(pool_size=pool_size)(model) 

  # 单向lstm
  model = CuDNNLSTM(180, return_sequences=True)(model) # GRU比LSTM少一个门，训练的参数少了，容易训练且可以防止过拟合。 CuDNNGRU  CuDNNLSTM
  model = CuDNNLSTM(180, return_sequences=True)(model) 
  model = CuDNNLSTM(180)(model)
  # model = CuDNNGRU(150, return_sequences=True)(model) # (None,180)
  # model = attention_3d_block(model)


  ## 双向lstm + attention
  # model = Bidirectional(LSTM(180, return_sequences=True))(model)  # 默认激活函数为tanh
  # model = Bidirectional(LSTM(180, return_sequences=True))(model)
  # model = Bidirectional(LSTM(180, return_sequences=True))(model)
  # model = attention_3d_block(model)

  model = Dropout(dropout_rate)(model) 
  model = Dense(n_classes)(model) 
  model = BatchNormalization()(model) # 尝试去掉。
  output = Activation('softmax', name="softmax")(model)

  ## 多任务，预测行为、场景、动作公用一个网络，可以缓解过拟合，其中行为作为主任务，场景、动作作为辅助任务。 Hard参数共享
  # model_behavior = Dense(n_classes)(model) 
  # model_behavior = BatchNormalization()(model_behavior) 
  # output_behavior = Activation('softmax', name="behavior_softmax")(model_behavior)

  # model_action = Dense(n_classes)(model) 
  # model_action = BatchNormalization()(model_action) 
  # output_action = Activation('softmax', name="action_softmax")(model_action)

  # model_scene = Dense(n_scene)(model) 
  # model_scene = BatchNormalization()(model_scene) 
  # output_scene = Activation('softmax', name="scene_softmax")(model_scene)

  # return Model(input, output_behavior), Model(input, output_action), Model(input, output_scene)
  return Model(input, output) 

kfold = StratifiedKFold(n_splits=20, shuffle=True, random_state=2020) 
for fold, (xx, yy) in enumerate(kfold.split(x, y)): 
  model = Net()
  model.compile(loss = 'CategoricalCrossentropy', optimizer = 'Nadam', metrics = ['acc']) # 函数的返回值为acc_combo get_acc_combo()
  graph_path = models_path + 'merged_model.png'
#   plot_model(model, to_file = graph_path, show_shapes = True)  # 绘制模型图
  plateau = ReduceLROnPlateau(monitor = 'val_acc', verbose = 0, mode = 'max', factor = 0.1, patience = 8) # 'val_acc_combo'
  early_stopping = EarlyStopping(monitor = 'val_acc', verbose = 0, mode = 'max', patience = 30) # 防止过拟合
  checkpoint = ModelCheckpoint(models_path + f'fold{fold}.h5', monitor = 'val_acc', verbose = 0, mode = 'max', save_best_only = True)
  # 单通道模型
  history = model.fit(x[xx],
              y_[xx], 
              epochs=epochs, 
              batch_size=batch_size, 
              verbose=500, 
              shuffle=True,
              validation_data=(x[yy], y_[yy]), 
              callbacks=[early_stopping, plateau, checkpoint] ## 注：callbacks：输入的是list类型的数据。有验证集，用’val_acc’
             ) ## 注：model.fit（）不返回Keras模型，而是一个History对象，其中包含训练的损失和度量值。

  valid[yy] += model.predict(x[yy], verbose=1, batch_size=batch_size)
  # model.save(models_path + "merged_dcl.h5")  # 存储模型
  ## 对于每一折的验证集都计算得分
  val = model.predict(x[yy], verbose=1, batch_size=batch_size) 

  val_labels = np.argmax(val, axis=1) # 每一行最大概率的索引，即行为id
  val_score = sum(acc_combo(y_true, y_pred, 'behavior') for y_true, y_pred in zip(y[yy], val_labels)) / val_labels.shape[0]
  ### 改：y --> new_y
  print('官方得分：', round(val_score, 5)) # 保留小数点后五位
  print('准确率得分：', round(accuracy_score(y[yy].values, val_labels), 5)) 
  ### 改：y --> new_y
  proba_tA += model.predict(t, verbose=1, batch_size=batch_size) / 20. # 5折预测测试集然后取均值 (7500, 19)

np.save('./npy_file/lstm_valid.npy', valid)
np.save('./npy_file/lstm_test.npy', proba_tA)