# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker(集装箱) image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import albumentations as A
import os
from tensorflow.keras.layers import *
from tensorflow.keras import Sequential
from tensorflow.keras import optimizers
from tensorflow.keras import layers
from tensorflow.keras import regularizers
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.signal import resample
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
# from numpy.random import seed
# from tensorflow import set_random_seed
from tensorflow.keras import regularizers
from sklearn.preprocessing import StandardScaler
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all" # 让结果全部显示
import seaborn as sns
import matplotlib.pyplot as plt
# from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Activation，AveragePooling2D, Input, Flatten
import gc
import albumentations as A
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import random
from tensorflow import random
import os
from tensorflow.keras import initializers
from time import *
begin_time = time()

# 设置随机数
SEED =2031

os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['PYTHONHASHSEED']=str(SEED)
np.random.seed(SEED)
random.set_seed(SEED)

data_path = './data/'
train = pd.read_csv(data_path+'sensor_train.csv')
test = pd.read_csv(data_path+'sensor_test.csv')
sub = pd.read_csv(data_path+'result.csv')
y = train.groupby('fragment_id')['behavior_id'].min()

data = pd.concat([train,test],sort =False)# 合并数据集

data1 = data[['fragment_id','behavior_id','time_point']]#跳出类别性特征 
data['acc_all'] = (data.acc_x ** 2 + data.acc_y ** 2 + data.acc_z ** 2) ** .5
data['acc_allg'] = (data.acc_xg ** 2 + data.acc_yg ** 2 + data.acc_zg ** 2) ** .5# 构造特征
numercial_feature = [i for i in data.columns if i not in ['fragment_id','behavior_id','time_point']]  #数值型特征
#划分数据
train = data[data.behavior_id.isna()==False].reset_index(drop=True)
test=data[data.behavior_id.isna()==True].reset_index(drop=True)

# 数据标准化
sta = StandardScaler().fit(data[numercial_feature])
sta_data1 = sta.transform(data[numercial_feature])
# 数据转化为pandas
sta_data1 = pd.DataFrame(sta_data1,columns=numercial_feature,index=data.index)
sta_data1.head()
# 合并其他变量
sta_data = pd.concat([data1,sta_data1],axis=1,ignore_index=False)
sta_data.head()

#划分数据
sta_train = sta_data[sta_data.behavior_id.isna()==False].reset_index(drop=True)
# sta_new_train = sta_data[sta_data.fragment_id>10000].reset_index(drop=True)
sta_test=sta_data[sta_data.behavior_id.isna()==True].reset_index(drop=True)
sta_train
sta_test
sta_test=sta_test.drop('behavior_id',axis=1)

x = np.zeros((7292, 60, 8, 1))# 创建维度的矩阵
t = np.zeros((7500, 60, 8, 1))
for i in tqdm(range(7292)):
    tmp = sta_train[sta_train.fragment_id == i][:]
    x[i,:,:, 0] = resample(tmp.drop(['fragment_id', 'time_point', 'behavior_id'],
                                    axis=1), 60, np.array(tmp.time_point))[0]
for i in tqdm(range(7500)):
    tmp = sta_test[sta_test.fragment_id == i][:]
    t[i,:,:, 0] = resample(tmp.drop(['fragment_id', 'time_point'],# resample重新采样
                                    axis=1), 60, np.array(tmp.time_point))[0]

# 图像中生成正方形黑块   测试过
def data_Cutout(x,y):
    x1 = np.zeros((x.shape[0],x.shape[1],x.shape[2],1))
    for i in range(x.shape[0]):
        transform = A.Cutout(num_holes=8, max_h_size=1, max_w_size=1,  always_apply=False, p=0.5)#fill_value=0,
        x1[i,:,:,0] = transform(image=x[i,:,:,0])['image']
    x = np.vstack((x,x1))
    y = np.hstack((y,y))
    return x,y

# 线上78.8分模型 

def Net():
    input = Input(shape=(60, 8, 1))
    X = Conv2D(filters=64,
               kernel_size=(3, 3),
#                activation='relu',
               padding='same')(input)
  
    X = Activation('relu')(X)
    X = Conv2D(filters=128,
               kernel_size=(3, 3),
#                activation='relu',
               padding='same')(X)
    
    X = Activation('relu')(X)
    X = AveragePooling2D()(X)
#     X = Dropout(0.4)(X)
    X = Conv2D(filters=256,
               kernel_size=(3, 3),
#                activation='relu',
               padding='same')(X)
    X = Activation('relu')(X)
    X = Conv2D(filters=512,
               kernel_size=(3, 3),
#                activation='relu',
               padding='same')(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = GlobalAveragePooling2D()(X)
    #     X = Dropout(0.4)(X)
    
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
#     X = GaussianNoise(0.01)(X)
#     X = Dropout(0.3)(X)
# kernel_regularizer=regularizers.l2(0.002)
    X = Dense(19,activation='softmax')(X)
    return Model([input], X)




pre = {}
scores_all = []
val_train = np.zeros((7500,19))
nums = 1
cv= 25
for num in range(nums):
    seed = np.random.randint(0,10000)
    print('='*15,'第{}次'.format(num),'随机因子',seed,'='*15)
#     kfold = StratifiedKFold(n_splits=cv, shuffle=True,random_state=seed)
    kfold = StratifiedKFold(n_splits=cv, shuffle=True)
    proba_t = np.zeros((7500, 19))
    train_pred= np.zeros((7292, 19))
    cvscores = []
    valid_fold = []
    # for fold, (xx1, yy1) in enumerate(kfold.split(new_x, new_y)):#返回索引xx,yy
    #     valid_fold.append(yy1)
    for fold, (xx, yy) in enumerate(kfold.split(x, y)):#返回索引xx,yy
        print('='*15,'第{}次'.format(num),'fold=',fold,'='*15)
        
        y_ = to_categorical(y, num_classes=19)# 转换成二进制矩阵 
        # 样本平衡
        
      #  x1,y1= unba_randomos(x[xx],y[xx])
        x1,y1= data_Cutout(x[xx],y[xx])
        y1 = to_categorical(y1, num_classes=19)
        
        

        # x1 =  np.vstack((x[xx],new_x[valid_fold[fold]]))
        # y1 =  np.hstack((y[xx],new_y.values[valid_fold[fold]]))
        # y1 = to_categorical(y1, num_classes=19)
        model = Net()
        model.compile(optimizer=optimizers.Adam(),
                 loss='categorical_crossentropy',#编译网络
                 metrics=['acc'])
        plateau = ReduceLROnPlateau(monitor="val_acc",
                                    verbose=0,
                                    mode='max',
                                    factor=0.10,
                                    patience=6)
        early_stopping = EarlyStopping(monitor='val_acc',
                                       verbose=0,
                                       mode='max',
                                       patience=25)
        checkpoint = ModelCheckpoint(f'fold{fold}.h5',
                                     monitor='val_acc',
                                     verbose=0,
                                     mode='max',
                                     save_best_only=True)
        model.fit(x1,y1,
                  epochs=500,
                  batch_size=32,
                  verbose=1,
                  shuffle=True,
                  validation_data=(x[yy], y_[yy]),
                  callbacks=[plateau, early_stopping, checkpoint])
        
        scores = model.evaluate(x[yy], y_[yy], verbose=0)
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
        cvscores.append(scores[1] * 100)
        model.load_weights(f'fold{fold}.h5')
        proba_t += model.predict(t,verbose=0, batch_size=1024) /cv/nums#最终的预测，5折交叉验证的平均
        train_pred[yy] += model.predict(x[yy],verbose=0, batch_size=1024) #最终的预测，5折交叉验证的平均
        
    
    print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores))) 
    scores_all.append(np.mean(cvscores))
print("综合分数：%.2f%% (+/- %.2f%%)" % (np.mean(scores_all), np.std(scores_all)))
# 计算执行时间
end_time = time()
run_time = (end_time-begin_time)/60
print ('该循环程序运行时间：',run_time,'分钟') #该循环程序运行时间： 1.4201874732

np.save('./npy_file/CNN_valid', train_pred)
np.save('./npy_file/CNN_test', proba_t)


########### 生成伪标签用于其他模型的训练 ############################
proba = proba_t
res_sub = pd.read_csv(data_path+'result.csv') 
## proba_t为预测的测试集的y
jiandu = np.max(proba, axis=1)  

## 记录大于0.95的位置
list_num = []
for i in range(len(jiandu)):
    if jiandu[i]>0.95:
        list_num.append(i)

## 创建成pandas文件
Semi_supervision = pd.DataFrame(list_num, columns=['fragment_id'])
test = pd.read_csv(data_path+'sensor_test.csv')
res_sub.behavior_id = np.argmax(proba, axis=1)
## 为了方便，读取预测文件进行merge
result = res_sub
Semi_supervision = Semi_supervision.merge(result, how='left', on='fragment_id')

## 将labelmerge到test中
sps = test.merge(Semi_supervision, how='left', on='fragment_id')

## 将test的非空部分读取出来的就是新增的数据集
other = sps[~sps['behavior_id'].isna()]

other['behavior_id'] = other['behavior_id'].astype('int16')
other['fragment_id']+=10000
other.to_csv('./pseudo_labels/pseudo_labels.csv',index=False)