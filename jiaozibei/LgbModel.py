import pandas as pd
from sklearn.model_selection import StratifiedKFold,KFold
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.preprocessing import LabelEncoder#标签编码
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import entropy, skew, kurtosis
import numpy as np
import gc  
import os 
from tqdm import tqdm
pd.set_option('display.max_columns', 600)
pd.set_option('display.max_rows', 600) # 显示的最大行数和列数
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all" # 让结果全部显示
import lightgbm as lgb

data_path = './data/'
data_train = pd.read_csv(data_path+'sensor_train.csv')
data_test = pd.read_csv(data_path+'sensor_test.csv')
data_test['fragment_id'] += 10000# 为什么给这一列加10000
label = 'behavior_id'
sub = pd.read_csv(data_path+'result.csv')
data = pd.concat([data_train, data_test], sort=False)

def feature(data):
    data['acc'] = (data['acc_x'] ** 2 + data['acc_y'] ** 2 + data['acc_z'] ** 2) ** 0.5
    # 不含重力加速度的总和
    data['accg'] = (data['acc_xg'] ** 2 + data['acc_yg'] ** 2 + data['acc_zg'] ** 2) ** 0.5
    # 含有重力加速度的总和
    data['angle'] = (data['acc_x']*data['acc_xg']+data['acc_y']*data['acc_yg']+data['acc_z']*data['acc_zg'])/(data['acc']*data['accg'])
    # 角度和
    # 平面各个加速度之和
    data['acc_xy'] = (data['acc_x'] ** 2 + data['acc_y'] ** 2 ) ** 0.5
    data['accg_xy'] = (data['acc_xg'] ** 2 + data['acc_yg'] ** 2 ) ** 0.5
    data['acc_xz'] = (data['acc_x'] ** 2 + data['acc_z'] ** 2) ** 0.5
    data['accg_xz'] = (data['acc_xg'] ** 2 + data['acc_zg'] ** 2) ** 0.5
    data['acc_yz'] = (data['acc_y'] ** 2 + data['acc_z'] ** 2) ** 0.5
    data['accg_yz'] = (data['acc_yg'] ** 2 + data['acc_zg'] ** 2) ** 0.5
    # 平面角度
    data['angle_xy'] = (data['acc_x']*data['acc_xg']+data['acc_y']*data['acc_yg'])/(data['acc_xy']*data['accg_xy'])
    data['angle_xz'] = (data['acc_x']*data['acc_xg']+data['acc_z']*data['acc_zg'])/(data['acc_xz']*data['accg_xz'])
    data['angle_yz'] = (data['acc_y']*data['acc_yg']+data['acc_z']*data['acc_zg'])/(data['acc_yz']*data['accg_yz'])
    return data

data = feature(data)
data['behavior_id'].fillna('test',inplace=True)
data.fillna(0,inplace=True)

map_features_fun={
    'fft_dc' : lambda x: np.abs(np.fft.fft(x))[0],
    'fft_mean' : lambda x: np.mean(np.abs(np.fft.fft(x))[1:int(len(x) / 2)+1]),
    'fft_var' : lambda x: np.var(np.abs(np.fft.fft(x))[1:int(len(x) / 2)+1]),
    'fft_std' : lambda x: np.std(np.abs(np.fft.fft(x))[1:int(len(x) / 2)+1]),
    'fft_sum' : lambda x: np.sum(np.abs(np.fft.fft(x))[1:int(len(x) / 2)+1]),
    'fft_entropy' : lambda x: -1.0 * np.sum(np.log2(np.abs(np.fft.fft(x))[1:int(len(x) / 2)+1]/
                                                    np.sum(np.abs(np.fft.fft(x))[1:int(len(x) / 2)+1]))),
    'fft_energy' : lambda x: np.sum(np.power(np.abs(np.fft.fft(x))[1:int(len(x) / 2)+1],2)),
    'fft_skew' : lambda x: skew(np.abs(np.fft.fft(x))[1:int(len(x) / 2)+1]),
    'fft_kurtosis' : lambda x: kurtosis(np.abs(np.fft.fft(x))[1:int(len(x) / 2)+1]),
    'fft_max' : lambda x: np.max(np.abs(np.fft.fft(x))[1:int(len(x) / 2)+1]),
    'fft_min' : lambda x: np.min(np.abs(np.fft.fft(x))[1:int(len(x) / 2)+1]),
    'fft_maxind' : lambda x: np.argmax(np.abs(np.fft.fft(x))[1:int(len(x) / 2)+1]),
    'fft_minind' : lambda x: np.argmin(np.abs(np.fft.fft(x))[1:int(len(x) / 2)+1]),
    # 频域特征
    # 时域特征
    'time_kurtosis' : lambda x: kurtosis(x),
    'time_energy' : lambda x: np.sum(np.power(x,2)),
    'time_mad' : lambda x: np.mean(np.absolute(x - np.mean(x))),
    'time_minind' : lambda x: np.argmin(x),
    'time_maxind' : lambda x: np.argmax(x),
    'time_percent_9' : lambda x: np.percentile(x, 0.9),
    'time_percent_75' : lambda x: np.percentile(x, 0.75),
    'time_percent_25' : lambda x: np.percentile(x, 0.25),
    'time_percent_1' : lambda x: np.percentile(x, 0.1),
    'time_percent_75_25' : lambda x: np.percentile(x,75)-np.percentile(x,25),
    'time_zcr': lambda x: (np.diff(np.sign(x))!= 0).sum(),
    'time_mcr' : lambda x: (np.diff(np.sign(x-np.mean(x)))!= 0).sum(),
}
map_feature_fun = ['min', 'max', 'median', 'std', 'skew','var']
    
def feature1(x):
    df = x.drop_duplicates(subset=['fragment_id']).reset_index(drop=True)[['fragment_id', 'behavior_id']]
    for f in tqdm([i for i in x.columns if i not in ['fragment_id','time_point','behavior_id']]):
        # tpdm设置进度条
        # 拿出所有‘acc’的索引
    #     for stat in ['min', 'max', 'mean', 'median', 'std', 'skew']:
        df[f] = x.groupby('fragment_id')[f].agg('mean').values
                       # 利用fragment_id 进行分组[f]特征 使用agg进行列的聚合
        # tpdm设置进度条
        # 拿出所有‘acc’的索引
        for stat in map_feature_fun:
            df[f+'_'+stat] = x.groupby('fragment_id')[f].agg(stat).values
                                    # 利用fragment_id 进行分组[f]特征 使用agg进行列的聚合
        for f_name, f_fun in map_features_fun.items():
            df[f + '_' + f_name] = x.groupby('fragment_id')[f].agg(stat).values
#             df_data_list[col].map(f_fun)
                                    # 利用fragment_id 进行分组[f]特征 使用agg进行列的聚合
    return df

data_feature = feature1(data)
def feature2(data_norm):  
    data_norm['acc_x_diff'] = data_norm['acc_x_max']-data_norm['acc_x_min']
    data_norm['acc_y_diff'] = data_norm['acc_y_max']-data_norm['acc_y_min']
    data_norm['acc_z_diff'] = data_norm['acc_z_max']-data_norm['acc_z_min']
    data_norm['acc_xg_diff'] = data_norm['acc_xg_max']-data_norm['acc_xg_min']
    data_norm['acc_yg_diff'] = data_norm['acc_yg_max']-data_norm['acc_yg_min']
    data_norm['acc_zg_diff'] = data_norm['acc_zg_max']-data_norm['acc_zg_min']
    data_norm['acc_diff'] = data_norm['acc_max']-data_norm['acc_min']
    data_norm['accg_diff'] = data_norm['acc_max']-data_norm['acc_min']
    data_norm['angle_diff'] = data_norm['angle_max']-data_norm['angle_min']

    data_norm['acc_xy_diff'] = data_norm['acc_xy_max']-data_norm['acc_xy_min']
    data_norm['acc_xy_diff'] = data_norm['acc_xy_max']-data_norm['acc_xy_min']
    data_norm['acc_yz_diff'] = data_norm['acc_yz_max']-data_norm['acc_yz_min']

    data_norm['accg_xy_diff'] = data_norm['accg_xy_max']-data_norm['accg_xy_min']
    data_norm['accg_xy_diff'] = data_norm['accg_xy_max']-data_norm['accg_xy_min']
    data_norm['accg_yz_diff'] = data_norm['accg_yz_max']-data_norm['accg_yz_min']

    data_norm['angle_xy_diff'] = data_norm['angle_xy_max']-data_norm['angle_xy_min']
    data_norm['angle_xy_diff'] = data_norm['angle_xy_max']-data_norm['angle_xy_min']
    data_norm['angle_yz_diff'] = data_norm['angle_yz_max']-data_norm['angle_yz_min']
    return data_norm

feature2(data_feature)
# 划分数据集
train_feature = data_feature[data_feature[label]!='test'].reset_index(drop=True)
test_feature= data_feature[data_feature[label]=='test'].reset_index(drop=True)
x = train_feature[[i for i in train_feature.columns if i!=label]]
y = train_feature[label].astype('int')

drop_feat = []
used_feat = [f for f in x.columns if f not in (['fragment_id', label] + drop_feat)]
x_trn, x_vld, y_trn, y_vld = train_test_split(x, y, test_size=0.3, random_state=42) 
train_x = x_trn[used_feat]# 划分训练集
x_vld = x_vld[used_feat]
train_y = y_trn


scores = []
# imp = pd.DataFrame()
# imp['feat'] = used_feat

params = {
    'learning_rate': 0.11642644888214702,
    'metric': 'multi_error',
    'objective': 'multiclass',
    'num_class': 19,
    'feature_fraction': 0.7280260366928102,
    'bagging_fraction': 0.8106130717132529,
    'bagging_freq': 5,
    'n_jobs': 4,
    'seed': 2020,
    'max_depth': 27,# 最大深度
    'num_leaves': 90,# 叶子节点数目，小于2**max_depth
    'lambda_l1': 0.7778891428757128,
    'lambda_l2': 0.8715624879670723,
}
oof_train = np.zeros((len(train_x), 19))

train_set = lgb.Dataset(train_x, train_y) 
val_set = lgb.Dataset(x_vld, y_vld)
 
model = lgb.train(params, train_set, num_boost_round=500000,
                      valid_sets=(train_set, val_set), early_stopping_rounds=100,
                      verbose_eval=20)


# 对于每一折都计算得分
val = model.predict(x_vld, num_iteration=model.best_iteration) 

val_labels = np.argmax(val, axis=1)
# val_score = sum(acc_combo(y_true, y_pred) for y_true, y_pred in zip(y_vld, val_labels)) / val_labels.shape[0]
# print('官方得分：', round(val_score, 5))
print('准确率得分：', round(accuracy_score(y_vld, val_labels), 5))  


## 综合考虑重要性
imp = pd.DataFrame()
imp['feat'] = used_feat
imp['gain'] = model.feature_importance(importance_type='gain')
imp['split'] = model.feature_importance(importance_type='split')

imp_gain = imp[['feat', 'gain']].sort_values(by=['gain'], ascending=False)
imp_split = imp[['feat', 'split']].sort_values(by=['split'], ascending=False)


from sklearn.preprocessing import MinMaxScaler
gain_scaler = imp[['gain']].copy()
split_scaler = imp[['split']].copy() 

gain_scaler_gui_one = MinMaxScaler().fit_transform(gain_scaler[['gain']])
gain_scaler_gui_one = pd.DataFrame(gain_scaler_gui_one,columns=['gain'])

split_scaler_gui_one = MinMaxScaler().fit_transform(split_scaler[['split']])
split_scaler_gui_one = pd.DataFrame(split_scaler_gui_one,columns=['split'])

sum = split_scaler_gui_one + split_scaler_gui_one
sum = pd.concat([sum, imp_gain.feat],axis=1)

sum_rank = sum.sort_values(by=['split'], ascending=False)
sum_rank = sum_rank[:250]



scores = []
imp = pd.DataFrame()
imp['feat'] = used_feat

params = {
    'learning_rate': 0.11642644888214702,
    'metric': 'multi_error',
    'objective': 'multiclass',
    'num_class': 19,
    'feature_fraction': 0.7280260366928102,
    'bagging_fraction': 0.8106130717132529,
    'bagging_freq': 5,
    'n_jobs': 4,
    'seed': 2020,
    'max_depth': 27,# 最大深度
    'num_leaves': 90,# 叶子节点数目，小于2**max_depth
    'lambda_l1': 0.7778891428757128,
    'lambda_l2': 0.8715624879670723,
}


train_x = x[used_feat]# 划分训练集
train_y = y
test_x = test_feature[used_feat]
# 学习lv,误差，分类类别，类数，每次迭代选择80%的参数进行建模（boosting为随机森林的时候使用），每次迭代时的数据比例（加快训练速度和减少过拟合），树深，cpu，l1正则，12正则
# 'num_leaves': 64,# 叶子节点数目，小于2**max_depth， 'max_depth': 10,# 最大深度
oof_train = np.zeros((len(train_x), 19))
preds = np.zeros((len(test_x), 19))
folds = 5 # 5折交叉验证
# seeds = [44]#, 2020, 527, 1527]
seed = np.random.randint(0,2020)
# for seed in seeds:

kfold = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed) # 5折交叉验证，随机因子44
for fold, (trn_idx, val_idx) in enumerate(kfold.split(train_x, train_y)):# 生成索引 将数据划分为训练集和测试集
    x_trn, y_trn, x_val, y_val = train_x.iloc[trn_idx], train_y.iloc[trn_idx], train_x.iloc[val_idx], train_y.iloc[val_idx]# 使用索引进行划分
    train_set = lgb.Dataset(x_trn, y_trn)# lightgbm和xgboost一样，x和y在一起
    val_set = lgb.Dataset(x_val, y_val)

    model = lgb.train(params, train_set, num_boost_round=1500,
                      valid_sets=(train_set, val_set), early_stopping_rounds=100,
                      verbose_eval=100)
    # 训练模型  
    oof_train[val_idx] += model.predict(x_val, num_iteration=model.best_iteration) 
#     / len(seeds)
    preds += model.predict(test_x, num_iteration=model.best_iteration) / folds
#     / len(seeds)
    # scores.append(model.best_score['valid_1']['multi_error'])
    # 垃圾回收机制
    del x_trn, y_trn, x_val, y_val,train_set,val_set
    gc.collect()
print('随机因子%d综合分数%0.4f%%'%(seed,(1-np.mean(scores))*100))

valid_labels = np.argmax(oof_train, axis=1) 
print('准确率：', round(accuracy_score(y, valid_labels), 5))

np.save('./npy_file/lgb_valid.npy', oof_train)

np.save('./npy_file/lgb_test.npy', preds)