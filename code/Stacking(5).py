import numpy as np
from sklearn import linear_model
from sklearn.model_selection import RepeatedKFold
import pandas as pd
from tensorflow.keras.utils import to_categorical
import lightgbm as lgb
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

SEED =2031
import random
from tensorflow import random
import os
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['PYTHONHASHSEED']=str(SEED)
np.random.seed(SEED)
random.set_seed(SEED)


lstm_cnn_valid = np.load('./npy_file/lstm_valid.npy')
lstm_cnn_test = np.load('./npy_file/lstm_test.npy')

cnn_valid = np.load('./npy_file/CNN_valid.npy')
cnn_test = np.load('./npy_file/CNN_test.npy')

lgb_valid = np.load('./npy_file/lgb_valid.npy')
lgb_test = np.load('./npy_file/lgb_test.npy')

oneD_CNN_valid = np.load('./npy_file/oneD_CNN_valid.npy')
oneD_CNN_test = np.load('./npy_file/oneD_CNN_test.npy')


# oneD_cnn_valid = np.load('./npy_file/oneD_cnn_valid.npy')
# oneD_cnn_test = np.load('./npy_file/oneD_cnn_test.npy')


Model = {'RidgeClassifier': RidgeClassifier(), 'SVC':SVC(), 'KNN':KNeighborsClassifier(), 'RF':RandomForestClassifier(),
       'LR':LogisticRegression(), 'Bayes':MultinomialNB()}


# 将lgb和xgb和ctb的结果进行stacking
train_stack = np.hstack([lstm_cnn_valid, cnn_valid, lgb_valid, oneD_CNN_valid])
test_stack = np.hstack([lstm_cnn_test, cnn_test, lgb_test, oneD_CNN_test])

folds_stack = RepeatedKFold(n_splits=10, n_repeats=3, random_state=2031)
oof_stack = np.zeros((train_stack.shape[0],))

predictions = np.zeros((test_stack.shape[0],))

valid=[]
test=[]

train = pd.read_csv('./data/sensor_train.csv')
y_train = train.groupby('fragment_id')['behavior_id'].min()

# y_train = to_categorical(y_train, num_classes=19)

vote=[]

for name, md in Model.items():
    for fold_, (trn_idx, val_idx) in enumerate(folds_stack.split(train_stack,y_train)):
        trn_data, trn_y = train_stack[trn_idx], y_train[trn_idx]
        val_data, val_y = train_stack[val_idx], y_train[val_idx]

        model = md
        model.fit(trn_data, trn_y)
        oof_stack[val_idx] = model.predict(val_data)

        predictions = model.predict(test_stack)
        test.append(predictions)
        if fold_==9 or fold_==19 or fold_==29:
            valid.append(oof_stack)

    valid_df1 = pd.DataFrame()
    for i in valid:
        valid_df = pd.DataFrame(i)
        valid_df1 = pd.concat([valid_df, valid_df1], axis=1)

    valid_vote = valid_df1.apply(lambda x: x.value_counts().index[0], axis=1)
    print(name, '准确率：', round(accuracy_score(valid_vote, y_train.values), 5))
    valid=[]

    test_df1 = pd.DataFrame()
    for i in test:
        test_df = pd.DataFrame(i)
        test_df1 = pd.concat([test_df, test_df1], axis=1)

    vote_tmp = test_df1.apply(lambda x: x.value_counts().index[0], axis=1)
    vote.append(vote_tmp)
    test = []

################### lgb ############################
folds_stack = RepeatedKFold(n_splits=10, n_repeats=3, random_state=2020)
oof_stack = np.zeros((train_stack.shape[0],19))

predictions = np.zeros((test_stack.shape[0],19))

test=[]
params = {
    'learning_rate': 0.0530273007911208,
    'metric': 'multi_error',
    'objective': 'multiclass',
    'num_class': 19,
    'feature_fraction': 0.6445972781422298,
    'bagging_fraction': 0.8544485598507204,
    'bagging_freq': 2,
    'n_jobs': 4,
    'seed': 2020,
    'max_depth': 32,
    'num_leaves': 103,
    'lambda_l1': 0.7157186720503712,
    'lambda_l2': 0.9895439677729682,
}


for fold_, (trn_idx, val_idx) in enumerate(folds_stack.split(train_stack,y_train)):
    print("fold {}".format(fold_))
    trn_data, trn_y = train_stack[trn_idx], y_train[trn_idx]
    val_data, val_y = train_stack[val_idx], y_train[val_idx]

    train_set = lgb.Dataset(trn_data, trn_y)
    val_set = lgb.Dataset(val_data, val_y)

    model = lgb.train(params, train_set, num_boost_round=1400, valid_sets=(train_set, val_set),early_stopping_rounds=100, verbose_eval=100)

    oof_stack[val_idx] = model.predict(val_data, num_iteration=model.best_iteration)/3

    predictions += model.predict(test_stack, num_iteration=model.best_iteration)/30

vote_res_df = pd.DataFrame()
for i in vote[:]:
    vote_tmp = pd.DataFrame(i.values)
    vote_res_df = pd.concat([vote_res_df, vote_tmp], axis=1)
pres = np.argmax(predictions, axis=1)
vote_lgb = pd.DataFrame(pres)
vote_res_df = pd.concat([vote_res_df, vote_lgb], axis=1)
vote_last = vote_res_df.apply(lambda x: x.value_counts().index[0], axis=1)
sub = pd.read_csv('./data/result.csv')
sub.behavior_id = vote_last 
sub.to_csv('./submit.csv', index=False)