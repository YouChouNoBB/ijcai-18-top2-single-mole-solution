# coding=utf-8
# @author:bryan
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from scipy import sparse
from sklearn.model_selection import train_test_split

def LR_test(train_x,train_y,test_x,test_y):
    from sklearn.metrics import log_loss
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=0.06, random_state=None,
                            solver='liblinear', max_iter=100,
                            verbose=1)
    lr.fit(train_x,train_y)
    predict = lr.predict_proba(test_x)[:, 1]
    print(log_loss(test_y,predict))


def LGB_test(train_x,train_y,test_x,test_y,cate_col=None):
    if cate_col:
        data = pd.concat([train_x, test_x])
        for fea in cate_col:
            data[fea]=data[fea].fillna('-1')
            data[fea] = LabelEncoder().fit_transform(data[fea].apply(str))
        train_x=data[:len(train_x)]
        test_x=data[len(train_x):]
    print("LGB test")
    clf = lgb.LGBMClassifier(
        boosting_type='gbdt', num_leaves=31, reg_alpha=0.0, reg_lambda=1,
        max_depth=-1, n_estimators=3000, objective='binary',
        subsample=0.7, colsample_bytree=0.7, subsample_freq=1,  # colsample_bylevel=0.7,
        learning_rate=0.01, min_child_weight=25,random_state=2018,n_jobs=50
    )
    clf.fit(train_x, train_y,eval_set=[(train_x,train_y),(test_x,test_y)],early_stopping_rounds=100)
    feature_importances=sorted(zip(train_x.columns,clf.feature_importances_),key=lambda x:x[1])
    return clf.best_score_[ 'valid_1']['binary_logloss'],feature_importances


def LGB_predict(data,file):
    import math
    data=data.drop(['hour48','hour', 'user_id', 'shop_id','query1','query',
               'item_property_list', 'context_id', 'context_timestamp', 'predict_category_property'], axis=1)
    data['item_category_list'] = LabelEncoder().fit_transform(data['item_category_list'])
    train=data[data['is_trade']>-1]
    predict=data[data['is_trade']==-2]
    res=predict[['instance_id']]
    train_y=train.pop('is_trade')
    train_x=train.drop(['day','instance_id'], axis=1)
    test_x = predict.drop(['day', 'instance_id','is_trade'], axis=1)
    clf = lgb.LGBMClassifier(
        boosting_type='gbdt', num_leaves=31, reg_alpha=0.0, reg_lambda=1,
        max_depth=-1, n_estimators=3000, objective='binary',
        subsample=0.7, colsample_bytree=0.7, subsample_freq=1,  # colsample_bylevel=0.7,
        learning_rate=0.01, min_child_weight=25, random_state=2018, n_jobs=50
    )
    clf.fit(train_x, train_y, eval_set=[(train_x, train_y)])
    res['predicted_score']=clf.predict_proba(test_x)[:,1]
    testb = pd.read_csv('../data/round2_ijcai_18_test_b_20180510.txt', sep=' ')[['instance_id']]
    res=pd.merge(testb,res,on='instance_id',how='left')
    res.to_csv('../submit/' + file + '.txt', sep=' ', index=False)


"""随机划分15%作为测试"""
def off_test_split(org,cate_col=None):
    data = org[org.is_trade >-1]
    data = data.drop(
        ['hour48', 'hour',  'user_id','query1','query',
         'instance_id', 'item_property_list', 'context_id', 'context_timestamp', 'predict_category_property'], axis=1)
    data['item_category_list'] = LabelEncoder().fit_transform(data['item_category_list'])
    y = data.pop('is_trade')
    train_x, test_x, train_y, test_y = train_test_split(data, y, test_size=0.15, random_state=2018)
    train_x.drop('day', axis=1, inplace=True)
    test_x.drop('day', axis=1, inplace=True)
    score = LGB_test(train_x, train_y, test_x, test_y,cate_col)
    return score

"""取最后2个小时作为测试"""
def off_test_2hour(org,cate_col=None):
    data = org[org.is_trade >-1]
    data = data.drop(
        ['hour48', 'user_id','query1','query',
         'instance_id', 'item_property_list', 'context_id', 'context_timestamp', 'predict_category_property'], axis=1)
    # data = data.drop(
    #     ['hour48', 'shop_score_delivery', 'shop_star_level', 'user_id', 'shop_review_num_level', 'shop_id',
    #      'instance_id', 'item_property_list', 'context_id', 'context_timestamp', 'predict_category_property'], axis=1)
    data['item_category_list'] = LabelEncoder().fit_transform(data['item_category_list'])
    train=data[data.hour<10]
    test=data[data.hour>=10]
    train=train.drop(['hour','day'],axis=1)
    test = test.drop(['hour','day'], axis=1)
    train_y=train.pop('is_trade')
    test_y=test.pop('is_trade')
    train_x=train
    test_x=test
    score = LGB_test(train_x, train_y, test_x, test_y,cate_col)
    return score


"""合并多部分特征，f1为train,f2为其他特征的集合"""
def add(f1,f2):
    for i in f2:
        f1=pd.merge(f1,i,on='instance_id',how='left')
    return f1

"""测试两种划分方式"""
"""返回验证分数，和base提升分数，特征重要性"""
def test(org,cate_col=None):
    s1=off_test_split(org,cate_col)
    s2=off_test_2hour(org,cate_col)
    def f(x):
        return float('%.6f' % x)
    print(f(s1[0]),f(s2[0]))
    print(f(0.175994-s1[0]),f(0.16987-s2[0]))
    print(s1[1])
    print(s2[1])


