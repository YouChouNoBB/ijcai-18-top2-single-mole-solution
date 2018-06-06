# coding=utf-8
# @author:bryan
"""
top100的特征强制相除交叉
"""
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import numpy as np

cross_feature_num=100

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
    return score[1]

def LGB_predict(data,file):
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
    res[['instance_id', 'predicted_score']].to_csv('../submit/' + file + '.txt', sep=' ', index=False)

def add(f1,f2):
    for i in f2:
        f1=pd.merge(f1,i,on='instance_id',how='left')
    return f1

if __name__ == '__main__':
    org=pd.read_csv('../data/origion_concat.csv')
    train=org[org.day==7]
    query = pd.read_csv('../data/query_all.csv')
    leak = pd.read_csv('../data/leak_all.csv')
    comp = pd.read_csv('../data/compare_all.csv')
    day6_cvr = pd.read_csv('../data/6day_cvr_feature.csv')
    days7_cvr = pd.read_csv('../data/7days_cvr_feature.csv')
    day6_rank = pd.read_csv('../data/rank_feature_6day.csv')
    days7_rank = pd.read_csv('../data/rank_feature_7days.csv')
    nobuy = pd.read_csv('../data/nobuy_feature.csv')
    trend = pd.read_csv('../data/trend_feature.csv')
    trend = trend[[i for i in trend.columns if 'cnt6' not in i]]
    var = pd.read_csv('../data/item_shop_var_feature.csv')
    user_buy_click = pd.read_csv('../data/user_buy_click_feature.csv') #need proc caterory feature
    property = pd.read_csv('../data/property_feature.csv') #need proc caterory feature
    full = pd.read_csv('../data/full_count_feature.csv')
    day6 = pd.read_csv('../data/day6_count_feature.csv')
    days7 = pd.read_csv('../data/days7_count_feature.csv')
    # user_buy_click,property need proc caterory feature
    data = add(train,[query, leak, day6_cvr, days7_cvr,
        day6_rank, days7_rank, comp, nobuy, trend, full,day6, days7, var
        ])
    data.to_csv('../data/final_base.csv',index=False)
    base = pd.read_csv('../data/final_base.csv')
    cross=base[['hour48', 'hour',  'user_id','query1','query','is_trade','day','item_category_list',
             'instance_id', 'item_property_list', 'context_id', 'context_timestamp', 'predict_category_property']]
    features=off_test_split(base)
    feature=[i[0] for i in features[-cross_feature_num:]]
    feature.remove('shop_id')
    feature.remove('item_id')
    # shop_id,item_id
    for i in range(len(feature)):
        for j in range(i+1,len(feature)):
            cross['cross_'+str(i)+'_'+str(j)]=base[feature[i]]/base[feature[j]]

    score=off_test_split(cross)
    add_feature=[i[0] for i in score[-cross_feature_num:]]
    base_add=pd.concat([base,cross[add_feature]],axis=1)
    LGB_predict(base_add,'bryan_submit')