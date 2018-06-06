# coding=utf-8
# @author: bryan

import pandas as pd

"""
7号之前所有天的统计特征
用户/商品/品牌/店铺/类别/城市/page/query 点击次数，购买次数，转化率(buy/cnt+3)

"""
def all_days_feature(org):
    data=org[org['day']<7]
    col=['user_id','item_id','item_brand_id','shop_id','item_category_list','item_city_id','query1','query','context_page_id','predict_category_property']
    train=org[org['day']==7][['instance_id']+col]
    user=data.groupby('user_id',as_index=False)['is_trade'].agg({'user_buy':'sum','user_cnt':'count'})
    user['user_7days_cvr']=(user['user_buy'])/(user['user_cnt']+3)
    items=col[1:]
    train=pd.merge(train,user[['user_id','user_7days_cvr']],on='user_id',how='left')
    for item in items:
        tmp=data.groupby(item,as_index=False)['is_trade'].agg({item+'_buy':'sum',item+'_cnt':'count'})
        tmp[item+'_7days_cvr'] = tmp[item+'_buy'] / tmp[item+'_cnt']
        train = pd.merge(train, tmp[[item, item+'_7days_cvr']], on=item, how='left')
        print(item)
    for i in range(len(items)):
        for j in range(i+1,len(items)):
            egg=[items[i],items[j]]
            tmp = data.groupby(egg, as_index=False)['is_trade'].agg({'_'.join(egg) + '_buy': 'sum', '_'.join(egg) + '_cnt': 'count'})
            tmp['_'.join(egg) + '_7days_cvr'] = tmp['_'.join(egg) + '_buy'] / tmp['_'.join(egg) + '_cnt']
            train = pd.merge(train, tmp[egg+['_'.join(egg) + '_7days_cvr']], on=egg, how='left')
            print(egg)
    train.drop(col, axis=1).to_csv('../data/7days_cvr_feature.csv',index=False)
    return train


"""
7号前一天，6号的统计特征
用户/商品/品牌/店铺/类别/城市 点击次数，购买次数，转化率，占前面所有天的占比

"""
def latest_day_feature(org):
    data = org[org['day'] ==6]
    col = ['user_id', 'item_id', 'item_brand_id', 'shop_id', 'item_category_list', 'item_city_id', 'query1', 'query','context_page_id','predict_category_property']
    train = org[org['day'] == 7][['instance_id'] + col]
    user = data.groupby('user_id', as_index=False)['is_trade'].agg({'user_buy': 'sum', 'user_cnt': 'count'})
    user['user_6day_cvr'] = (user['user_buy']) / (user['user_cnt'] + 3)
    train = pd.merge(train, user[['user_id', 'user_6day_cvr']], on='user_id', how='left')
    items = col[1:]
    for item in items:
        tmp=data.groupby(item,as_index=False)['is_trade'].agg({item+'_buy':'sum',item+'_cnt':'count'})
        tmp[item+'_6day_cvr'] = tmp[item+'_buy'] / tmp[item+'_cnt']
        train = pd.merge(train, tmp[[item, item+'_6day_cvr']], on=item, how='left')
        print(item)
    for i in range(len(items)):
        for j in range(i+1,len(items)):
            egg=[items[i],items[j]]
            tmp = data.groupby(egg, as_index=False)['is_trade'].agg({'_'.join(egg) + '_buy': 'sum', '_'.join(egg) + '_cnt': 'count'})
            tmp['_'.join(egg) + '_6day_cvr'] = tmp['_'.join(egg) + '_buy'] / tmp['_'.join(egg) + '_cnt']
            train = pd.merge(train, tmp[egg+['_'.join(egg) + '_6day_cvr']], on=egg, how='left')
            print(egg)
    train.drop(col, axis=1).to_csv('../data/6day_cvr_feature.csv',index=False)
    return train

"""
当天的交易率特征，交叉统计
"""

# calc data，join data
# user_id,item_id,item_brand_id,shop_id,item_category_list,item_city_id,predict_category_property
def cvr(c_data, j_data):
    col=['user_id','item_id','item_brand_id','shop_id','item_category_list','item_city_id','predict_category_property','context_page_id', 'query1', 'query']
    j_data=j_data[['instance_id']+col]
    user = c_data.groupby('user_id', as_index=False)['is_trade'].agg({'user_buy': 'sum', 'user_cnt': 'count'})
    user['user_today_cvr'] = (user['user_buy']) / (user['user_cnt'] + 3)
    j_data = pd.merge(j_data, user[['user_id', 'user_today_cvr']], on='user_id', how='left')
    for item in col[1:]:
        tmp=c_data.groupby(item, as_index=False)['is_trade'].agg({item+'_today_cvr': 'mean'})
        j_data = pd.merge(j_data, tmp, on=item, how='left')
    for i in range(len(col)):
        for j in range(i+1,len(col)):
            tmp=c_data.groupby([col[i],col[j]], as_index=False)['is_trade'].agg({'today_'+col[i]+col[j]+'_cvr': 'mean'})
            j_data = pd.merge(j_data, tmp, on=[col[i],col[j]], how='left')
            print([col[i],col[j]])
    return j_data
# [['instance_id','today_user_cvr','today_item_cvr','today_brand_cvr','today_shop_cvr','today_cate_cvr','today_city_cvr']]

def split(data, index, size):
    import math
    size = math.ceil(len(data) / size)
    start = size * index
    end = (index + 1) * size if (index + 1) * size < len(data) else len(data)
    return data[start:end]

def today_cvr_feature(org):
    col = ['user_id', 'item_id', 'item_brand_id', 'shop_id', 'item_category_list', 'item_city_id',
           'predict_category_property', 'context_page_id', 'query1', 'query']
    data=org[org['day']==7]
    train=data[data['is_trade']>-1]
    predict=data[data['is_trade']<0]
    predict=cvr(train,predict)
    trains=[]
    size=10
    for i in range(size):
        trains.append(split(train, i, size))
    res=[]
    res.append(predict)
    for i in range(size):
        res.append(cvr(pd.concat([trains[j] for j in range(size) if i !=j]).reset_index(drop=True),trains[i]))
    data=pd.concat(res).reset_index(drop=True)
    #data=data[['instance_id','today_user_cvr','today_item_cvr','today_brand_cvr','today_shop_cvr','today_cate_cvr','today_city_cvr','today_query_cvr']]
    data=data.drop(col,axis=1)
    data.to_csv('../data/today_cvr_feature.csv', index=False)
    return data

"""
#todo
排名特征,前7天的算一次，第7天的算一次
用户转化率在品牌，店铺，类别，城市下面的排名

商品转化率在店铺下面的排名
商品转化率在品牌下面的排名
商品转化率在类别下面的排名
商品转化率在城市下面的排名
商品转化率在query1下面的排名
商品转化率在query下面的排名

店铺转化率在品牌下面的排名
店铺转化率在城市下面的排名
店铺转化率在类别下面的排名
店铺转化率在query1下面的排名
店铺转化率在query下面的排名

品牌在城市下面的转化率排名
品牌在店铺下面的转化率排名
品牌转化率在query1下面的排名
品牌转化率在query下面的排名

类别在城市下面的转换率排名
类别在店铺下面的转换率排名
"""
# ['user_id','item_id','item_brand_id','shop_id','item_category_list','item_city_id','predict_category_property','context_page_id', 'query1', 'query']
def rank_6day_feature(data):
    data['user_cvr_brand_6day_rank']=data.groupby('item_brand_id')['user_6day_cvr'].rank(ascending=False,method='dense')
    data['user_cvr_shop_6day_rank'] = data.groupby('shop_id')['user_6day_cvr'].rank(ascending=False, method='dense')
    data['user_cvr_cate_6day_rank'] = data.groupby('item_category_list')['user_6day_cvr'].rank(ascending=False, method='dense')
    data['user_cvr_city_6day_rank'] = data.groupby('item_city_id')['user_6day_cvr'].rank(ascending=False, method='dense')
    data['item_cvr_shop_6day_rank'] = data.groupby('shop_id')['item_id_6day_cvr'].rank(ascending=False, method='dense')
    data['item_cvr_brand_6day_rank'] = data.groupby('item_brand_id')['item_id_6day_cvr'].rank(ascending=False, method='dense')
    data['item_cvr_cate_6day_rank'] = data.groupby('item_category_list')['item_id_6day_cvr'].rank(ascending=False, method='dense')
    data['item_cvr_city_6day_rank'] = data.groupby('item_city_id')['item_id_6day_cvr'].rank(ascending=False, method='dense')
    data['shop_cvr_brand_6day_rank'] = data.groupby('item_brand_id')['shop_id_6day_cvr'].rank(ascending=False, method='dense')
    data['shop_cvr_cate_6day_rank'] = data.groupby('item_category_list')['shop_id_6day_cvr'].rank(ascending=False, method='dense')
    data['shop_cvr_city_6day_rank'] = data.groupby('item_city_id')['shop_id_6day_cvr'].rank(ascending=False, method='dense')
    data['brand_cvr_city_6day_rank'] = data.groupby('item_city_id')['item_brand_id_6day_cvr'].rank(ascending=False, method='dense')
    data['brand_cvr_shop_6day_rank'] = data.groupby('shop_id')['item_brand_id_6day_cvr'].rank(ascending=False, method='dense')
    data['cate_cvr_city_6day_rank'] = data.groupby('item_city_id')['item_category_list_6day_cvr'].rank(ascending=False, method='dense')
    data['cate_cvr_shop_6day_rank'] = data.groupby('shop_id')['item_category_list_6day_cvr'].rank(ascending=False, method='dense')
    data['item_cvr_query_6day_rank'] = data.groupby('query')['item_id_6day_cvr'].rank(ascending=False, method='dense')
    data['item_cvr_query1_6day_rank'] = data.groupby('query1')['item_id_6day_cvr'].rank(ascending=False, method='dense')
    data['shop_cvr_query_6day_rank'] = data.groupby('query')['shop_id_6day_cvr'].rank(ascending=False, method='dense')
    data['shop_cvr_query1_6day_rank'] = data.groupby('query1')['shop_id_6day_cvr'].rank(ascending=False, method='dense')
    data['brand_cvr_query_6day_rank'] = data.groupby('query')['item_brand_id_6day_cvr'].rank(ascending=False, method='dense')
    data['brand_cvr_query1_6day_rank'] = data.groupby('query1')['item_brand_id_6day_cvr'].rank(ascending=False, method='dense')
    data=data[['instance_id','user_cvr_brand_6day_rank','user_cvr_shop_6day_rank','user_cvr_cate_6day_rank','user_cvr_city_6day_rank','item_cvr_shop_6day_rank','item_cvr_brand_6day_rank','item_cvr_cate_6day_rank','item_cvr_city_6day_rank','shop_cvr_brand_6day_rank','shop_cvr_cate_6day_rank','shop_cvr_city_6day_rank','brand_cvr_city_6day_rank','brand_cvr_shop_6day_rank','cate_cvr_city_6day_rank','cate_cvr_shop_6day_rank','item_cvr_query_6day_rank','item_cvr_query1_6day_rank','shop_cvr_query_6day_rank','shop_cvr_query1_6day_rank','brand_cvr_query_6day_rank','brand_cvr_query1_6day_rank'
    ]]
    data.to_csv('../data/rank_feature_6day.csv',index=False)

def rank_7days_feature(data):
    data['user_cvr_brand_7days_rank']=data.groupby('item_brand_id')['user_7days_cvr'].rank(ascending=False,method='dense')
    data['user_cvr_shop_7days_rank'] = data.groupby('shop_id')['user_7days_cvr'].rank(ascending=False, method='dense')
    data['user_cvr_cate_7days_rank'] = data.groupby('item_category_list')['user_7days_cvr'].rank(ascending=False, method='dense')
    data['user_cvr_city_7days_rank'] = data.groupby('item_city_id')['user_7days_cvr'].rank(ascending=False, method='dense')
    data['item_cvr_shop_7days_rank'] = data.groupby('shop_id')['item_id_7days_cvr'].rank(ascending=False, method='dense')
    data['item_cvr_brand_7days_rank'] = data.groupby('item_brand_id')['item_id_7days_cvr'].rank(ascending=False, method='dense')
    data['item_cvr_cate_7days_rank'] = data.groupby('item_category_list')['item_id_7days_cvr'].rank(ascending=False, method='dense')
    data['item_cvr_city_7days_rank'] = data.groupby('item_city_id')['item_id_7days_cvr'].rank(ascending=False, method='dense')
    data['shop_cvr_brand_7days_rank'] = data.groupby('item_brand_id')['shop_id_7days_cvr'].rank(ascending=False, method='dense')
    data['shop_cvr_cate_7days_rank'] = data.groupby('item_category_list')['shop_id_7days_cvr'].rank(ascending=False, method='dense')
    data['shop_cvr_city_7days_rank'] = data.groupby('item_city_id')['shop_id_7days_cvr'].rank(ascending=False, method='dense')
    data['brand_cvr_city_7days_rank'] = data.groupby('item_city_id')['item_brand_id_7days_cvr'].rank(ascending=False, method='dense')
    data['brand_cvr_shop_7days_rank'] = data.groupby('shop_id')['item_brand_id_7days_cvr'].rank(ascending=False, method='dense')
    data['cate_cvr_city_7days_rank'] = data.groupby('item_city_id')['item_category_list_7days_cvr'].rank(ascending=False, method='dense')
    data['cate_cvr_shop_7days_rank'] = data.groupby('shop_id')['item_category_list_7days_cvr'].rank(ascending=False, method='dense')
    data['item_cvr_query_7days_rank'] = data.groupby('query')['item_id_7days_cvr'].rank(ascending=False, method='dense')
    data['item_cvr_query1_7days_rank'] = data.groupby('query1')['item_id_7days_cvr'].rank(ascending=False, method='dense')
    data['shop_cvr_query_7days_rank'] = data.groupby('query')['shop_id_7days_cvr'].rank(ascending=False, method='dense')
    data['shop_cvr_query1_7days_rank'] = data.groupby('query1')['shop_id_7days_cvr'].rank(ascending=False, method='dense')
    data['brand_cvr_query_7days_rank'] = data.groupby('query')['item_brand_id_7days_cvr'].rank(ascending=False, method='dense')
    data['brand_cvr_query1_7days_rank'] = data.groupby('query1')['item_brand_id_7days_cvr'].rank(ascending=False, method='dense')
    data=data[['instance_id','user_cvr_brand_7days_rank','user_cvr_shop_7days_rank','user_cvr_cate_7days_rank','user_cvr_city_7days_rank','item_cvr_shop_7days_rank','item_cvr_brand_7days_rank','item_cvr_cate_7days_rank','item_cvr_city_7days_rank','shop_cvr_brand_7days_rank','shop_cvr_cate_7days_rank','shop_cvr_city_7days_rank','brand_cvr_city_7days_rank','brand_cvr_shop_7days_rank','cate_cvr_city_7days_rank','cate_cvr_shop_7days_rank','item_cvr_query_7days_rank','item_cvr_query1_7days_rank','shop_cvr_query_7days_rank','shop_cvr_query1_7days_rank','brand_cvr_query_7days_rank','brand_cvr_query1_7days_rank'
    ]]
    data.to_csv('../data/rank_feature_7days.csv',index=False)

def rank_today_feature(data):
    data=data.reset_index(drop=True)
    data['user_cvr_brand_today_rank']=data.groupby('item_brand_id')['user_today_cvr'].rank(ascending=False,method='dense')
    data['user_cvr_shop_today_rank'] = data.groupby('shop_id')['user_today_cvr'].rank(ascending=False, method='dense')
    data['user_cvr_cate_today_rank'] = data.groupby('item_category_list')['user_today_cvr'].rank(ascending=False, method='dense')
    data['user_cvr_city_today_rank'] = data.groupby('item_city_id')['user_today_cvr'].rank(ascending=False, method='dense')
    data['item_cvr_shop_today_rank'] = data.groupby('shop_id')['item_id_today_cvr'].rank(ascending=False, method='dense')
    data['item_cvr_brand_today_rank'] = data.groupby('item_brand_id')['item_id_today_cvr'].rank(ascending=False, method='dense')
    data['item_cvr_cate_today_rank'] = data.groupby('item_category_list')['item_id_today_cvr'].rank(ascending=False, method='dense')
    data['item_cvr_city_today_rank'] = data.groupby('item_city_id')['item_id_today_cvr'].rank(ascending=False, method='dense')
    data['shop_cvr_brand_today_rank'] = data.groupby('item_brand_id')['shop_id_today_cvr'].rank(ascending=False, method='dense')
    data['shop_cvr_cate_today_rank'] = data.groupby('item_category_list')['shop_id_today_cvr'].rank(ascending=False, method='dense')
    data['shop_cvr_city_today_rank'] = data.groupby('item_city_id')['shop_id_today_cvr'].rank(ascending=False, method='dense')
    data['brand_cvr_city_today_rank'] = data.groupby('item_city_id')['item_brand_id_today_cvr'].rank(ascending=False, method='dense')
    data['brand_cvr_shop_today_rank'] = data.groupby('shop_id')['item_brand_id_today_cvr'].rank(ascending=False, method='dense')
    data['cate_cvr_city_today_rank'] = data.groupby('item_city_id')['item_category_list_today_cvr'].rank(ascending=False, method='dense')
    data['cate_cvr_shop_today_rank'] = data.groupby('shop_id')['item_category_list_today_cvr'].rank(ascending=False, method='dense')
    data['item_cvr_query_today_rank'] = data.groupby('query')['item_id_today_cvr'].rank(ascending=False, method='dense')
    data['item_cvr_query1_today_rank'] = data.groupby('query1')['item_id_today_cvr'].rank(ascending=False, method='dense')
    data['shop_cvr_query_today_rank'] = data.groupby('query')['shop_id_today_cvr'].rank(ascending=False, method='dense')
    data['shop_cvr_query1_today_rank'] = data.groupby('query1')['shop_id_today_cvr'].rank(ascending=False, method='dense')
    data['brand_cvr_query_today_rank'] = data.groupby('query')['item_brand_id_today_cvr'].rank(ascending=False, method='dense')
    data['brand_cvr_query1_today_rank'] = data.groupby('query1')['item_brand_id_today_cvr'].rank(ascending=False, method='dense')
    data=data[['instance_id','user_cvr_brand_today_rank','user_cvr_shop_today_rank','user_cvr_cate_today_rank','user_cvr_city_today_rank','item_cvr_shop_today_rank','item_cvr_brand_today_rank','item_cvr_cate_today_rank','item_cvr_city_today_rank','shop_cvr_brand_today_rank','shop_cvr_cate_today_rank','shop_cvr_city_today_rank','brand_cvr_city_today_rank','brand_cvr_shop_today_rank','cate_cvr_city_today_rank','cate_cvr_shop_today_rank','item_cvr_query_today_rank','item_cvr_query1_today_rank','shop_cvr_query_today_rank','shop_cvr_query1_today_rank','brand_cvr_query_today_rank','brand_cvr_query1_today_rank'
    ]]
    data.to_csv('../data/rank_feature_today.csv',index=False)

if __name__ == '__main__':
    org=pd.read_csv('../data/origion_concat.csv')
    rank_7days_feature(all_days_feature(org))
    rank_6day_feature(latest_day_feature(org))
    rank_today_feature(today_cvr_feature(org))