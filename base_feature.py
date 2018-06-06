# coding=utf-8
# @author:bryan
from multiprocessing import Pool
from multiprocessing import cpu_count
import math
import pandas as pd
import datetime
import numpy as np
import gc

processor=cpu_count()-2

"""
concate feature
"""
def concate_feature(data):
    features=[('user_occupation_id','shop_star_level'),('item_collected_level','item_pv_level'),('user_star_level','hour48'),('item_price_level','hour48'),('item_sales_level','context_page_id')]
    con_fea=[]
    def concate_feature(data, f1, f2, name):
        data[name] = data.apply(lambda x: str(x[f1]) + ';' + str(x[f2]), axis=1)
        return data
    for i in features:
        data=concate_feature(data, i[0], i[1], '_con_'.join(i))
        con_fea.append('_con_'.join(i))
    return data,con_fea


"""
    query特征,之前，之后有几次相同的query
    相同query，相同item，之前之后有多少个
    相同query,相同shop,之前之后个数
    相同query,相同brand,之前之后个数
    相同query,相同city,之前之后个数
    cate,page
    这个query之前之后是否搜过其他商品
    当前query之前之后点击了几个query
    """
def run_query_feature(i):
    data=pd.read_csv('../data/user_data/query_'+str(i)+'.csv')
    features=[]
    for index,row in data.iterrows():
        feature={}
        feature['instance_id']=row['instance_id']
        if index%100==0:
            print(index)
        col=['user_id','predict_category_property','context_timestamp','day','query1','query','item_id','shop_id','item_brand_id','item_city_id','context_page_id','item_category_list']
        tmp=data[data['user_id']==row['user_id']][['instance_id']+col]
        before_query_cnt=len(tmp[(tmp['predict_category_property']==row['predict_category_property'])& (tmp['context_timestamp']<row['context_timestamp'])&(tmp['day']<=row['day'])])
        before_query_1_cnt = len(tmp[(tmp['query1'] == row['query1']) & (tmp['context_timestamp'] < row['context_timestamp']) & (tmp['day'] <= row['day'])])
        before_query_all_cnt = len(tmp[(tmp['query'] == row['query']) & (tmp['context_timestamp'] < row['context_timestamp']) & (tmp['day'] <= row['day'])])
        after_query_cnt = len(tmp[(tmp['predict_category_property'] == row['predict_category_property']) & (tmp['context_timestamp'] > row['context_timestamp'])&(tmp['day']<=row['day'])])
        after_query_1_cnt = len(tmp[(tmp['query1'] == row['query1']) & (tmp['context_timestamp'] > row['context_timestamp']) & (tmp['day'] <= row['day'])])
        after_query_all_cnt = len(tmp[(tmp['query'] == row['query']) & (tmp['context_timestamp'] > row['context_timestamp']) & (tmp['day'] <= row['day'])])
        before_query_item_cnt=len(tmp[(tmp['item_id']==row['item_id'])&(tmp['predict_category_property']==row['predict_category_property'])& (tmp['context_timestamp']<row['context_timestamp'])&(tmp['day']<=row['day'])])
        before_query_1_item_cnt = len(tmp[(tmp['item_id'] == row['item_id']) & (tmp['query1'] == row['query1']) & (tmp['context_timestamp'] < row['context_timestamp']) & (tmp['day'] <= row['day'])])
        before_query_all_item_cnt = len(tmp[(tmp['item_id'] == row['item_id']) & (tmp['query'] == row['query']) & (tmp['context_timestamp'] < row['context_timestamp']) & (tmp['day'] <= row['day'])])
        after_query_item_cnt = len(tmp[(tmp['item_id'] == row['item_id']) & ( tmp['predict_category_property'] == row['predict_category_property']) & (tmp['context_timestamp'] > row['context_timestamp'])&(tmp['day']<=row['day'])])
        after_query_1_item_cnt = len(tmp[(tmp['item_id'] == row['item_id']) & (tmp['query1'] == row['query1']) & (tmp['context_timestamp'] > row['context_timestamp']) & (tmp['day'] <= row['day'])])
        after_query_all_item_cnt = len(tmp[(tmp['item_id'] == row['item_id']) & (tmp['query'] == row['query']) & (tmp['context_timestamp'] > row['context_timestamp']) & (tmp['day'] <= row['day'])])
        before_query_shop_cnt=len(tmp[(tmp['shop_id']==row['shop_id'])&(tmp['predict_category_property']==row['predict_category_property'])& (tmp['context_timestamp']<row['context_timestamp'])&(tmp['day']<=row['day'])])
        before_query_1_shop_cnt = len(tmp[(tmp['shop_id'] == row['shop_id']) & (tmp['query1'] == row['query1']) & (tmp['context_timestamp'] < row['context_timestamp']) & (tmp['day'] <= row['day'])])
        before_query_all_shop_cnt = len(tmp[(tmp['shop_id'] == row['shop_id']) & (tmp['query'] == row['query']) & (tmp['context_timestamp'] < row['context_timestamp']) & (tmp['day'] <= row['day'])])
        after_query_shop_cnt=len(tmp[(tmp['shop_id'] == row['shop_id']) & ( tmp['predict_category_property'] == row['predict_category_property']) & (tmp['context_timestamp'] > row['context_timestamp'])&(tmp['day']<=row['day'])])
        after_query_all_shop_cnt = len(tmp[(tmp['shop_id'] == row['shop_id']) & (tmp['query'] == row['query']) & (tmp['context_timestamp'] > row['context_timestamp']) & (tmp['day'] <= row['day'])])
        after_query_1_shop_cnt = len(tmp[(tmp['shop_id'] == row['shop_id']) & (tmp['query1'] == row['query1']) & (tmp['context_timestamp'] > row['context_timestamp']) & (tmp['day'] <= row['day'])])
        before_query_brand_cnt=len(tmp[(tmp['item_brand_id']==row['item_brand_id'])&(tmp['predict_category_property']==row['predict_category_property'])& (tmp['context_timestamp']<row['context_timestamp'])&(tmp['day']<=row['day'])])
        before_query_all_brand_cnt = len(tmp[(tmp['item_brand_id'] == row['item_brand_id']) & (tmp['query'] == row['query']) & (tmp['context_timestamp'] < row['context_timestamp']) & (tmp['day'] <= row['day'])])
        before_query_1_brand_cnt = len(tmp[(tmp['item_brand_id'] == row['item_brand_id']) & (tmp['query1'] == row['query1']) & (tmp['context_timestamp'] < row['context_timestamp']) & (tmp['day'] <= row['day'])])
        after_query_brand_cnt=len(tmp[(tmp['item_brand_id'] == row['item_brand_id']) & ( tmp['predict_category_property'] == row['predict_category_property']) & (tmp['context_timestamp'] > row['context_timestamp'])&(tmp['day']<=row['day'])])
        after_query_all_brand_cnt = len(tmp[(tmp['item_brand_id'] == row['item_brand_id']) & (tmp['query'] == row['query']) & (tmp['context_timestamp'] > row['context_timestamp']) & (tmp['day'] <= row['day'])])
        after_query_1_brand_cnt = len(tmp[(tmp['item_brand_id'] == row['item_brand_id']) & (tmp['query1'] == row['query1']) & (tmp['context_timestamp'] > row['context_timestamp']) & (tmp['day'] <= row['day'])])
        before_query_city_cnt = len(tmp[(tmp['item_city_id'] == row['item_city_id']) & (tmp['predict_category_property'] == row['predict_category_property']) & (tmp['context_timestamp'] < row['context_timestamp'])&(tmp['day']<=row['day'])])
        before_query_all_city_cnt = len(tmp[(tmp['item_city_id'] == row['item_city_id']) & (tmp['query'] == row['query']) & (tmp['context_timestamp'] < row['context_timestamp']) & (tmp['day'] <= row['day'])])
        before_query_1_city_cnt = len(tmp[(tmp['item_city_id'] == row['item_city_id']) & (tmp['query1'] == row['query1']) & (tmp['context_timestamp'] < row['context_timestamp']) & (tmp['day'] <= row['day'])])
        after_query_city_cnt = len(tmp[(tmp['item_city_id'] == row['item_city_id']) & (tmp['predict_category_property'] == row['predict_category_property']) & (tmp['context_timestamp'] > row['context_timestamp'])&(tmp['day']<=row['day'])])
        after_query_all_city_cnt = len(tmp[(tmp['item_city_id'] == row['item_city_id']) & (tmp['query'] == row['query']) & (tmp['context_timestamp'] > row['context_timestamp']) & (tmp['day'] <= row['day'])])
        after_query_1_city_cnt = len(tmp[(tmp['item_city_id'] == row['item_city_id']) & (tmp['query1'] == row['query1']) & (tmp['context_timestamp'] > row['context_timestamp']) & (tmp['day'] <= row['day'])])
        before_query_page_cnt = len(tmp[(tmp['context_page_id'] == row['context_page_id']) & (tmp['predict_category_property'] == row['predict_category_property']) & (tmp['context_timestamp'] < row['context_timestamp']) & (tmp['day'] <= row['day'])])
        before_query_1_page_cnt = len(tmp[(tmp['context_page_id'] == row['context_page_id']) & (tmp['query1'] == row['query1']) & (tmp['context_timestamp'] < row['context_timestamp']) & (tmp['day'] <= row['day'])])
        before_query_all_page_cnt = len(tmp[(tmp['context_page_id'] == row['context_page_id']) & (tmp['query'] == row['query']) & (tmp['context_timestamp'] < row['context_timestamp']) & (tmp['day'] <= row['day'])])
        after_query_page_cnt = len(tmp[(tmp['context_page_id'] == row['context_page_id']) & (tmp['predict_category_property'] == row['predict_category_property']) & (tmp['context_timestamp'] > row['context_timestamp']) & (tmp['day'] <= row['day'])])
        after_query_1_page_cnt = len(tmp[(tmp['context_page_id'] == row['context_page_id']) & (tmp['query1'] == row['query1']) & (tmp['context_timestamp'] > row['context_timestamp']) & (tmp['day'] <= row['day'])])
        after_query_all_page_cnt = len(tmp[(tmp['context_page_id'] == row['context_page_id']) & (tmp['query'] == row['query']) & (tmp['context_timestamp'] > row['context_timestamp']) & (tmp['day'] <= row['day'])])
        before_query_cate_cnt = len(tmp[(tmp['item_category_list'] == row['item_category_list']) & (tmp['predict_category_property'] == row['predict_category_property']) & (tmp['context_timestamp'] < row['context_timestamp']) & (tmp['day'] <= row['day'])])
        before_query_1_cate_cnt = len(tmp[(tmp['item_category_list'] == row['item_category_list']) & (tmp['query1'] == row['query1']) & (tmp['context_timestamp'] < row['context_timestamp']) & (tmp['day'] <= row['day'])])
        before_query_all_cate_cnt = len(tmp[(tmp['item_category_list'] == row['item_category_list']) & (tmp['query'] == row['query']) & (tmp['context_timestamp'] < row['context_timestamp']) & (tmp['day'] <= row['day'])])
        after_query_cate_cnt = len(tmp[(tmp['item_category_list'] == row['item_category_list']) & (tmp['predict_category_property'] == row['predict_category_property']) & (tmp['context_timestamp'] > row['context_timestamp']) & (tmp['day'] <= row['day'])])
        after_query_1_cate_cnt = len(tmp[(tmp['item_category_list'] == row['item_category_list']) & (tmp['query1'] == row['query1']) & (tmp['context_timestamp'] > row['context_timestamp']) & (tmp['day'] <= row['day'])])
        after_query_all_cate_cnt = len(tmp[(tmp['item_category_list'] == row['item_category_list']) & (tmp['query'] == row['query']) & (tmp['context_timestamp'] > row['context_timestamp']) & (tmp['day'] <= row['day'])])
        before_diff_query_cnt= len(set(tmp[(tmp['context_timestamp']<row['context_timestamp'])&(tmp['predict_category_property']!=row['predict_category_property'])]))
        before_diff_query_all_cnt = len(set(tmp[(tmp['context_timestamp'] < row['context_timestamp']) & (tmp['query'] != row['query'])]))
        before_diff_query_1_cnt = len(set(tmp[(tmp['context_timestamp'] < row['context_timestamp']) & (tmp['query1'] != row['query1'])]))
        after_diff_query_cnt = len(set(tmp[(tmp['context_timestamp'] > row['context_timestamp']) & (tmp['predict_category_property'] != row['predict_category_property'])]))
        after_diff_query_all_cnt = len(set(tmp[(tmp['context_timestamp'] > row['context_timestamp']) & (tmp['query'] != row['query'])]))
        after_diff_query_1_cnt = len(set(tmp[(tmp['context_timestamp'] > row['context_timestamp']) & (tmp['query1'] != row['query1'])]))
        query_min_time=np.min(tmp[(tmp['predict_category_property'] == row['predict_category_property'])]['context_timestamp'])
        query_all_min_time = np.min(tmp[(tmp['query'] == row['query'])]['context_timestamp'])
        query_1_min_time = np.min(tmp[(tmp['query1'] == row['query1'])]['context_timestamp'])
        before_query_items= len(set(tmp[(tmp['context_timestamp'] <query_min_time)]['item_id']))
        before_query_all_items = len(set(tmp[(tmp['context_timestamp'] < query_all_min_time)]['item_id']))
        before_query_1_items = len(set(tmp[(tmp['context_timestamp'] < query_1_min_time)]['item_id']))
        before_query_shops = len(set(tmp[(tmp['context_timestamp'] < query_min_time)]['shop_id']))
        before_query_all_shops = len(set(tmp[(tmp['context_timestamp'] < query_all_min_time)]['shop_id']))
        before_query_1_shops = len(set(tmp[(tmp['context_timestamp'] < query_1_min_time)]['shop_id']))
        query_max_time = np.max(tmp[(tmp['predict_category_property'] == row['predict_category_property'])]['context_timestamp'])
        query_all_max_time = np.max(tmp[(tmp['query'] == row['query'])]['context_timestamp'])
        query_1_max_time = np.max(tmp[(tmp['query1'] == row['query1'])]['context_timestamp'])
        after_query_items = len(set(tmp[(tmp['context_timestamp'] > query_max_time)]['item_id']))
        after_query_all_items = len(set(tmp[(tmp['context_timestamp'] > query_all_max_time)]['item_id']))
        after_query_1_items = len(set(tmp[(tmp['context_timestamp'] > query_1_max_time)]['item_id']))
        after_query_shops = len(set(tmp[(tmp['context_timestamp'] > query_max_time)]['shop_id']))
        after_query_all_shops = len(set(tmp[(tmp['context_timestamp'] > query_all_max_time)]['shop_id']))
        after_query_1_shops = len(set(tmp[(tmp['context_timestamp'] > query_1_max_time)]['shop_id']))
        feature['before_query_cnt'] = before_query_cnt
        feature['after_query_cnt'] = after_query_cnt
        feature['before_query_item_cnt'] = before_query_item_cnt
        feature['after_query_item_cnt'] = after_query_item_cnt
        feature['before_query_shop_cnt'] = before_query_shop_cnt
        feature['after_query_shop_cnt'] = after_query_shop_cnt
        feature['before_query_brand_cnt'] = before_query_brand_cnt
        feature['after_query_brand_cnt'] = after_query_brand_cnt
        feature['before_query_city_cnt'] = before_query_city_cnt
        feature['after_query_city_cnt'] = after_query_city_cnt
        feature['before_diff_query_cnt'] = before_diff_query_cnt
        feature['after_diff_query_cnt'] = after_diff_query_cnt
        feature['before_query_items'] = before_query_items
        feature['before_query_shops'] = before_query_shops
        feature['after_query_items'] = after_query_items
        feature['after_query_shops'] = after_query_shops
        feature['before_query_1_cnt'] = before_query_1_cnt
        feature['before_query_all_cnt'] = before_query_all_cnt
        feature['after_query_1_cnt'] = after_query_1_cnt
        feature['after_query_all_cnt'] = after_query_all_cnt
        feature['before_query_1_item_cnt'] = before_query_1_item_cnt
        feature['before_query_all_item_cnt'] = before_query_all_item_cnt
        feature['after_query_1_item_cnt'] = after_query_1_item_cnt
        feature['after_query_all_item_cnt'] = after_query_all_item_cnt
        feature['before_query_1_shop_cnt'] = before_query_1_shop_cnt
        feature['before_query_all_shop_cnt'] = before_query_all_shop_cnt
        feature['after_query_all_shop_cnt'] = after_query_all_shop_cnt
        feature['after_query_1_shop_cnt'] = after_query_1_shop_cnt
        feature['before_query_all_brand_cnt'] = before_query_all_brand_cnt
        feature['before_query_1_brand_cnt'] = before_query_1_brand_cnt
        feature['after_query_all_brand_cnt'] = after_query_all_brand_cnt
        feature['after_query_1_brand_cnt'] = after_query_1_brand_cnt
        feature['before_query_all_city_cnt'] = before_query_all_city_cnt
        feature['before_query_1_city_cnt'] = before_query_1_city_cnt
        feature['after_query_all_city_cnt'] = after_query_all_city_cnt
        feature['after_query_1_city_cnt'] = after_query_1_city_cnt
        feature['before_diff_query_all_cnt'] = before_diff_query_all_cnt
        feature['before_diff_query_1_cnt'] = before_diff_query_1_cnt
        feature['after_diff_query_all_cnt'] = after_diff_query_all_cnt
        feature['after_diff_query_1_cnt'] = after_diff_query_1_cnt
        feature['before_query_all_items'] = before_query_all_items
        feature['before_query_1_items'] = before_query_1_items
        feature['before_query_all_shops'] = before_query_all_shops
        feature['before_query_1_shops'] = before_query_1_shops
        feature['after_query_all_items'] = after_query_all_items
        feature['after_query_1_items'] = after_query_1_items
        feature['after_query_all_shops'] = after_query_all_shops
        feature['after_query_1_shops'] = after_query_1_shops
        feature['before_query_page_cnt'] = before_query_page_cnt
        feature['before_query_1_page_cnt'] = before_query_1_page_cnt
        feature['before_query_all_page_cnt'] = before_query_all_page_cnt
        feature['after_query_page_cnt'] = after_query_page_cnt
        feature['after_query_1_page_cnt'] = after_query_1_page_cnt
        feature['after_query_all_page_cnt'] = after_query_all_page_cnt
        feature['before_query_cate_cnt'] = before_query_cate_cnt
        feature['before_query_1_cate_cnt'] = before_query_1_cate_cnt
        feature['before_query_all_cate_cnt'] = before_query_all_cate_cnt
        feature['after_query_cate_cnt'] = after_query_cate_cnt
        feature['after_query_1_cate_cnt'] = after_query_1_cate_cnt
        feature['after_query_all_cate_cnt'] = after_query_all_cate_cnt
        features.append(feature)
    features=pd.DataFrame(features)
    print(str(i) + ' processor finished !')
    return features

def query_data_prepare():
    data=pd.read_csv('../data/origion_concat.csv')
    data=data[data.day>=6]
    data = data.sort_values(by=['user_id', 'context_timestamp']).reset_index(drop=True)
    users = pd.DataFrame(list(set(data['user_id'].values)), columns=['user_id'])
    l_data = len(users)
    size = math.ceil(l_data / processor)
    for i in range(processor):
        start = size * i
        end = (i + 1) * size if (i + 1) * size < l_data else l_data
        user = users[start:end]
        t_data = pd.merge(data, user, on='user_id').reset_index(drop=True)
        t_data.to_csv('../data/user_data/query_'+str(i)+'.csv',index=False)
        print(len(t_data))

def query_feature():
    res = []
    p = Pool(processor)
    for i in range(processor):
        res.append(p.apply_async(run_query_feature, args=( i,)))
        print(str(i) + ' processor started !')
    p.close()
    p.join()
    data=pd.concat([i.get() for i in res])
    data.to_csv('../data/query_all.csv',index=False)

"""
    最大最小点击间隔，平均点击间隔，只有一条数据算-1,上一个下一个间隔
    距离最前最后一次点击分钟数
    之前之后点击过多少query,item,shop,brand,city,query次数占比，item次数占比，shop,brand,city次数占比
    搜索这个商品,店铺，品牌，城市，用了几个query
    :param data:
    :return:
"""
def sec_diff(a,b):
    if (a is np.nan) | (b is np.nan):
        return -1
    return (datetime.datetime.strptime(str(b), "%Y-%m-%d %H:%M:%S")-datetime.datetime.strptime(str(a), "%Y-%m-%d %H:%M:%S")).seconds

def run_leak_feature( i):
    col = ['user_id', 'predict_category_property', 'context_timestamp', 'day', 'query1', 'query', 'item_id', 'shop_id',
           'item_brand_id', 'item_city_id', 'item_category_list']
    data = pd.read_csv('../data/user_data/query_' + str(i) + '.csv')[['instance_id']+col]
    features=[]
    for index, row in data.iterrows():
        feature={}
        feature['instance_id']=row['instance_id']
        if index%1000==0:
            print(index)
        tmp = data[(data['user_id'] == row['user_id'])&(data['day']==row['day'])]
        tmp=tmp.sort_values(by='context_timestamp').reset_index(drop=True)
        diffs=[]
        if len(tmp)==1:
            diffs.append(-1)
        else:
            for ind in range(len(tmp)-1):
                diffs.append(sec_diff(tmp.loc[ind+1,'context_timestamp'],tmp.loc[ind,'context_timestamp']))
        max_diff=np.max(diffs)
        min_diff=np.min(diffs)
        avg_diff=np.mean(diffs)
        mid_diff=np.median(diffs)
        diff_first_click=sec_diff(row['context_timestamp'],tmp.loc[0,'context_timestamp'])
        diff_last_click = sec_diff(row['context_timestamp'], tmp.loc[len(tmp)-1, 'context_timestamp'])
        previous_diff=sec_diff(row['context_timestamp'], np.max(tmp[(tmp['context_timestamp'] < row['context_timestamp'])]['context_timestamp']))
        next_diff=sec_diff( np.min(tmp[(tmp['context_timestamp'] > row['context_timestamp'])]['context_timestamp']),row['context_timestamp'])
        query_cnt=len(set(tmp['predict_category_property']))
        query_1_cnt = len(set(tmp['query1']))
        query_all_cnt = len(set(tmp['query']))
        item_cnt=len(set(tmp['item_id']))
        shop_cnt=len(set(tmp['shop_id']))
        brand_cnt=len(set(tmp['item_brand_id']))
        city_cnt=len(set(tmp['item_city_id']))
        before_query_rate=len(set(tmp[(tmp['context_timestamp']<=row['context_timestamp'])&(tmp['predict_category_property'] == row['predict_category_property'])]['predict_category_property']))/query_cnt
        after_query_rate=1-before_query_rate
        before_query_all_rate = len(set(tmp[(tmp['context_timestamp'] <= row['context_timestamp']) & (tmp['query'] == row['query'])]['query'])) / query_all_cnt
        after_query_all_rate = 1 - before_query_all_rate
        before_query_1_rate = len(set(tmp[(tmp['context_timestamp'] <= row['context_timestamp']) & (tmp['query1'] == row['query1'])]['query1'])) / query_1_cnt
        after_query_1_rate = 1 - before_query_1_rate
        before_item_rate=len(set(tmp[(tmp['context_timestamp']<=row['context_timestamp'])&(tmp['item_id'] == row['item_id'])]['item_id']))/item_cnt
        after_item_rate=1-before_item_rate
        before_shop_rate=len(set(tmp[(tmp['context_timestamp']<=row['context_timestamp'])&(tmp['shop_id'] == row['shop_id'])]['shop_id']))/shop_cnt
        after_shop_rate=1-before_shop_rate
        before_brand_rate=len(set(tmp[(tmp['context_timestamp']<=row['context_timestamp'])&(tmp['item_brand_id'] == row['item_brand_id'])]['item_brand_id']))/brand_cnt
        after_brand_rate=1-before_brand_rate
        before_city_rate=len(set(tmp[(tmp['context_timestamp']<=row['context_timestamp'])&(tmp['item_city_id'] == row['item_city_id'])]['item_city_id']))/city_cnt
        after_city_rate=1-before_city_rate
        before_item_query_cnt=len(set(tmp[(tmp['context_timestamp'] <= row['context_timestamp']) & (tmp['item_id'] == row['item_id'])]['predict_category_property']))
        before_item_query_all_cnt = len(set(tmp[(tmp['context_timestamp'] <= row['context_timestamp']) & (tmp['item_id'] == row['item_id'])]['query']))
        before_item_query_1_cnt = len(set(tmp[(tmp['context_timestamp'] <= row['context_timestamp']) & (tmp['item_id'] == row['item_id'])]['query1']))
        after_item_query_cnt=len(set(tmp[(tmp['context_timestamp'] > row['context_timestamp']) & (tmp['item_id'] == row['item_id'])]['predict_category_property']))
        after_item_query_all_cnt = len(set(tmp[(tmp['context_timestamp'] > row['context_timestamp']) & (tmp['item_id'] == row['item_id'])]['query']))
        after_item_query_1_cnt = len(set(tmp[(tmp['context_timestamp'] > row['context_timestamp']) & (tmp['item_id'] == row['item_id'])]['query1']))
        before_shop_query_cnt=len(set(tmp[(tmp['context_timestamp'] <= row['context_timestamp']) & (tmp['shop_id'] == row['shop_id'])]['predict_category_property']))
        before_shop_query_all_cnt = len(set(tmp[(tmp['context_timestamp'] <= row['context_timestamp']) & (tmp['shop_id'] == row['shop_id'])]['query']))
        before_shop_query_1_cnt = len(set(tmp[(tmp['context_timestamp'] <= row['context_timestamp']) & (tmp['shop_id'] == row['shop_id'])]['query1']))
        after_shop_query_cnt=len(set(tmp[(tmp['context_timestamp'] > row['context_timestamp']) & (tmp['shop_id'] == row['shop_id'])]['predict_category_property']))
        after_shop_query_all_cnt = len(set(tmp[(tmp['context_timestamp'] > row['context_timestamp']) & (tmp['shop_id'] == row['shop_id'])]['query']))
        after_shop_query_1_cnt = len(set(tmp[(tmp['context_timestamp'] > row['context_timestamp']) & (tmp['shop_id'] == row['shop_id'])]['query1']))
        before_brand_query_cnt=len(set(tmp[(tmp['context_timestamp'] <= row['context_timestamp']) & (tmp['item_brand_id'] == row['item_brand_id'])]['predict_category_property']))
        before_brand_query_all_cnt = len(set(tmp[(tmp['context_timestamp'] <= row['context_timestamp']) & (tmp['item_brand_id'] == row['item_brand_id'])]['query']))
        before_brand_query_1_cnt = len(set(tmp[(tmp['context_timestamp'] <= row['context_timestamp']) & (tmp['item_brand_id'] == row['item_brand_id'])]['query1']))
        after_brand_query_cnt = len(set(tmp[(tmp['context_timestamp'] > row['context_timestamp']) & (tmp['item_brand_id'] == row['item_brand_id'])]['predict_category_property']))
        after_brand_query_all_cnt = len(set(tmp[(tmp['context_timestamp'] > row['context_timestamp']) & (tmp['item_brand_id'] == row['item_brand_id'])]['query']))
        after_brand_query_1_cnt = len(set(tmp[(tmp['context_timestamp'] > row['context_timestamp']) & (tmp['item_brand_id'] == row['item_brand_id'])]['query1']))
        before_city_query_cnt = len(set(tmp[(tmp['context_timestamp'] <= row['context_timestamp']) & (tmp['item_city_id'] == row['item_city_id'])]['predict_category_property']))
        before_city_query_all_cnt = len(set(tmp[(tmp['context_timestamp'] <= row['context_timestamp']) & (tmp['item_city_id'] == row['item_city_id'])]['query']))
        before_city_query_1_cnt = len(set(tmp[(tmp['context_timestamp'] <= row['context_timestamp']) & (tmp['item_city_id'] == row['item_city_id'])]['query1']))
        after_city_query_cnt = len(set(tmp[(tmp['context_timestamp'] > row['context_timestamp']) & (tmp['item_city_id'] == row['item_city_id'])]['predict_category_property']))
        after_city_query_all_cnt = len(set(tmp[(tmp['context_timestamp'] > row['context_timestamp']) & (tmp['item_city_id'] == row['item_city_id'])]['query']))
        after_city_query_1_cnt = len(set(tmp[(tmp['context_timestamp'] > row['context_timestamp']) & (tmp['item_city_id'] == row['item_city_id'])]['query1']))
        before_cate_query_cnt = len(set(tmp[(tmp['context_timestamp'] <= row['context_timestamp']) & (tmp['item_category_list'] == row['item_category_list'])]['predict_category_property']))
        before_cate_query_all_cnt = len(set(tmp[(tmp['context_timestamp'] <= row['context_timestamp']) & (tmp['item_category_list'] == row['item_category_list'])]['query']))
        before_cate_query_1_cnt = len(set(tmp[(tmp['context_timestamp'] <= row['context_timestamp']) & (tmp['item_category_list'] == row['item_category_list'])]['query1']))
        after_cate_query_cnt = len(set(tmp[(tmp['context_timestamp'] > row['context_timestamp']) & (tmp['item_category_list'] == row['item_category_list'])]['predict_category_property']))
        after_cate_query_all_cnt = len(set(tmp[(tmp['context_timestamp'] > row['context_timestamp']) & (tmp['item_category_list'] == row['item_category_list'])]['query']))
        after_cate_query_1_cnt = len(set(tmp[(tmp['context_timestamp'] > row['context_timestamp']) & (tmp['item_category_list'] == row['item_category_list'])]['query1']))
        feature['max_diff'] = max_diff
        feature['min_diff'] = min_diff
        feature['avg_diff'] = avg_diff
        feature['mid_diff'] = mid_diff
        feature['diff_first_click'] = diff_first_click
        feature['diff_last_click'] = diff_last_click
        feature['previous_diff'] = previous_diff
        feature['next_diff'] = next_diff
        feature['before_query_rate'] = before_query_rate
        feature['after_query_rate'] = after_query_rate
        feature['after_query_all_rate'] = after_query_all_rate
        feature['before_query_all_rate'] = before_query_all_rate
        feature['after_query_1_rate'] = after_query_1_rate
        feature['before_query_1_rate'] = before_query_1_rate
        feature['before_item_rate'] = before_item_rate
        feature['after_item_rate'] = after_item_rate
        feature['before_shop_rate'] = before_shop_rate
        feature['after_shop_rate'] = after_shop_rate
        feature['before_brand_rate'] = before_brand_rate
        feature['after_brand_rate'] = after_brand_rate
        feature['before_city_rate'] = before_city_rate
        feature['after_city_rate'] = after_city_rate
        feature['before_item_query_cnt'] = before_item_query_cnt
        feature['after_item_query_cnt'] = after_item_query_cnt
        feature['before_shop_query_cnt'] = before_shop_query_cnt
        feature['after_shop_query_cnt'] = after_shop_query_cnt
        feature['before_brand_query_cnt'] = before_brand_query_cnt
        feature['after_brand_query_cnt'] = after_brand_query_cnt
        feature['before_city_query_cnt'] = before_city_query_cnt
        feature['after_city_query_cnt'] = after_city_query_cnt
        feature['before_item_query_all_cnt'] = before_item_query_all_cnt
        feature['before_item_query_1_cnt'] = before_item_query_1_cnt
        feature['after_item_query_all_cnt'] = after_item_query_all_cnt
        feature['after_item_query_1_cnt'] = after_item_query_1_cnt
        feature['before_shop_query_all_cnt'] = before_shop_query_all_cnt
        feature['before_shop_query_1_cnt'] = before_shop_query_1_cnt
        feature['after_shop_query_all_cnt'] = after_shop_query_all_cnt
        feature['after_shop_query_1_cnt'] = after_shop_query_1_cnt
        feature['before_brand_query_all_cnt'] = before_brand_query_all_cnt
        feature['before_brand_query_1_cnt'] = before_brand_query_1_cnt
        feature['after_brand_query_all_cnt'] = after_brand_query_all_cnt
        feature['after_brand_query_1_cnt'] = after_brand_query_1_cnt
        feature['before_city_query_all_cnt'] = before_city_query_all_cnt
        feature['before_city_query_1_cnt'] = before_city_query_1_cnt
        feature['after_city_query_all_cnt'] = after_city_query_all_cnt
        feature['after_city_query_1_cnt'] = after_city_query_1_cnt
        feature['before_cate_query_cnt'] = before_cate_query_cnt
        feature['before_cate_query_all_cnt'] = before_cate_query_all_cnt
        feature['before_cate_query_1_cnt'] = before_cate_query_1_cnt
        feature['after_cate_query_cnt'] = after_cate_query_cnt
        feature['after_cate_query_all_cnt'] = after_cate_query_all_cnt
        feature['after_cate_query_1_cnt'] = after_cate_query_1_cnt
        features.append(feature)
    print(str(i) + ' processor finished !')
    return pd.DataFrame(features)

def leak_feature():
    res = []
    p = Pool(processor)
    for i in range(processor):
        res.append(p.apply_async(run_leak_feature, args=( i,)))
        print(str(i) + ' processor started !')
    p.close()
    p.join()
    data = pd.concat([i.get() for i in res])
    data.to_csv('../data/leak_all.csv',index=False)
    # return data

def run_compare_feature(i):
    data = pd.read_csv('../data/user_data/query_' + str(i) + '.csv')
    features=[]
    for index,row in data.iterrows():
        feature={}
        feature['instance_id']=row['instance_id']
        if index%1000==0:
            print(index)
        tmp = data[(data['user_id'] == row['user_id'])&(data['day']==row['day'])]
        # tmp=tmp.sort_values(by='context_timestamp').reset_index(drop=True)
        before_low_price_cnt=len(set(tmp[(tmp['context_timestamp']<row['context_timestamp']) &(tmp['item_price_level']<row['item_price_level'])]['item_id']))
        after_low_price_cnt = len(set(tmp[(tmp['context_timestamp'] > row['context_timestamp']) & (tmp['item_price_level'] < row['item_price_level'])]['item_id']))
        before_high_sale_cnt=len(set(tmp[(tmp['context_timestamp']<row['context_timestamp']) &(tmp['item_sales_level']>row['item_sales_level'])]['item_id']))
        after_high_sale_cnt = len(set(tmp[(tmp['context_timestamp'] > row['context_timestamp']) & (tmp['item_sales_level'] > row['item_sales_level'])]['item_id']))
        before_high_review_num_cnt = len(set(tmp[(tmp['context_timestamp'] < row['context_timestamp']) & (tmp['shop_review_num_level'] > row['shop_review_num_level'])]['shop_id']))
        after_high_review_num_cnt = len(set(tmp[(tmp['context_timestamp'] > row['context_timestamp']) & (tmp['shop_review_num_level'] > row['shop_review_num_level'])]['shop_id']))
        before_high_review_positive_cnt=len(set(tmp[(tmp['context_timestamp'] < row['context_timestamp']) & (tmp['shop_review_positive_rate'] > row['shop_review_positive_rate'])]['shop_id']))
        after_high_review_positive_cnt = len(set(tmp[(tmp['context_timestamp'] > row['context_timestamp']) & (tmp['shop_review_positive_rate'] > row['shop_review_positive_rate'])]['shop_id']))
        before_high_star_level_cnt=len(set(tmp[(tmp['context_timestamp'] < row['context_timestamp']) & (tmp['shop_star_level'] > row['shop_star_level'])]['shop_id']))
        after_high_star_level_cnt = len(set(tmp[(tmp['context_timestamp'] > row['context_timestamp']) & (tmp['shop_star_level'] > row['shop_star_level'])]['shop_id']))
        before_high_score_service_cnt=len(set(tmp[(tmp['context_timestamp'] < row['context_timestamp']) & (tmp['shop_score_service'] > row['shop_score_service'])]['shop_id']))
        after_high_score_service_cnt = len(set(tmp[(tmp['context_timestamp'] > row['context_timestamp']) & (tmp['shop_score_service'] > row['shop_score_service'])]['shop_id']))
        before_high_score_delivery_cnt=len(set(tmp[(tmp['context_timestamp'] < row['context_timestamp']) & (tmp['shop_score_delivery'] > row['shop_score_delivery'])]['shop_id']))
        after_high_score_delivery_cnt = len(set(tmp[(tmp['context_timestamp'] > row['context_timestamp']) & (tmp['shop_score_delivery'] > row['shop_score_delivery'])]['shop_id']))
        before_high_score_description_cnt=len(set(tmp[(tmp['context_timestamp'] < row['context_timestamp']) & (tmp['shop_score_description'] > row['shop_score_description'])]['shop_id']))
        after_high_score_description_cnt = len(set(tmp[(tmp['context_timestamp'] > row['context_timestamp']) & (tmp['shop_score_description'] > row['shop_score_description'])]['shop_id']))
        feature['before_low_price_cnt'] = before_low_price_cnt
        feature['after_low_price_cnt'] = after_low_price_cnt
        feature['before_high_sale_cnt'] = before_high_sale_cnt
        feature['after_high_sale_cnt'] = after_high_sale_cnt
        feature['before_high_review_num_cnt'] = before_high_review_num_cnt
        feature['after_high_review_num_cnt'] = after_high_review_num_cnt
        feature['before_high_review_positive_cnt'] = before_high_review_positive_cnt
        feature['after_high_review_positive_cnt'] = after_high_review_positive_cnt
        feature['before_high_star_level_cnt'] = before_high_star_level_cnt
        feature['after_high_star_level_cnt'] = after_high_star_level_cnt
        feature['before_high_score_service_cnt'] = before_high_score_service_cnt
        feature['after_high_score_service_cnt'] = after_high_score_service_cnt
        feature['before_high_score_delivery_cnt'] = before_high_score_delivery_cnt
        feature['after_high_score_delivery_cnt'] = after_high_score_delivery_cnt
        feature['before_high_score_description_cnt'] = before_high_score_description_cnt
        feature['after_high_score_description_cnt'] = after_high_score_description_cnt
        features.append(feature)
    print(str(i) + ' processor finished !')
    return pd.DataFrame(features)
    # return data[['instance_id','before_low_price_cnt','after_low_price_cnt','before_high_sale_cnt','after_high_sale_cnt'
    #              ,'before_high_review_num_cnt','after_high_review_num_cnt','before_high_review_positive_cnt','after_high_review_positive_cnt'
    #              ,'before_high_star_level_cnt','after_high_star_level_cnt','before_high_score_service_cnt','after_high_score_service_cnt'
    #              ,'before_high_score_delivery_cnt','after_high_score_delivery_cnt','before_high_score_description_cnt','after_high_score_description_cnt']]

"""
    当天的竞争特征
    之前之后点击了多少价格更低的商品，销量更高的商品，评价数更多的店铺，
    好评率高的店铺，星级高的店铺，服务态度高的店铺，物流好的店铺，描述平分高的店铺
    :param data:
    :return:
    """
def compare_feature():
    # users = pd.DataFrame(list(set(data['user_id'].values)), columns=['user_id'])
    res = []
    p = Pool(processor)
    for i in range(processor):
        res.append(p.apply_async(run_compare_feature, args=(i,)))
        print(str(i) + ' processor started !')
    p.close()
    p.join()
    data = pd.concat([i.get() for i in res])
    data.to_csv('../data/compare_all.csv',index=False)
    # return data

if __name__ == '__main__':
    query_data_prepare()
    gc.collect()
    query_feature()
    print('query_feature finish')
    leak_feature()
    print('leak_feature finish')
    compare_feature()
    print('compare_feature finish')