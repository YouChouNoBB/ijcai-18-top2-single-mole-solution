# coding=utf-8
# @author: bryan

import pandas as pd
"""
用户最多连续看了多少个商品/店铺没有购买,在6号连续看了多少个商品/店铺没有购买，6号一共没有购买的商品数，店铺数
商品，店铺，类别，城市，品牌点击购买趋势，前7天统计
商品，店铺，类别，城市，品牌 被一次性购买的比例 ，一次性购买次数/购买次数
商品，店铺，类别，城市，品牌  第一次出现到第一次购买的时间间隔

"""

"""连续未购买7个特征，线下提升万0.5"""
def user_continue_nobuy(org):
    data = org[org['day'] < 7].sort_values(by=['user_id','context_timestamp'])
    train=org[org.day==7][['instance_id','user_id']]
    def f(x):
        max_no_buy=0
        res=[]
        for i in x:
            if i==0:
                max_no_buy+=1
                res.append(max_no_buy)
            else:
                max_no_buy=0
        return 0 if len(res)==0 else max(res)
    user_nobuy= data.groupby('user_id',as_index=False)['is_trade'].agg({'user_continue_nobuy_click_cnt':lambda x:f(x)})
    print('user_continue_nobuy_click_cnt finish')
    data=data[data.day==6].sort_values(by=['user_id','context_timestamp'])
    day6_user_nobuy=data.groupby('user_id', as_index=False)['is_trade'].agg({'day6_user_continue_nobuy_click_cnt': lambda x: f(x)})
    print('day6_user_continue_nobuy_click_cnt finish')
    train=pd.merge(train,user_nobuy,on='user_id',how='left')
    train = pd.merge(train, day6_user_nobuy, on='user_id', how='left')
    data = org[org['day'] ==6]
    user_buy_items=data[data.is_trade==1].groupby('user_id', as_index=False)['item_id'].agg({'day6_user_buy_items':lambda x:len(set(x))})
    user_nobuy_items=data.groupby('user_id', as_index=False)['item_id'].agg({'day6_user_nobuy_items': lambda x: len(set(x))})
    user_buy_shops = data[data.is_trade == 1].groupby('user_id', as_index=False)['item_id'].agg({'day6_user_buy_shops': lambda x: len(set(x))})
    user_nobuy_shops = data.groupby('user_id', as_index=False)['item_id'].agg({'day6_user_nobuy_shops': lambda x: len(set(x))})
    print('day6_user_nobuy finish')
    train=pd.merge(train,user_buy_items,on='user_id',how='left')
    train = pd.merge(train, user_nobuy_items, on='user_id', how='left')
    train = pd.merge(train, user_buy_shops, on='user_id', how='left')
    train = pd.merge(train, user_nobuy_shops, on='user_id', how='left')
    train['day6_user_items_d_shops']=train['day6_user_nobuy_items']/train['day6_user_nobuy_shops']
    train=train.drop('user_id',axis=1)
    train.to_csv('../data/nobuy_feature.csv',index=False)
    print('nobuy_feature finish')
    # return train


def trend(data,item):
    tmp = data.groupby([item, 'day'], as_index=False)['is_trade'].agg({'buy': 'sum', 'cnt': 'count'})
    features = []
    for key, df in tmp.groupby(item, as_index=False):
        feature = {}
        feature[item] = key
        for index, row in df.iterrows():
            feature[item+'buy' + str(row['day'])] = row['buy']
            feature[item+'cnt' + str(row['day'])] = row['cnt']
        features.append(feature)
    features =pd.DataFrame(features)
    return features
    # def f(x):
    #     return 1 if x>0 else 0
    # for i in range(6):
    #     features[item + 'buy_trend_'+str(i+1)] = features[item + 'buy'+str(i+1)] - features[item + 'buy'+str(i)]
    #     features[item + 'buy_trend_' + str(i + 1)]=features[item + 'buy_trend_'+str(i+1)].apply(f)
    # features[item+'buy_trend'] = features[item + 'buy_trend_1'] + features[item + 'buy_trend_2']+ features[item + 'buy_trend_3']+ features[item + 'buy_trend_4']+ features[item + 'buy_trend_5']+ features[item + 'buy_trend_6']
    # for i in range(6):
    #     features[item + 'cnt_trend_'+str(i+1)] = features[item + 'cnt'+str(i+1)] - features[item + 'cnt'+str(i)]
    #     features[item + 'cnt_trend_' + str(i + 1)] = features[item + 'cnt_trend_' + str(i + 1)].apply(f)
    # features[item+'cnt_trend'] = features[item + 'cnt_trend_1'] + features[item + 'cnt_trend_2'] + features[item + 'cnt_trend_3'] + features[item + 'cnt_trend_4'] + features[item + 'cnt_trend_5'] + features[item + 'cnt_trend_6']
    # return features.drop([item + 'buy_trend_'+str(i+1) for i in range(6)]+[item + 'cnt_trend_'+str(i+1) for i in range(6)],axis=1)
"""
商品，店铺，类别，城市，品牌点击购买趋势，前7天统计，比上一天高为1，否则为0，再统计1的次数，7个特征*5
"""


def trend_f(data, item):
    tmp = data.groupby([item, 'day'], as_index=False)['is_trade'].agg({'buy': 'sum', 'cnt': 'count'})
    features = []
    for key, df in tmp.groupby(item, as_index=False):
        feature = {}
        feature[item] = key
        for index, row in df.iterrows():
            feature[item + 'buy' + str(int(row['day']))] = row['buy']
            feature[item + 'cnt' + str(int(row['day']))] = row['cnt']
        features.append(feature)
    features = pd.DataFrame(features)
    return features

def trend_feature(org):
    data=org[org.day<7]
    col = ['item_id', 'item_brand_id', 'shop_id', 'item_category_list', 'item_city_id',
           'predict_category_property', 'context_page_id', 'query1', 'query']
    train=org[org.day==7][['instance_id']+col]
    items=col
    for item in items:
        train=pd.merge(train,trend_f(data, item),on=item,how='left')
        print(item+' finish')
    train=train.drop(items,axis=1)
    for item in items:
        for day in range(6):
            train['_'.join([item,str(day+1),'d',str(day),'cnt'])]=train[item + 'cnt' +str(day+1)]/train[item + 'cnt' +str(day)]
            train['_'.join([item, str(day + 1), 'd', str(day), 'buy'])]=train[item + 'buy' +str(day+1)]/train[item + 'buy' +str(day)]
    train=train[[i for i in train.columns if 'cnt6' not in i]]
    train.to_csv('../data/trend_feature.csv',index=False)
    print('trend_feature finish')
    # return train

# 商品，店铺，类别，城市，品牌，页面 被一次性购买的比例,次数 ，一次性购买次数/购买次数  线下测试只有item,shop的shot_rate有用
# 用户，商品，店铺，类别，城市，品牌，页面 7号一次性购买次数，交叉提取
# 如何定义一次性购买  cvr=1
def oneshot(data,item):
    tmp = data.groupby([item], as_index=False)['is_trade'].agg({item + '_buy': 'sum'})
    shot = data.groupby([item, 'user_id'], as_index=False)['is_trade'].agg({'is_shot': 'mean'})
    shot = shot[shot.is_shot == 1].groupby([item], as_index=False)['is_shot'].agg({item + 'shot_num': 'count'})
    tmp = pd.merge(tmp, shot, on=[item], how='left')
    tmp[item+'_shot_rate'] = tmp[item +'shot_num'] / tmp[item + '_buy']
    return tmp[[item,item+'_shot_rate']]

# calc data，join data
def today_shot(c_data, j_data):
    items=['item_id','shop_id','query','query1']
    j_data=j_data[['instance_id']+items]
    for item in items:
        j_data = pd.merge(j_data, oneshot(c_data, item), on=item, how='left')
    j_data=j_data.drop(items,axis=1)
    j_data.columns=['instance_id','today_item_shot_rate','today_shop_shot_rate','today_query_shot_rate','today_query1_shot_rate']
    return j_data

def today_shot_feature(org):
    from sklearn.model_selection import train_test_split
    data=org[org['day']==7]
    train=data[data['is_trade']>-1]
    predict=data[data['is_trade']<0]
    predict=today_shot(train,predict)
    train1,train2=train_test_split(train,test_size=0.5,random_state=1024)
    train22=today_shot(train1, train2)
    train11=today_shot(train2, train1)
    data=pd.concat([train11,train22,predict]).reset_index(drop=True)
    return data

def day6_shot_feature(org):
    data=org[org.day==6]
    items = ['item_id', 'shop_id', 'query', 'query1']
    train = org[org.day == 7][['instance_id']+items]
    for item in items:
        train = pd.merge(train, oneshot(data, item), on=item, how='left')
    train=train.drop(items,axis=1)
    train.columns=['instance_id','day6_item_shot_rate','day6_shop_shot_rate','day6_query_shot_rate','day6_query1_shot_rate']
    return train

def oneshot_feature(org):
    data=org[org.day<7]
    items = ['item_id', 'shop_id', 'query', 'query1']
    train = org[org.day == 7][['instance_id']+items]
    for item in items:
        train=pd.merge(train,oneshot(data, item),on=item,how='left')
        print(item+' finish')
    train = train.drop(items, axis=1)
    print(train.columns)
    today=today_shot_feature(org)
    print(today.columns)
    day6=day6_shot_feature(org)
    print(day6.columns)
    train=pd.merge(train,today,on='instance_id',how='left')
    train = pd.merge(train, day6, on='instance_id', how='left')
    train.to_csv('../data/oneshot_feature.csv', index=False)
    print('oneshot_feature finish')

# 商品，店铺，类别，城市，品牌，query  第一次出现到第一次购买的时间间隔
# 前所有天，第七天
def first_ocr(data,item):
    import numpy as np
    import datetime
    def sec_diff(a, b):
        if (a is np.nan) | (b is np.nan):
            return np.nan
        return (datetime.datetime.strptime(str(b), "%Y-%m-%d %H:%M:%S") - datetime.datetime.strptime(str(a),"%Y-%m-%d %H:%M:%S")).seconds
    ocr=data.groupby(item,as_index=False)['context_timestamp'].agg({'min_ocr_time':'min'})
    buy=data[data.is_trade==1].groupby(item,as_index=False)['context_timestamp'].agg({'min_buy_time':'min'})
    data=pd.merge(ocr,buy,on=item,how='left')
    data[item+'_ocr_buy_diff_day6']=data.apply(lambda x:sec_diff(x['min_ocr_time'],x['min_buy_time']),axis=1)
    return data[[item,item+'_ocr_buy_diff_day6']]

# calc data，join data
def today_ocr(c_data, j_data):
    items=['item_id','shop_id','predict_category_property']
    item_shot=first_ocr(c_data, items[0])
    shop_shot=first_ocr(c_data, items[1])
    query_shot=first_ocr(c_data, items[2])
    j_data=pd.merge(j_data,item_shot,on=items[0],how='left')
    j_data = pd.merge(j_data, shop_shot, on=items[1], how='left')
    j_data = pd.merge(j_data, query_shot, on=items[2], how='left')
    j_data= j_data[['instance_id','item_id_ocr_buy_diff','shop_id_ocr_buy_diff','predict_category_property_ocr_buy_diff']]
    j_data.columns=['instance_id','today_item_id_ocr_buy_diff','today_shop_id_ocr_buy_diff','today_predict_category_property_ocr_buy_diff']
    return j_data

def today_ocr_feature(org):
    from sklearn.model_selection import train_test_split
    data=org[org['day']==7]
    train=data[data['is_trade']!=-1]
    predict=data[data['is_trade']==-1]
    predict=today_ocr(train,predict)
    train1,train2=train_test_split(train,test_size=0.5,random_state=1024)
    train22=today_ocr(train1, train2)
    train11=today_ocr(train2, train1)
    data=pd.concat([train11,train22,predict]).reset_index(drop=True)
    return data

def first_ocr_feature(org):
    items=['item_id','query','query1']
    data=org[org.day<7]
    train=org[org.day==7][['instance_id']+items]
    # for item in items:
    #     tmp=first_ocr(data, item)
    #     tmp.columns=[item,item+'_ocr_buy_diff_all_day']
    #     train=pd.merge(train,tmp,on=item,how='left')
    #     print(item)
    data=data[data.day==6]
    for item in items:
        tmp=first_ocr(data, item)
        train=pd.merge(train,tmp,on=item,how='left')
        print(item)
    #today=today_ocr_feature(org)
    #train=pd.merge(train,today,on='instance_id',how='left')
    train=train.drop(items, axis=1)
    train.to_csv('../data/ocr_feature.csv',index=False)
    print('ocr_feature finish')


"""
item和shop 属性的变化，前7天的均值，第7天和前七天均值的差值，第7天和第六天的差值
item_price_level,item_sales_level,item_collected_level,item_pv_level
shop_review_num_level,shop_review_positive_rate,shop_star_level,shop_score_service,shop_score_delivery,shop_score_description
线下可以提升1个万分位
"""
def item_shop_var_feature(org):
    import numpy as np
    col=['item_id','shop_id']
    item_cates=['item_price_level','item_sales_level','item_collected_level','item_pv_level']
    shop_cates=['shop_review_num_level','shop_review_positive_rate','shop_star_level','shop_score_service','shop_score_delivery','shop_score_description']
    data=org[org.day<7]
    train=org[org.day==7][['instance_id']+col+item_cates+shop_cates]
    for cate in item_cates:
        train=pd.merge(train,data.groupby('item_id',as_index=False)[cate].agg({'item_id_'+cate+'_var':np.std,'item_id_'+cate+'_avg':'mean'}),on='item_id',how='left')
        train['_'.join(['diff',cate,'today_d_7days'])]=train[cate]-train['item_id_'+cate+'_avg']
    for cate in shop_cates:
        train=pd.merge(train,data.groupby('shop_id',as_index=False)[cate].agg({'shop_id_'+cate+'_var':np.std,'shop_id_'+cate+'_avg':'mean'}),on='shop_id',how='left')
        train['_'.join(['diff', cate, 'today_d_7days'])] = train[cate] - train['shop_id_' + cate + '_avg']
    data = org[org.day == 6]
    for cate in item_cates:
        avg=data.groupby('item_id',as_index=False)[cate].agg({'item_id_day6'+cate+'_avg':'mean'})
        tmp=pd.merge(train,avg,on='item_id',how='left')
        train['_'.join(['diff',cate,'today_d_6day'])]=tmp[cate]-tmp['item_id_day6'+cate+'_avg']
    for cate in shop_cates:
        avg=data.groupby('shop_id',as_index=False)[cate].agg({'shop_id_day6'+cate+'_avg':'mean'})
        tmp=pd.merge(train,avg,on='shop_id',how='left')
        train['_'.join(['diff',cate,'today_d_6day'])]=tmp[cate]-tmp['shop_id_day6'+cate+'_avg']
    train.drop(col + item_cates + shop_cates, axis=1).to_csv('../data/item_shop_var_feature.csv',index=False)

if __name__ == '__main__':
    org=pd.read_csv('../data/origion_concat.csv')
    user_continue_nobuy(org)
    trend_feature(org)
    oneshot_feature(org)
    first_ocr_feature(org)
    item_shop_var_feature(org)