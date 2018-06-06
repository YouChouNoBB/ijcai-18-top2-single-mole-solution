# coding=utf-8
# @author:bryan
"""
使用全量数据提取特征，点击数，交叉点击数，占比
"""
import pandas as pd

def full_count_feature(org,name):
    col=['user_id', 'item_id', 'item_brand_id', 'shop_id', 'item_category_list', 'item_city_id','cate','top10',
           'predict_category_property', 'context_page_id', 'query1', 'query']
    train=org[org.day==7][['instance_id']+col]
    if name=='day6':
        data = org[org.day==6][col]
    elif name=='days7':
        data=org[org.day<7][col]
    elif name == 'day7':
        data = org[org.day == 7][col]
    elif name=='full':
        data=org[col]
    for item in col:
        train=pd.merge(train,data.groupby(item,as_index=False)['user_id'].agg({'_'.join([name,item,'cnt']):'count'}),on=item,how='left')
        print(item)
    items=col
    for i in range(len(items)):
        for j in range(i+1,len(items)):
            egg=[items[i],items[j]]
            tmp = data.groupby(egg, as_index=False)['user_id'].agg({'_'.join([name,items[i],items[j],'cnt']): 'count'})
            train = pd.merge(train, tmp, on=egg, how='left')
            print(egg)
    cross=[['user_id','query'],['user_id','query1'],['user_id','shop_id'],['user_id','item_id'],['item_id','shop_id'],['item_id', 'item_brand_id'],
           ['item_brand_id', 'shop_id'],['item_id','item_category_list'],['item_id','query'],
           [ 'item_id','item_city_id'],['item_id','cate'],['item_id','top10'],['item_id','context_page_id'],['item_id','query1'],
           ['item_brand_id', 'shop_id'],['shop_id','item_city_id'],[ 'shop_id','context_page_id']
           ]
    for i in cross:
        train['_'.join(i+['cross'])]=train['_'.join([name,i[0],i[1],'cnt'])]/train['_'.join([name,i[1],'cnt'])]
        print(i)
    train=train.drop(col, axis=1)
    train.to_csv('../data/'+name+'_count_feature.csv',index=False)
    # return train

if __name__ == '__main__':
    org=pd.read_csv('../data/origion_concat.csv')
    full_count_feature(org, 'day6')
    full_count_feature(org, 'days7')
    full_count_feature(org, 'full')