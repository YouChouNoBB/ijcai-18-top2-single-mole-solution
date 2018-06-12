# -*- coding: utf-8 -*-
# @author: bryan
import time
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from functools import reduce

def today(x):
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(x))

def getday(x):
    day=int(x.split(' ')[0].split('-')[-1])
    if day==31:
        day=0
    return day

def gethour(x):
    hour=int(x.split(' ')[1].split(':')[0])
    minute=int(x.split(' ')[1].split(':')[1])
    minute=1 if minute>=30 else 0
    return hour*2+minute

def same_cate(x):
    cate = set(x['item_category_list'].split(';'))
    cate2 = set([i.split(':')[0] for i in x['predict_category_property'].split(';')])
    return len(cate & cate2)

def same_property(x):
    property_a = set(x['item_property_list'].split(';'))
    a = []
    for i in [i.split(':')[1].split(',') for i in x['predict_category_property'].split(';') if
              len(i.split(':')) > 1]:
        a += i
    property_b = set(a)
    return len(property_a & property_b)



def fillna(data):
    numeric_feature = ['day', 'item_price_level', 'item_sales_level', 'item_collected_level', 'item_pv_level',
                       'user_age_level', 'user_star_level', 'shop_review_num_level',
                       'shop_review_positive_rate', 'shop_star_level', 'shop_score_service', 'shop_score_delivery',
                       'shop_score_description', 'context_page_id'
                       ]
    string_feature = ['shop_id', 'item_id', 'user_id', 'item_brand_id', 'item_city_id', 'user_gender_id',
                      'user_occupation_id', 'context_page_id', 'hour']
    other_feature = ['item_property_list', 'predict_category_property']
    #填充缺失值
    for i in string_feature+other_feature:
        mode_num = data[i].mode()[0]
        if (mode_num != -1):
            print(i)
            data.loc[data[i] == -1, i] = mode_num
        else:
            print(-1)
    for i in numeric_feature:
        mean_num = data[i].mean()
        if (mean_num != -1):
            print(i)
            data.loc[data[i] == -1, i] = mean_num
        else:
            print(-1)
    return data

def fix_instance_id(data):
    #修复训练集instance_id重复问题
    enc=LabelEncoder()
    enc.fit(data['instance_id'].values)
    data['instance_id']=enc.transform(data['instance_id'])
    instances=set()
    size=5000000
    for index,row in data.iterrows():
        if index%10000==0:
            print(index)
        if row['instance_id'] in instances:
            data.loc[index,'instance_id']=row['instance_id']+size
            size-=1
        else:
            instances.add(row['instance_id'])
    return data

"""
统计属性出现的次数，取top1的属性作为特征，top1-5合并作为特征
预测的属性，top1,合并top1-5
"""
def property_feature(org):
    tmp=org['item_property_list'].apply(lambda x:x.split(';')).values
    property_dict={}
    property_list=[]
    for i in tmp:
        property_list+=i
    for i in property_list:
        if i in property_dict:
            property_dict[i]+=1
        else:
            property_dict[i] = 1
    print('dict finish')
    def top(x):
        propertys=x.split(';')
        cnt=[property_dict[i] for i in propertys]
        res=sorted(zip(propertys,cnt),key=lambda x:x[1],reverse=True)
        top1=res[0][0]
        top2 = '_'.join([i[0] for i in res[:2]])
        top3 = '_'.join([i[0] for i in res[:3]])
        top4 = '_'.join([i[0] for i in res[:4]])
        top5='_'.join([i[0] for i in res[:5]])
        top10 = '_'.join([i[0] for i in res[:10]])
        return (top1,top2,top3,top4,top5,top10)
    org['top']=org['item_property_list'].apply(top)
    print('top finish')
    org['top1']=org['top'].apply(lambda x:x[0])
    org['top2'] = org['top'].apply(lambda x: x[1])
    org['top3'] = org['top'].apply(lambda x: x[2])
    org['top4'] = org['top'].apply(lambda x: x[3])
    org['top5'] = org['top'].apply(lambda x: x[4])
    org['top10'] = org['top'].apply(lambda x: x[5])
    return org[['instance_id','top1','top2','top3','top4','top5','top10']]

#类别特征全部编码
def encode(data):
    id_features=['shop_id', 'item_id', 'user_id', 'item_brand_id', 'item_city_id', 'user_gender_id','item_property_list', 'predict_category_property',
                      'user_occupation_id', 'context_page_id','top1','top2','top3','top4','top5','top10','query1','query','cate']
    for feature in id_features:
        data[feature] = LabelEncoder().fit_transform(data[feature])
    return data

if os.path.exists('../data/origion_concat.csv'):
    # data=pd.read_csv('../data/origion_concat.csv')
    print('data prepared ~')
else:
    train = pd.read_csv('../data/round2_train.txt', sep=' ')
    testa = pd.read_csv('../data/round2_ijcai_18_test_a_20180425.txt', sep=' ')
    testb = pd.read_csv('../data/round2_ijcai_18_test_b_20180510.txt', sep=' ')
    testa['is_trade'] = -1
    testb['is_trade'] = -2
    # train=fix_instance_id(train)
    test=pd.concat([testa,testb])
    data = pd.concat([train, test]).reset_index(drop=True)
    data['day']=data['context_timestamp'].apply(lambda x:getday(today(x)))
    data['hour']=data['context_timestamp'].apply(lambda x:int(today(x).split()[1].split(':')[0]))
    data['context_timestamp']=data['context_timestamp'].apply(lambda x:today(x))
    data['hour48']=data['context_timestamp'].apply(gethour)
    data['same_cate']=data.apply(same_cate,axis=1) #相同类别数
    # data.loc[data['same_cate']>2,'same_cate']=2 #将相同类别数3规范为2，因为测试集中没有3
    data['same_property']=data.apply(same_property,axis=1) #相同属性数
    data['property_num']=data['item_property_list'].apply(lambda x:len(x.split(';'))) #属性的数目
    data['pred_cate_num']=data['predict_category_property'].apply(lambda x:len(x.split(';'))) #query的类别数目
    def f(x):
        try:
            return len([i for i in reduce((lambda x, y: x + y), [i.split(':')[1].split(',') for i in x.split(';') if len(i.split(':'))>1]) if i != '-1'])
        except:
            return 0
    data['pred_prop_num']=data['predict_category_property'].apply(f) #query的属性数目
    data['query1']=data['predict_category_property'].apply(lambda x:x.split(';')[0].split(':')[0]) #query第一个类别
    data['query']=data['predict_category_property'].apply(lambda x:'-'.join(sorted([i.split(':')[0] for i in [i for i in x.split(';')]]))) #query的全部类别
    data['cate'] = data['item_category_list'].apply(lambda x: x.split(';')[1])
    # data.loc[data['same_property']>5,'same_property']=5
    data=fillna(data.copy())
    data=pd.merge(data,property_feature(data),on='instance_id',how='left') #拆分属性
    data=encode(data)
    data.to_csv('../data/origion_concat.csv',index=False)