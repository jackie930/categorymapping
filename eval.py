
## test mapping result
import json
import pandas as pd

    
def get_res(input_data,df_pred):
    res_score = []
    res_country_de = []
    res_country_us = []
    res_label = []
    for i in range(len(input_data)):
        res_score.append(df_pred['predict'][i])
        res_label.append(input_data[i]['label'])
        res_country_de.append(input_data[i]['text1'])
        res_country_us.append(input_data[i]['text2'])
    return pd.DataFrame({'category_de':res_country_de,'category_us':res_country_us,'score':res_score})

def get_maping_res_v2(res,rankn):
    de_cate = list(res['category_de'].unique())
    map_res_ls = []
    for i in de_cate:
        res_cate = res[res['category_de']==i].reset_index()
        res_cate.sort_values(['score'],ascending=False,inplace=True)
               
        map_res = list(res_cate.head(rankn)['category_us'])
        map_res_ls.append(map_res)
    
    res_df = pd.DataFrame({'de_cate':de_cate,'us_cate':map_res_ls})
    return res_df

def cal_acc(res,rankn,df_test):
    res_mapping = get_maping_res_v2(res,rankn)
    n_total = 0
    truth = 0
    for i in range(len(res_mapping)):
        de_cate = res_mapping['de_cate'][i]
        label = list(df_test[df_test['德亚亚马逊类目名称']==de_cate]['对应美亚类目'])[0]
        pred = res_mapping['us_cate'][i]
        #print ("label: ",label)
        #print ("pred: ", pred)
        n_total +=1
        if label in pred:
            truth +=1
    print ("<<< accuracy :", truth/n_total)
    
def main():
    df_test = pd.read_csv('../data/train/test_mapping.csv')
    df_pred = pd.read_csv('vit_pairwise_model_emb/pred_single_sample.csv')
    with open('../data/train/val_vit.json', 'r') as f:
        input_data = json.load(f)

    res = get_res(input_data,df_pred)
    for rankn in [1,2,3,4]:
        print ("<<<< ")
        print ("<<rank n", rankn)
        cal_acc(res,rankn,df_test)

        
if __name__ == "__main__":
    main()
