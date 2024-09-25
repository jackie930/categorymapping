### 针对爬虫数据, 对于站点a, 生成一个sku信息表, 内容为sku_image/category_name/sku_title
import pandas as pd
import os
from tqdm import tqdm
import numpy as np
import argparse
#!pip install openpyxl

def get_de_res(df_de_cate_list,df):
    sku_image_ls = []
    category_name_ls = []
    sku_title_ls = []
    for j in tqdm(range(len(df_de_cate_list))):
        cate_de = df_de_cate_list[j]
        sku_info = df[df['nodelabelpath'] == cate_de].reset_index()
        # Set a random seed for reproducibility
        np.random.seed(42)
        # Shuffle the DataFrame
        sku_info = sku_info.sample(frac=1, random_state=42).reset_index(drop=True)

        x = 0
        brand_list = []
        def image_exists(file_path):
            return os.path.isfile(file_path)

        i = 0
        try:
            while x <= 2 & i <=len(sku_info):
                #print ("<<< i ", i )
                brand = sku_info['brand'][i]
                #print ("<<< brand: ", brand)
                sku_title = sku_info['title'][i]
                sku_image = sku_info['imageurl'][i]
                file_path = os.path.join('../data/images',sku_image.split('/')[-1])
                i = i +1
                if image_exists(file_path):
                    #if brand in brand_list:
                     #   continue
                  #  else:
                    x = x+1
                    brand_list.append(brand)
                    sku_image_ls.append(file_path)
                    category_name_ls.append(cate_de)
                    sku_title_ls.append(sku_title)
                    continue
                else:
                    continue
        except:
            print ('<<< error cate name', cate_de)
            continue
    res = pd.DataFrame({'sku_image':sku_image_ls,'category_name':category_name_ls,'sku_title':sku_title_ls})
    return res

if __name__ == "__main__":
    ## prepare data
    parser = argparse.ArgumentParser()
    parser.add_argument('--main_excel', type=str, default='../data/明确对应关系的类目数据.xlsx')
    parser.add_argument('--df_sku_1', type=str, default='../data/s_vevor_crs_crp_sellersprite_amazon_category_topn_asin_df临时数据.txt')
    parser.add_argument('--df_sku_2', type=str, default='../data/mongo_crs.crp_sellersprite_amazon_category_topn_asin美亚202312数据.txt')
    parser.add_argument('--save_folder', type=str, default='../data/train')
    args=parser.parse_args()

    print ("<<< load data")
    df_master = pd.read_excel(args.main_excel)
    df_de_cate_list = list(df_master['德亚亚马逊类目名称'])
    df_us_cate_list = list(df_master['对应美亚类目'])

    df = pd.read_csv(args.df_sku_1, sep=',')
    df_us = pd.read_csv(args.df_sku_2, sep=',')
    print("<<< process data")
    res_de = get_de_res(df_de_cate_list, df)
    res_us = get_de_res(df_us_cate_list, df_us)
    print("<<< save data")
    res_de.to_csv(os.path.join(args.save_folder, 'de_data.csv'))
    res_us.to_csv(os.path.join(args.save_folder, 'us_data.csv'))
