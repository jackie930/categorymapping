##生成用于训练的train/test split
import pandas as pd
import os
from tqdm import tqdm
import random
import argparse
import json

def get_res(df, res_de, res_us):
    cates = df['德亚亚马逊类目名称']
    res = []
    for cate in tqdm(cates):
        map_info = list(df[df['德亚亚马逊类目名称'] == cate]['对应美亚类目'])[0]
        de_infors = res_de[res_de['category_name'] == cate].reset_index()
        pos_us_infors = res_us[res_us['category_name'] == map_info].reset_index()
        neg_us_infors = res_us[res_us['category_name'] != map_info]
        random_20_rows = neg_us_infors.sample(n=3, random_state=42).reset_index()

        # pos samples
        for i in range(len(de_infors)):
            for j in range(len(pos_us_infors)):
                sample = {"image1": de_infors['sku_image'][i].split('/')[-1], \
                          "text1": de_infors['category_name'][i], \
                          "text3": de_infors['sku_title'][i], \
                          "image2": pos_us_infors['sku_image'][j].split('/')[-1], \
                          "text2": pos_us_infors['category_name'][j], \
                          "text4": pos_us_infors['sku_title'][j], \
                          "label": 1}
                res.append(sample)

        # neg samples
        for i in range(len(de_infors)):
            for j in range(len(random_20_rows)):
                sample = {"image1": de_infors['sku_image'][i].split('/')[-1], \
                          "text1": de_infors['category_name'][i], \
                          "text3": de_infors['sku_title'][i], \
                          "image2": random_20_rows['sku_image'][j].split('/')[-1], \
                          "text2": random_20_rows['category_name'][j], \
                          "text4": random_20_rows['sku_title'][j], \
                          "label": 0}
                res.append(sample)
    return res


def get_test_res(df):
    de_cates = df['德亚亚马逊类目名称']
    us_cates = df['对应美亚类目']
    res = []
    for cate in tqdm(de_cates):
        de_infors = res_de[res_de['category_name'] == cate].reset_index()
        random_20_rows = de_infors.sample(n=1, random_state=42).reset_index()

        # de samples
        for i in range(len(random_20_rows)):
            sample = {"image": random_20_rows['sku_image'][i].split('/')[-1], \
                      "text": random_20_rows['category_name'][i], \
                      "label": 1}  # label 0 means us, label 1 means de
            res.append(sample)

    for cate in tqdm(us_cates):
        us_infors = res_us[res_us['category_name'] == cate].reset_index()
        random_20_rows = us_infors.sample(n=1, random_state=42).reset_index()

        # de samples
        for i in range(len(random_20_rows)):
            sample = {"image": us_infors['sku_image'][i].split('/')[-1], \
                      "text": us_infors['category_name'][i], \
                      "label": 0}  # label 0 means us, label 1 means de
            res.append(sample)

    return res


def main(df_master, res_de, res_us, split_idx):
    train_set = df_master.iloc[:split_idx, :]
    # todo: update test mapping
    test_set = df_master.iloc[split_idx:, :]
    print("<< 总类目匹配数量: ", len(train_set))
    print("<< 训练匹配数量: ", split_idx)
    print("<< 评估类目: ", len(test_set))

    train_res_df = get_res(train_set, res_de, res_us)
    ##version 1, single tower
    # test_res_df = get_test_res(test_set)
    ## version 2
    test_res_df = get_res(test_set, res_de, res_us)

    ## train test split
    random.shuffle(train_res_df)
    idx = int(len(train_res_df) * 0.9)
    train_json = train_res_df[:idx]
    test_json = train_res_df[idx:]

    # Open a file for writing (text mode with UTF-8 encoding)
    with open(os.path.join('../data/train', 'train_vit_pairwise.json'), 'w') as f:
        json.dump(train_json, f)  # Optional parameter for indentation
    print('Data written to json')

    with open(os.path.join('../data/train', 'test_vit_pairwise.json'), 'w') as f:
        json.dump(test_json, f)  # Optional parameter for indentation
    print('Data written to json')

    print(f"train samples:{len(train_json)},test samples:{len(test_json)}")

    # Open a file for writing (text mode with UTF-8 encoding)
    with open(os.path.join('../data/train', 'val_vit.json'), 'w') as f:
        json.dump(test_res_df, f)  # Optional parameter for indentation
    print('Data written to json')

    test_set.to_csv('../data/train/test_mapping.csv')

    return train_json, test_res_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--split_idx', type=int, default=100)
    df_master = pd.read_excel('../data/明确对应关系的类目数据.xlsx')

    res_de = pd.read_csv('../data/train/de_data.csv')
    res_us = pd.read_csv('../data/train/us_data.csv')
    res_de_cate = res_de['category_name'].unique()
    res_us_cate = res_us['category_name'].unique()

    df_master = pd.merge(df_master, pd.DataFrame({'德亚亚马逊类目名称': res_de_cate}), on='德亚亚马逊类目名称')
    df_master = pd.merge(df_master, pd.DataFrame({'对应美亚类目': res_us_cate}), on='对应美亚类目')
    print("<< 总类目数量: ", len(df_master))

    args = parser.parse_args()
    main(df_master, res_de, res_us, args.split_idx)
