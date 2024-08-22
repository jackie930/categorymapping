# hotel_first_page_pic_sort

## 数据准备

首先将数据准备在两个文件夹目录下

* **project1**
    * data
        * imgs
    * scripts

其中, imgs包含了所有候选站点的采样商品图片 

运行如下预处理脚本, 会生成三个json文件, 分别用于模[]()型训练/模型测试/模型评估

```shell
python prepare.py --split_idx 100
```

## 模型训练

单机多卡: 启动一个带有GPU机型的机器, 至少是g5.12xlarge, 训练时运行
```shell
accelerate launch train_Clip_ViT_pairwise_emb.py --model_name 'openai/clip-vit-large-patch14-336' --train_data_file '../data/train/train_vit_pairwise.json' --test_data_file '../data/train/test_vit_pairwise.json' \
--image_dir '../data/images' --max_epoch 3 \
--batch_size 8
```

单卡a10:
```shell
python train_Clip_ViT_pairwise_emb.py --model_name 'openai/clip-vit-large-patch14-336' --train_data_file '../data/train/train_vit_pairwise.json' --test_data_file '../data/train/test_vit_pairwise.json' \
--image_dir '../data/images' --max_epoch 3 \
--batch_size 8
```

## 模型测试
```shell
python infer_Clip_ViT_fullfc.py  --model_name 'vit_pairwise_model_emb/model.pth' --test_data_file '../data/train/val_vit.json' --image_dir '../data/images' 
```

## 模型
评估
```shell
python eval.py 
```