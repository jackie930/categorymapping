# e-com category mapping 

## 数据准备

首先将数据准备在两个文件夹目录下

* **project1**
    * data
        * imgs
    * scripts

其中, imgs包含了所有候选站点的采样商品图片 

运行如下预处理脚本, 会生成三个json文件, 分别用于模[]()型训练/模型测试/模型评估

```shell
!pip install openpyxl accelerate transformers
python prepare.py --split_idx 100
```

## 模型训练

单机多卡: 启动一个带有GPU机型的多卡机器, 至少是g5.12xlarge, 训练时运行
```shell
accelerate launch train_Clip_ViT_fullfc.py --model_name 'openai/clip-vit-large-patch14-336' --train_data_file '../data/train/train_vit_pairwise.json' --test_data_file '../data/train/test_vit_pairwise.json' \
--image_dir '../data/images' --max_epoch 3 \
--batch_size 8
```

单卡a10:
```shell
python train_Clip_ViT_fullfc.py --model_name 'openai/clip-vit-large-patch14-336' --train_data_file '../data/train/train_vit_pairwise.json' --test_data_file '../data/train/test_vit_pairwise.json' \
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

## 模型使用-单条推理
```shell
python local_infer_single.py --model_path 'vit_pairwise_model_emb/model.pth' \
--image1 '../data/images/41H+cMdoqqL._AC_US200_.jpg' \
--text1 '../data/images/61bf5e+XrAL._AC_US200_.jpg'\
--image2 'Gewerbe, Industrie & Wissenschaft:Materialtransport, Ladungssicherung & Zubehör:Zieh- & Hebevorrichtungen:Haken' \
--text2 'Beauty & Personal Care:Hair Care:Hair Extensions, Wigs & Accessories:Wigs'
```