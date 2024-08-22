import torch
from torchvision import transforms
from transformers import CLIPVisionModel, CLIPImageProcessor, AutoTokenizer, AutoModel
from torch.utils.data import Dataset as Dataset
from torch.utils.data import DataLoader as Dataloader
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor
import json
import pandas as pd
import argparse
import os
from tqdm import tqdm
from accelerate import Accelerator
from PIL import Image

from accelerate import DistributedDataParallelKwargs

ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

#accelerator = Accelerator()

# 检查是否有GPU可用
print(f"是否有GPU可用: {torch.cuda.is_available()}")

# 检查GPU数量
print(f"GPU数量: {torch.cuda.device_count()}")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def expand2square(pil_img, background_color):
    width, height = pil_img.size  # 获得图像宽高
    if width == height:  # 相等直接返回不用重搞
        return pil_img
    elif width > height:  # w大构建w尺寸图
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))  # w最大，以坐标x=0,y=(width - height) // 2位置粘贴原图
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result

class myDataset(Dataset):
    def __init__(self, file_path,image_dir,image_processor,tokenizer,emb_model):
        with open(file_path, 'r') as f:
            self.input_data = json.load(f)
        self.image_dir = image_dir
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.emb_model=emb_model

    def __len__(self):
        return len(self.input_data)

    def get_images(self, idx):
        image1 = Image.open(os.path.join(self.image_dir,self.input_data[idx]['image1'])).convert('RGB') 
        image2 = Image.open(os.path.join(self.image_dir,self.input_data[idx]['image2'])).convert('RGB') 
        image1 = expand2square(image1, tuple(int(x * 255) for x in self.image_processor.image_mean))
        image2 = expand2square(image2, tuple(int(x * 255) for x in self.image_processor.image_mean))
        image1 = self.image_processor.preprocess(image1, return_tensors='pt')['pixel_values'][0]
        image2 = self.image_processor.preprocess(image2, return_tensors='pt')['pixel_values'][0]
        return image1,image2
    
    def get_inputs(self, idx):
        # Tokenize sentences
        encoded_input1 = self.tokenizer(self.input_data[idx]['text1'], padding=True, truncation=True, return_tensors='pt') 
        encoded_input2 = self.tokenizer(self.input_data[idx]['text2'], padding=True, truncation=True, return_tensors='pt')
        text_output1 = torch.nn.functional.normalize(self.emb_model(**encoded_input1)[0][:, 0], p=2, dim=1)
        text_output2 = torch.nn.functional.normalize(self.emb_model(**encoded_input2)[0][:, 0], p=2, dim=1)

        return text_output1,text_output2

    def get_labels(self, idx):
        return self.input_data[idx]['label']

    def get_ids(self, idx):
        return self.input_data[idx]['image1']+'^^^'+self.input_data[idx]['image2']

    def __getitem__(self, idx):

        # Get input data in a batch
        id = self.get_ids(idx)
        image1,image2 = self.get_images(idx)
        label = self.get_labels(idx)
        text1,text2 = self.get_inputs(idx)

        return id, text1, text2, image1, image2, label

class Pairwise_ViT(nn.Module):
    def __init__(self,vision_tower, num_labels=2):
        super(Pairwise_ViT, self).__init__()

        self.vit = vision_tower
        self.score_layer = torch.nn.Linear(4096, 1)

    def forward(self,text1, text2, x1, x2):
        x1 = self.vit(x1,output_hidden_states=True)['last_hidden_state']
        x2 = self.vit(x2, output_hidden_states=True)['last_hidden_state']
        
        # print(x1[:, 0, :].shape)
        # print(text1.shape)
        feature1 = torch.concat([x1[:, 0, :],text1.squeeze(dim=1), x2[:, 0, :],text2.squeeze(dim=1)],dim=-1)
        # Use the embedding of [CLS] token
        output1 = self.score_layer(feature1)

        output = torch.sigmoid(output1)

        return output,output1


def train(args):
    model_name = args.model_name
    in_channels = args.in_channels
    train_data_file = args.train_data_file
    test_data_file = args.test_data_file
    image_dir = args.image_dir
    batch_size = args.batch_size
    num_cls = args.num_cls
    max_epoch = args.max_epoch
    lr = args.lr
    weight_decay = args.weight_decay
    num_workers = args.num_workers
    print_interval_steps = args.print_interval_steps
    use_score_model = args.use_score_model
    use_dropout = args.use_dropout
    keep_prob = args.keep_prob
    clip_value = args.clip_value
    save_path = args.save_path
    save_name = args.save_name

    infer_hyper_params = {
        "model_name": model_name,
        "in_channels": in_channels,
        "num_cls": num_cls,
        "use_score_model": use_score_model,
        "use_dropout": use_dropout,
        "keep_prob": keep_prob,
        "save_name": save_name
    }
    
    image_processor = CLIPImageProcessor.from_pretrained(model_name)  # 加载图像预处理
    vision_tower = CLIPVisionModel.from_pretrained(model_name)  # 加载图像模型
    
    tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-large-zh-v1.5',model_max_length=512)
    emb_model = AutoModel.from_pretrained('BAAI/bge-large-zh-v1.5')

    dataset = myDataset(train_data_file,image_dir,image_processor,tokenizer,emb_model)
    data_loader = Dataloader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        sampler=None,
        batch_sampler=None,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    dataset_test = myDataset(test_data_file,image_dir,image_processor,tokenizer,emb_model)
    data_loader_test = Dataloader(
        dataset_test,
        batch_size=1,
        shuffle=False,
        sampler=None,
        batch_sampler=None,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    print('-----!!',len(data_loader),len(data_loader_test))

    model = Pairwise_ViT(vision_tower).to(device)
    criterion = torch.nn.BCELoss().to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    data_loader, data_loader_test, model, optimizer = accelerator.prepare(
        data_loader,data_loader_test,  model, optimizer
    )
    
    print('!!!!!!',len(data_loader),len(data_loader_test))
    
    # for name, param in model.named_parameters():
    #     if name.startswith('emb_model'):
    #         param.requires_grad = False

    # for name, param in model.named_parameters():
    #     if param.requires_grad == True:
    #         print(name)      
            
    model.train()
    for epoch in range(max_epoch):
        # total_acc_train = 0
        total_loss_train = 0.0

        for ids, text1, text2, train_image1, train_image2, train_label in tqdm(data_loader):
            # print(train_image.shape)
            output,score = model(text1, text2,train_image1, train_image2)
            # print(output)
            # print( train_label.reshape(output.shape).to(device))
            # print(output)
            # print(train_label.reshape(output.shape))
            loss = criterion(output.to(device), train_label.reshape(output.shape).to(device).to(torch.float32))
            # print(loss)
            # acc = (output.argmax(dim=1) == train_label.to(device)).sum().item()
            # total_acc_train += acc
            total_loss_train += loss.item()
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

        print(
            f'Epochs: {epoch + 1} | Loss: {total_loss_train / len(data_loader): .3f} ')

    
    # torch.save(model, os.path.join(save_path, save_name))
    print("Training is finished!")
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, "infer_hyper_params.json"), "w") as f:
        json.dump(infer_hyper_params, f)
     
    accelerator.wait_for_everyone()
    
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        print(os.path.join(args.save_path, "model.pth"))
        os.makedirs(args.save_path, exist_ok=True)
        accelerator.save(unwrapped_model.state_dict(), os.path.join(args.save_path, "model.pth"))
        print("Model is saved: {}".format(os.path.join(args.save_path, "model.pth")))

    model.eval()
    ss = []
    for id,text1, text2, image1,image2, label in tqdm(data_loader_test):
        output,score = model(text1, text2,image1,image2)
        
        # print(output)
        prediction = accelerator.gather_for_metrics(output.item())
        score = accelerator.gather_for_metrics(score.item())
        # print(prediction)
        # prediction = prediction.cpu().numpy()


        ss.append([id[0],prediction,score,label.cpu().numpy()[0]])
    ss = pd.DataFrame(ss,columns=['image','predict','score','label'])
    ss.to_csv(os.path.join(args.save_path, 'pred.csv'), index=False)


if __name__ == "__main__":
    try:
        train_data_dir = os.environ['SM_CHANNEL_TRAIN']
        test_data_dir = os.environ['SM_CHANNEL_TEST']
        model_dir = os.environ['SM_MODEL_DIR']
    except:
        train_data_dir = ''
        test_data_dir = ''
        model_dir = 'vit_pairwise_model_emb'
        
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name", type=str, default='Salesforce/SFR-Embedding-Mistral',
        help="The name of the backbone (default BERT)."
    )
    parser.add_argument("--in_channels", type=int, default=1024+8,
                        help="The channels of the backbone output (default 768).")
    parser.add_argument("--train_data_file", type=str)
    parser.add_argument("--test_data_file", type=str)
    parser.add_argument("--image_dir", type=str, default='imgs')              
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_cls", type=int, default=3, help="It is not used in the score model.")
    parser.add_argument("--max_epoch", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.005)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--print_interval_steps", type=int, default=100)
    parser.add_argument("--use_score_model", type=int, default=1,
                        help="Whether to use score model (default 1). 1 denotes yes, 0 denotes no."
                        )
    parser.add_argument("--use_dropout", type=int, default=0,
                        help="Whether to use dropout to mitigate overfitting (default 0). 1 denotes yes, 0 denotes no."
                        )
    parser.add_argument("--keep_prob", type=float, default=0.8)
    parser.add_argument("--clip_value", type=float, default=1.0)
    parser.add_argument("--save_path", type=str, default=model_dir)
    parser.add_argument("--save_name", type=str)

    args = parser.parse_args()
    print(args)

    train(args)
