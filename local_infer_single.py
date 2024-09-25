
import torch
import os
from PIL import Image
import argparse
from transformers import CLIPVisionModel, CLIPImageProcessor, AutoTokenizer, AutoModel

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
    
class Pairwise_ViT_Infer(torch.nn.Module):
    def __init__(self, vision_tower, num_labels=2):
        super(Pairwise_ViT_Infer, self).__init__()

        self.vit = vision_tower
        self.score_layer = torch.nn.Linear(4096, 1)

    def forward(self, text1, text2, x1, x2):
        x1 = self.vit(x1, output_hidden_states=True)['last_hidden_state']
        x2 = self.vit(x2, output_hidden_states=True)['last_hidden_state']

        # print(x1[:, 0, :].shape)
        # print(text1.shape)
        feature1 = torch.concat([x1[:, 0, :], text1.squeeze(dim=1), x2[:, 0, :], text2.squeeze(dim=1)], dim=-1)
        # Use the embedding of [CLS] token
        output1 = self.score_layer(feature1)

        return output1
    
def init_model(model_path):
    ## load first-page-pic-scoring model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name=os.path.join(model_path)

    vit_image_processor = CLIPImageProcessor.from_pretrained('openai/clip-vit-large-patch14-336')  # 加载图像预处理
    vision_tower = CLIPVisionModel.from_pretrained('openai/clip-vit-large-patch14-336')  # 加载图像模型
    vit_model = Pairwise_ViT_Infer(vision_tower).to(device)
    vit_model.load_state_dict(torch.load(model_name, map_location=device), strict=True)

    tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-large-zh-v1.5', model_max_length=512)
    emb_model = AutoModel.from_pretrained('BAAI/bge-large-zh-v1.5')

    vit_image_processor = vit_image_processor
    vit_model = vit_model
    return tokenizer,emb_model,vit_image_processor,vit_model

###
def process_image(img_path,vit_image_processor,device):
    image1 = Image.open(img_path).convert("RGB")
    image1 = expand2square(image1, tuple(int(x * 255) for x in vit_image_processor.image_mean))
    image1 = vit_image_processor.preprocess(image1, return_tensors='pt')['pixel_values'][0].unsqueeze(0).to(device)    
    return image1

def get_inputs(tokenizer,emb_model,text1, text2, device):
        # Tokenize sentences
        encoded_input1 = tokenizer(text1, padding=True, truncation=True,
                                        return_tensors='pt')
        text_output1 = torch.nn.functional.normalize(emb_model(**encoded_input1)[0][:, 0], p=2, dim=1).to(device)

        encoded_input2 = tokenizer(text2, padding=True, truncation=True,
                                        return_tensors='pt')
        
        text_output2 = torch.nn.functional.normalize(emb_model(**encoded_input2)[0][:, 0], p=2, dim=1).to(device)

        return text_output1,text_output2

def main(model_path, img1, img2, text1, text2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  #  model_path = 'vit_pairwise_model_emb/model.pth'
    tokenizer,emb_model,vit_image_processor,vit_model = init_model(model_path)
    
    image1 = process_image(img1,vit_image_processor,device)
    image2 = process_image(img2,vit_image_processor,device) 

    text1, text2 = get_inputs(tokenizer,emb_model,text1, text2, device)

    score = vit_model(text1, text2, image1, image2)
    
    print(score)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='vit_pairwise_model_emb/model.pth')
    parser.add_argument('--image1', type=str)
    parser.add_argument('--text1', type=str)
    parser.add_argument('--image2', type=str)
    parser.add_argument('--text2', type=str)
    args = parser.parse_args()

    main(args.model_path, args.image1, args.text1, args.image2, args.text2)
