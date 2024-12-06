import torch
from PIL import Image


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

def process_image(image1,vit_image_processor,device):
    #image1 = Image.open(img_path).convert("RGB")
    image1 = expand2square(image1, tuple(int(x * 255) for x in vit_image_processor.image_mean))
    image1 = vit_image_processor.preprocess(image1, return_tensors='pt')['pixel_values'][0].unsqueeze(0).to(device)
    return image1

def get_inputs(tokenizer, emb_model, text1, text2, device):
    # Tokenize sentences
    encoded_input1 = tokenizer(text1, padding=True, truncation=True,
                               return_tensors='pt')
    text_output1 = torch.nn.functional.normalize(emb_model(**encoded_input1)[0][:, 0], p=2, dim=1).to(device)

    encoded_input2 = tokenizer(text2, padding=True, truncation=True,
                               return_tensors='pt')

    text_output2 = torch.nn.functional.normalize(emb_model(**encoded_input2)[0][:, 0], p=2, dim=1).to(device)

    return text_output1, text_output2

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
