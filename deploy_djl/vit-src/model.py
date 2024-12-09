import os
import boto3
import torch
import logging

from PIL import Image
from djl_python import Input, Output
import requests
from io import BytesIO
import base64, json

from first_page_pic_infer import expand2square,process_image,get_inputs,Pairwise_ViT_Infer
from transformers import CLIPVisionModel, CLIPImageProcessor, AutoTokenizer, AutoModel

model_dict = None


def load_model(properties):
    s3 = boto3.client('s3')
    # print('!!!!!',properties)
    model_location = properties["model_dir"]
    if "model_id" in properties:
        model_location = properties["model_id"]
    logging.info(f"Loading model from: {model_location}")
    logging.info(f"Loading model contains: {os.listdir(model_location)}")

    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

    vit_image_processor = CLIPImageProcessor.from_pretrained('openai/clip-vit-large-patch14-336')  # 加载图像预处理
    vision_tower = CLIPVisionModel.from_pretrained('openai/clip-vit-large-patch14-336')  # 加载图像模型
    tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-large-zh-v1.5', model_max_length=512)
    emb_model = AutoModel.from_pretrained('BAAI/bge-large-zh-v1.5')

    vit_model = Pairwise_ViT_Infer(vision_tower).to(device)
    vit_model.load_state_dict(torch.load(os.path.join(model_location,'model.pth')), strict=True)

    model_dict = {
        "vit_image_processor": vit_image_processor,
        'tokenizer':tokenizer,
        'emb_model':emb_model,
        "vit_model": vit_model
    }

    return model_dict


def handle(inputs: Input):
    global model_dict
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

    if not model_dict:
        model_dict = load_model(inputs.get_properties())

    if inputs.is_empty():
        # Model server makes an empty call to warmup the model on startup
        return None

    data = inputs.get_as_json()
    image_file1 = data["input_image1"]
    image_file2 = data["input_image2"]
    text1 = data["text1"]
    text2 = data["text2"]
    if image_file1.startswith("http") or image_file1.startswith("https"):
        response = requests.get(image_file1)
        image1 = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image1 = Image.open(BytesIO(base64.b64decode(image_file1)))


    if image_file2.startswith("http") or image_file2.startswith("https"):
        response = requests.get(image_file2)
        image2 = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image2 = Image.open(BytesIO(base64.b64decode(image_file2)))

    tokenizer, emb_model, vit_image_processor, vit_model = model_dict['tokenizer'], model_dict['emb_model'], model_dict['vit_image_processor'], model_dict['vit_model']

    image1 = process_image(image1, vit_image_processor, device)
    image2 = process_image(image2, vit_image_processor, device)

    text1, text2 = get_inputs(tokenizer, emb_model, text1, text2, device)

    score = vit_model(text1, text2, image1, image2)


    return Output().add({"score":str(score.cpu().detach().numpy()[0][0])})
