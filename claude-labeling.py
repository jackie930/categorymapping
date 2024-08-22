from collections import deque
from threading import Lock
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed
import logging
import traceback
import boto3
import json
import base64
import json
import os
from PIL import Image
from tqdm import tqdm
import requests
import os
import time

region = 'us-west-2'  # 替换为您想要使用的AWS区域

boto_session = boto3.Session(region_name=region)
sagemaker_client = boto_session.client('sagemaker')
sagemaker_runtime_client = boto_session.client('sagemaker-runtime')

bedrock_client = boto3.client("bedrock-runtime", region_name="us-west-2")

max_workers = 16
request_count = 0
last_print_time = time.time()
REQUEST_LIMIT = int(300 / max_workers)
REQUEST_WINDOW = 60

request_queue = deque(maxlen=REQUEST_LIMIT)
request_lock = Lock()


def generate_message(bedrock_runtime, model_id, messages, max_tokens, top_p,
                     temp):
    string = '''
You have perfect vision and pay great attention to detail which makes you an expert at describing image captions.
Write a new image caption or description based on the image I have given you.
Not more than 20 words.
Describing the content of the image directly without any loose descriptions (with a focus on describing in detail the object instances that are present in the image and how they are related to each other).
Without beginning with something like “this image”.

Your output is formatted as:
Caption: ...
'''

    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "system": string,
        "max_tokens": max_tokens,
        "messages": messages,
        "temperature": temp,
        "top_p": top_p,
    })

    response = bedrock_runtime.invoke_model(body=body, modelId=model_id)
    response_body = json.loads(response.get("body").read())

    return response_body


def claude3_labeling(image):
    message_mm = [
        {
            "role":
                "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": image,
                    },
                },
                {
                    "type": "text",
                    "text": "caption: ",
                },
            ],
        },
    ]
    try:
        result = generate_message(
            bedrock_client,
            model_id="anthropic.claude-3-sonnet-20240229-v1:0",
            # "anthropic.claude-3-haiku-20240307-v1:0",#"anthropic.claude-3-sonnet-20240229-v1:0",#
            messages=message_mm,
            max_tokens=512,
            temp=0.3,
            top_p=0.9,
        )
        return result["content"][0]["text"]
    except Exception as e:
        error_message = f"Error occurred while generating caption: {e}\n{traceback.format_exc()}"
        print(error_message)
        return 'None'


def throttled_claude3_labeling(image_bytes):
    global request_count, last_print_time
    with request_lock:
        # 如果队列已满,则等待到下一个时间窗口
        while len(request_queue) >= REQUEST_LIMIT:
            wait_time = REQUEST_WINDOW - (time.time() - request_queue[0])
            if wait_time > 0:
                time.sleep(wait_time)
            request_queue.clear()

        # 将当前时间加入队列
        request_queue.append(time.time())

    try:
        # 调用 claude3_labeling 函数
        result = claude3_labeling(image_bytes)
    finally:
        request_count += 1

        # 每分钟输出一次请求数量
        current_time = time.time()
        if current_time - last_print_time >= 60:
            print(f"Requests per minute: {request_count}")
            request_count = 0
            last_print_time = current_time

    return result


def process_shard(img_path):
    try:
        with open(img_path, "rb") as image_file:
            image_data = image_file.read()
            image_base64 = base64.b64encode(image_data).decode("utf-8")

        result = throttled_claude3_labeling(image_base64)
        # 写入内容到文件
        os.makedirs("/home/ec2-user/SageMaker/vevor/data/txts", exist_ok=True)
        txt_name = os.path.join("/home/ec2-user/SageMaker/vevor/data/txts",
                                img_path.split("/")[-1].replace('jpg', 'txt'))
        with open(txt_name, 'w', encoding='utf-8') as file:
            file.write(result)

        if result == 'None':
            print(result)
    # print(type(result))

    except:
        print("bug in image reading: ", img_path)
        pass


if __name__ == "__main__":
    save_dir = '/home/ec2-user/SageMaker/vevor/data/images'
    imgs = os.listdir(save_dir)
    imgs = [os.path.join(save_dir, i) for i in imgs]

    start_time = time.time()

    # 创建进度条
    total_tasks = len(imgs)
    with tqdm(total=total_tasks, desc="Processing Images") as pbar:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            futures = [executor.submit(process_shard, shard) for shard in imgs]

            # 处理完成的任务
            for future in as_completed(futures):
                future.result()  # 获取结果（如果有的话）
                pbar.update(1)  # 更新进度条

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")
