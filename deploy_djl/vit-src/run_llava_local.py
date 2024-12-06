import json
'''
#install djl locally
git clone https://github.com/deepjavalibrary/djl-serving.git
cd djl-serving
cd engines/python/setup
pip install -U -e .
'''
from djl_python.inputs import Input
from model import handle

# ---------- Initialization (Model loading) -------------
init_input = Input()
init_input.properties = {
    "tensor_parallel_degree": 1,
    "model_dir": "../../model_v2_10epoch"
}
handle(init_input)


# ---------- Invocation -------------
prompt_input = Input()
prompt_input.properties = {
    "Content-Type": "application/json"
}

prompt = "Describe the image"
payload = bytes(json.dumps(
        {
            "text": [prompt],
            "input_image": "https://raw.githubusercontent.com/haotian-liu/LLaVA/main/images/llava_logo.png",
        }
    ), 'utf-8')
prompt_input.content.add(key="data", value=payload)
model_output = handle(prompt_input)
print(model_output)

output = str(model_output.content.value_at(0), "utf-8")
print(output)
