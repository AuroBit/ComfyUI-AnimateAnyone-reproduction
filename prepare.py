
import os
import requests

file_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__))) + '/models/animate_anyone'
print(file_dir)

os.makedirs(file_dir, exist_ok=True)


model_files = [
    ('model_index.json', 
        'https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt/resolve/main/model_index.json?download=true'),
    ('feature_extractor/preprocessor_config.json', 
        'https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt/resolve/main/feature_extractor/preprocessor_config.json?download=true'),
    ('image_encoder/config.json', 
        'https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt/resolve/main/image_encoder/config.json?download=true'),
    ('image_encoder/model.safetensors', 
        'https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt/resolve/main/image_encoder/model.fp16.safetensors?download=true'),
    ('scheduler/scheduler_config.json', 
        'https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt/resolve/main/scheduler/scheduler_config.json?download=true'),
    ('vae/config.json', 
        'https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt/resolve/main/vae/config.json?download=true'),
    ('vae/diffusion_pytorch_model.safetensors', 
        'https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt/resolve/main/vae/diffusion_pytorch_model.fp16.safetensors?download=true'),
    
    
    ('unet/config.json', 
        'https://modelscope.cn/api/v1/models/lightnessly/animate-anyone-v1/repo?Revision=master&FilePath=unet/config.json'),
    ('unet/diffusion_pytorch_model.safetensors', 
        'https://modelscope.cn/api/v1/models/lightnessly/animate-anyone-v1/repo?Revision=master&FilePath=unet/diffusion_pytorch_model.safetensors'),    
    ('controlnet/config.json', 
        'https://modelscope.cn/api/v1/models/lightnessly/animate-anyone-v1/repo?Revision=master&FilePath=controlnet/config.json'),
    ('controlnet/diffusion_pytorch_model.safetensors', 
        'https://modelscope.cn/api/v1/models/lightnessly/animate-anyone-v1/repo?Revision=master&FilePath=controlnet/diffusion_pytorch_model.safetensors'),
    
]



for file_info in model_files:
    file_name, file_url = file_info
    file_path = os.path.join(file_dir, file_name)
    print(f"Start download file: {file_url}")
    print(f"    to: {file_path}")
    file_dirname = os.path.dirname(file_path)
    os.makedirs(file_dirname, exist_ok=True)

    if not os.path.exists(file_path):
        response = requests.get(file_url)
        with open(file_path, "wb") as f:
            f.write(response.content)
        print(f"Done: {file_path}")
    else:
        print(f"File exist: {file_path}")
