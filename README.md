# ComfyUI-AnimateAnyone-reproduction

A ComfyUI custom node that simply integrates the [animate-anyone-reproduction](https://github.com/bendanzzc/AnimateAnyone-reproduction) functionality.

一个简单接入 animate-anyone-reproduction 的 ComfyUI 节点。


## Instruction 指南

We use 'COMFYUI_PATH' to represent your comfyui directory.
我们用'COMFYUI_PATH'表示你comfyui的目录

### clone repo 克隆仓库
* Clone this repo into 'COMFYUI_PATH/custum_nodes' 将这个仓库克隆到'COMFYUI_PATH/custum_nodes'目录下
```txt
git clone https://github.com/AuroBit/ComfyUI-Animate-Anyone-reproduction.git custom_nodes/ComfyUI-Animate-Anyone-reproduction
```
* install dependences: 安装依赖
```txt
pip install -r custom_nodes/ComfyUI-Animate-Anyone-reproduction/requirements.txt
```


### prepare checkpoints files 准备模型文件
#### Method 1: clone file repos from huggingface and modelscope. 方法1：通过clone仓库下载模型文件

* clone (or download all files) the SVD model repo from: https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt/tree/main  to any where you like. 将[SVD模型](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt/tree/main)克隆到一个你喜欢的目录下。
* Create a folder named 'animate_anyone' under the 'COMFYUI_PATH/models' folder. 在'COMFYUI_PATH/models'下创建一个'animate_anyone'文件夹
* Copy all files and folders (except the 'unet' folder) to 'COMFYUI_PATH/models/animate_anyone' 将所有文件和文件夹（除了'unet'文件夹）复制到'COMFYUI_PATH/models/animate_anyone'下面

* clone (or download model and files) from: https://modelscope.cn/models/lightnessly/animate-anyone-v1/files 克隆原作者训练的[模型仓库](https://modelscope.cn/models/lightnessly/animate-anyone-v1/files)
   * Copy the 'controlnet' folder to the 'COMFYUI_PATH/models/animate_anyone' folder.
   * Copy the 'unet' folder to the 'COMFYUI_PATH/models/animate_anyone' folder.

* Your file structure should look like this: 下载完后你的模型文件夹的目录结构应该是这样的
```txt
- ComfyUI
  ...
  - models
    - animate_anyone
      - controlnet
        config.json
        diffusion_pytorch_model.safetensors
      - feature_extractor
        preprocessor_config.json
      - image_encoder
        ...
      - scheduler
        ...
      - unet
        ...
      - vae
        ...
      ...
```


#### Method 2
Download all these file automatically using python script
通过python脚本自动下载模型文件

* run:
```txt
python custom_nodes\ComfyUI-AnimateAnyon-reproduction\prepare.py
```


## Examples
[]