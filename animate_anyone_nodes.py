


import os
import torch
import numpy as np
from PIL import Image
from .pipeline_stable_video_diffusion_controlnet_long import StableVideoDiffusionPipelineControlNet
from .controlnet_sdv import ControlNetSDVModel
#from diffusers import T2IAdapter
from .unet_spatio_temporal_condition_controlnet import UNetSpatioTemporalConditionControlNetModel
import re 
import folder_paths


def load_images_from_folder_to_pil(folder, target_size=(512, 512)):
    images = []
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"}  # Add or remove extensions as needed

    def frame_number(filename):
        matches = re.findall(r'\d+', filename)  # Find all sequences of digits in the filename
        if matches:
            if matches[-1] == '0000' and len(matches) > 1:
                return int(matches[-2])  # Return the second-to-last sequence if the last is '0000'
            return int(matches[-1])  # Otherwise, return the last sequence
        return float('inf')  # Return 'inf'


    # Sorting files based on frame number
    sorted_files = sorted(os.listdir(folder))

    # Load, resize, and convert images
    for filename in sorted_files:
        ext = os.path.splitext(filename)[1].lower()
        if ext in valid_extensions:
            img = Image.open(os.path.join(folder, filename)).convert('RGB')
            images.append(img)

    return images[::2]


class AnimateAnyone:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ('IMAGE',),
                "pose_images": ('IMAGE',)
            },
            "optional": {
                "width": ('INT', {'default': 512, 'min': 64, 'max': 10240, 'step': 64}),
                "height": ('INT', {'default': 768, 'min': 64, 'max': 10240, 'step': 64}),
                "frames_per_batch": ('INT', {'default': 14, 'min': 1, 'max': 1024}),
                "steps": ('INT', {'default': 25, 'min': 1, 'max': 1024}),
                "fps": ('INT', {'default': 7, 'min': 1, 'max': 1024})
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = 'run_inference'
    CATEGORY = 'AnymateAnyone'

    def _tensor_to_pil(self, tensor_img):
        new_tensor_img = tensor_img * 255

        pil_img_list = []
        for img in new_tensor_img:
           np_img = img.to(torch.uint8).cpu().numpy()
           pil_img = Image.fromarray(np_img).convert('RGB')
           pil_img_list.append(pil_img)
        return pil_img_list

    

    

    def _pil_to_tensor(self, images:list):
        """
        Input:
            images: list of pil.Image
        """
        w,h = images[0].size
        tensor_imgs = torch.zeros([len(images), h, w, 3], dtype=torch.float32)
        for i in range(len(images)):
            np_img = np.array(images[i])
            tensor_img = torch.from_numpy(np_img.astype(np.float32) / 255.)  
            tensor_imgs[i] = tensor_img
        return tensor_imgs

    
    def run_inference(self, image, pose_images, 
                      width=512, height=768,
                      frames_per_batch=14, steps=25, fps=7):

        args = {
            "pretrained_model_name_or_path": "models/animate_anyone",
        }
        assert width%64 ==0, "`height` and `width` have to be divisible by 64"
        assert height%64 ==0, "`height` and `width` have to be divisible by 64"

        # convert image from tensor to pil.Image
        ref_images = self._tensor_to_pil(image)
        pose_imgs_pil = self._tensor_to_pil(pose_images)

        # load checkpoints
        controlnet = ControlNetSDVModel.from_pretrained(args["pretrained_model_name_or_path"] + "/controlnet")
        unet = UNetSpatioTemporalConditionControlNetModel.from_pretrained(args["pretrained_model_name_or_path"]+'/unet')
        
        pipeline = StableVideoDiffusionPipelineControlNet.from_pretrained(args["pretrained_model_name_or_path"], controlnet=controlnet, unet=unet)
        pipeline.to(dtype=torch.float16)
        pipeline.enable_model_cpu_offload()
        
        print('Model loading: DONE.')

        # val_save_dir = os.path.join(folder_paths.get_output_directory(), "animate_anyone")
        # os.makedirs(val_save_dir, exist_ok=True)

        # Inference and saving loop
        final_result = []
        num_frames = len(pose_imgs_pil)
        if num_frames <= frames_per_batch:
            nb_append = frames_per_batch + 1 - num_frames
            pose_imgs_pil.extend([pose_imgs_pil[-1]] * nb_append)
        
        print('Image and frame data loading: DONE.')

        import time
        start = time.time()

        for ref_image in ref_images:
            video_frames = pipeline(ref_image, pose_imgs_pil, decode_chunk_size=2,num_frames=len(pose_imgs_pil),motion_bucket_id=127.0, 
                                    fps=fps,controlnet_cond_scale=1.0, width=width, height=height, 
                                    min_guidance_scale=3.5, max_guidance_scale=3.5, frames_per_batch=frames_per_batch, num_inference_steps=steps, overlap=4).frames[0]
            # [video_frames[i].save(f"{val_save_dir}/{i}.jpg") for i in range(len(video_frames))]
            final_result.extend(video_frames[:num_frames])

        end = time.time()
        print(f"Elipsed time: {end - start}")

        tensor_results = self._pil_to_tensor(final_result)
        return (tensor_results, )






NODE_CLASS_MAPPINGS = {
    "AnimateAnyone":AnimateAnyone,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AnimateAnyone": "AnimateAnyone"
}
