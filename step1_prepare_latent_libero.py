
# import tensorflow_datasets as tfds
import cv2
import mediapy
import os
import random
import math
from diffusers.models import AutoencoderKL
# from decord import VideoReader, cpu
import mediapy
import torch
import numpy as np
import json
from diffusers.models import AutoencoderKL,AutoencoderKLTemporalDecoder
import mediapy
import re

def load_hdf5(path):
    demos = []

    filename = os.path.basename(path)
    task_text = os.path.splitext(filename)[0]
    task_text = task_text.replace("_", " ")

    with h5py.File(path, 'r') as f:
        for demo_name in sorted(f["data"].keys()):
            demo = f["data"][demo_name]
            obs = demo["obs"]

            image_dict = {}
            for k in sorted(obs.keys()):
                image_dict[k] = obs[k][()]

            demos.append({
                "text": task_text,
                "images": image_dict,
            })

    return demos


############## raw data paths (teleoperation hdf5 files) ###############
raw_data_path = os.getenv("LIBERO_SOURCE_DIR", "/path/to/LIBERO-datasets")
task_name = "libero"
dir = os.getenv("LATENT_OUTPUT_DIR", f"./outputs/opensource_robotdata/{task_name}")
############## saved paths ###############
video_dir = os.path.join(dir, 'videos')
latent_video_dir = os.path.join(dir, 'latent_videos')
anno_dir = os.path.join(dir, 'annotation')
os.makedirs(video_dir, exist_ok=True)
os.makedirs(latent_video_dir, exist_ok=True)
os.makedirs(anno_dir, exist_ok=True)


import h5py
raw_file = []
subfolder = ['libero_10','libero_90','libero_goal','libero_object','libero_spatial']
# subfolder = ['libero_10']
print(subfolder)
for sub in subfolder:
    sub_path = os.path.join(raw_data_path, sub)
    sub_raw_file = os.listdir(sub_path)
    sub_raw_file = [f for f in sub_raw_file if f.endswith('.hdf5')]
    sub_raw_file.sort()
    sub_raw_file = [os.path.join(sub_path, f) for f in sub_raw_file]
    raw_file += sub_raw_file
####################################################
# start prepare vae latent data
vae = AutoencoderKLTemporalDecoder.from_pretrained(
    os.getenv("SVD_MODEL_PATH", "stabilityai/stable-video-diffusion-img2vid"),
    subfolder="vae",
).to("cuda")

failed_num =0
success_num = 0
anno_counter = 0
for file_num, file_name in enumerate(raw_file):
    demos = load_hdf5(file_name)
    print(file_name)
    if not demos:
        print(f"skip file {file_name} because it contains no demos")
        continue

    for demo_idx, demo in enumerate(demos):
        image_dict = demo['images']
        text = demo['text']

        # anno_ind_all = int(file_name.split('.')[-2].split('_')[-1])
        anno_ind_all = anno_counter
        anno_counter += 1
        data_type = 'val' if anno_ind_all%50==4 else 'train'

        text = re.sub(
            r'(KITCHEN|LIVING\s*ROOM|STUDY)\s+SCENE\s*\d+[_ ]*',
            '',
            text,
            flags=re.IGNORECASE
        )

        text = re.sub(r'\s*demo$', '', text)

        # split 1 trajectory into 5 trajectories if data is record at 50 hz. since the video model always predict 16 frames with frame intervel=0.1s
        skip_step = 5

        num_traj = 1 if data_type == 'val' else 1
        for j in range(num_traj):
            key_in_order = ['agentview_rgb', 'eye_in_hand_rgb']
            latent_key = ['agentview_rgb', 'eye_in_hand_rgb'] #['cam_high']

            anno_ind = skip_step*anno_ind_all+j
            for idx,cam_name in enumerate(key_in_order):
                if cam_name not in image_dict:
                    print(f"missing camera {cam_name} in demo {demo_idx} of file {file_name}")
                    continue
                img_all = image_dict[cam_name]

                if cam_name == 'agentview_rgb':
                    img_all = np.flip(img_all, axis=1)

                img = img_all[j:]
                img = img[::skip_step]

                # crop
                frames = np.array(img)

                
                
                # save latent video
                latent_video_path = f"{dir}/latent_videos/{data_type}/{anno_ind}"
                os.makedirs(latent_video_path, exist_ok=True)
                frames = torch.tensor(frames).permute(0, 3, 1, 2).float().to("cuda") / 255.0*2-1
                # resize to 256*256
                x = torch.nn.functional.interpolate(frames, size=(256, 256), mode='bilinear', align_corners=False)
                resize_video = ((x / 2.0 + 0.5).clamp(0, 1)*255)
                resize_video = resize_video.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
                # save images to video
                video_path = f"{dir}/videos/{data_type}/{anno_ind}"
                os.makedirs(video_path, exist_ok=True)
                mediapy.write_video(f"{dir}/videos/{data_type}/{anno_ind}/{idx}.mp4", resize_video, fps=10)

                img_path = f"{dir}/imgs/{data_type}/{anno_ind}/{idx}.mp4"

                if cam_name in latent_key:
                    with torch.no_grad():
                        batch_size = 64
                        latents = []
                        for i in range(0, len(x), batch_size):
                            batch = x[i:i+batch_size]
                            latent = vae.encode(batch).latent_dist.sample().mul_(vae.config.scaling_factor).cpu()
                            latents.append(latent)
                        x = torch.cat(latents, dim=0)
                    
                    torch.save(x, f"{latent_video_path}/{idx}.pt")
        
        success_num += 1

        # print("text", "success!!!",anno_ind_all,"failed_num", failed_num, "success_num", success_num, action.shape)
        print("text", text, "num", file_num, "total_num", len(raw_file))
        # save anno
        info = {
            "task": "robot_trajectory_prediction",
            "texts": [
                text
            ],
            "videos": [
                {
                    "video_path": f"videos/{data_type}/{anno_ind}/0.mp4"
                },
                {
                    "video_path": f"videos/{data_type}/{anno_ind}/1.mp4"
                }
            ],
            "episode_id": anno_ind,
            "video_length": len(frames),
            "latent_videos": [
                {
                    "latent_video_path": f"latent_videos/{data_type}/{anno_ind}/0.pt"
                },
                {
                    "latent_video_path": f"latent_videos/{data_type}/{anno_ind}/1.pt"
                }
                
            ],
            }
        os.makedirs(f"{dir}/annotation/{data_type}", exist_ok=True)
        with open(f"{dir}/annotation/{data_type}/{anno_ind}.json", "w") as f:
            json.dump(info, f, indent=2)  
