import os
import argparse
from typing import Sequence
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from natsort import natsorted
import glob
from tqdm import tqdm
import numpy as np
import torch
import torchvision.transforms as tvt
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import imageio
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
import dist_utils


def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    img = np.ones((sorted_anns[0]['segmentation'].shape[0],
                   sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)


def load_episode(data_root, j, image_key='rgb_static'):
    n_digits = 7
    ep = np.load(os.path.join(data_root, f'episode_{j:0{n_digits}d}.npz'))
    img = ep[image_key]
    img = Image.fromarray(img).resize((224, 224))
    return img


class CalvinDataset(Dataset):
    def __init__(self, data_root, input_size, image_key='rgb_static'):
        super().__init__()
        self.data_root = data_root
        self.image_key = image_key
        npz_pattern = os.path.join(data_root, 'episode_*.npz')
        npz_files = glob.glob(npz_pattern)
        if len(npz_files) == 0:
            raise RuntimeError(f"No npz files found under {npz_pattern}")
        indices = []
        for f in natsorted(npz_files):
            base = os.path.basename(f)
            name, _ = os.path.splitext(base)
            try:
                idx = int(name.split('_')[-1])
                indices.append(idx)
            except ValueError:
                continue
        self.all_indx = indices
        self.transform = ResizeLongestSide(input_size)

    def __len__(self):
        return len(self.all_indx)

    def __getitem__(self, idx):
        file_name = self.all_indx[idx]
        img = load_episode(self.data_root, file_name, self.image_key)
        img = np.array(img)
        img = self.transform.apply_image(img)
        input_image_torch = torch.as_tensor(img)
        input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()
        return {'img': input_image_torch, 'idx': file_name}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path",
                        default=os.getenv("CALVIN_SEMANTIC_DIR", "outputs/calvin_semantic"),
                        type=str)
    parser.add_argument("--data_root",
                        default=os.getenv("CALVIN_DATASET_DIR", "/path/to/calvin/task_ABC_D"),
                        type=str)
    parser.add_argument("--split",
                        default="training",
                        type=str)
    parser.add_argument("--image_key",
                        default='rgb_static',
                        type=str)
    parser.add_argument("--checkpoint",
                        default=os.getenv("SAM_CKPT", "checkpoints/sam_vit_b_01ec64.pth"),
                        help="SAM model parameters")
    parser.add_argument("--start_idx",
                        default=0,
                        type=int)
    parser.add_argument("--end_idx",
                        default=100000000,
                        type=int)
    parser.add_argument("--seq_to_process",
                        type=str,
                        default=None)
    parser.add_argument("--override_exist_files",
                        action="store_true",
                        default=False)
    parser.add_argument('--batch_size',
                        type=int,
                        default=16)

    args = parser.parse_args()
    args.save_path = os.path.join(args.save_path, args.image_key, args.split)

    if dist_utils.is_dist():
        dist_utils.ddp_setup()
        rank, world_size = dist_utils.get_rank(), dist_utils.get_world_size()
    else:
        rank, world_size = 0, 1
    print('rank: ', rank, 'world_size: ', world_size)

    if rank == 0:
        os.makedirs(args.save_path, exist_ok=True)
    data_root = os.path.join(args.data_root, args.split)

    model_type = '_'.join(os.path.basename(args.checkpoint).split('_')[1:3])
    sam = sam_model_registry[model_type](checkpoint=args.checkpoint)
    sam.to('cuda')

    mask_generator = SamAutomaticMaskGenerator(sam)

    dataset = CalvinDataset(data_root, sam.image_encoder.img_size, args.image_key)
    if dist_utils.is_dist():
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
    else:
        sampler = None
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, sampler=sampler)

    total_len = len(dataloader)
    print(f"[Rank {rank}] Total batches: {total_len}")

    for batch in tqdm(dataloader, desc=f"[Rank {rank}]"):
        imgs = batch['img'].to('cuda')
        output_mask = False
        with torch.no_grad():
            if output_mask:
                raise NotImplementedError
            else:
                imgs = sam.preprocess(imgs)
                features = sam.image_encoder(imgs)
                features = torch.nn.functional.avg_pool2d(features, kernel_size=4, stride=4, padding=0)
                features = features.flatten(start_dim=-2)

        for i in range(len(batch['idx'])):
            torch.save(features[i].to(torch.bfloat16).cpu(),
                       os.path.join(args.save_path, f'{batch["idx"][i].item()}.pt'))

    if dist_utils.is_dist():
        dist_utils.ddp_cleanup()
