import os
import argparse
from natsort import natsorted
import glob
from tqdm import tqdm
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
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


def encode_frames_with_sam(frames, sam, device):
    if len(frames) == 0:
        return None
    imgs = torch.stack(frames, dim=0).to(device)
    with torch.no_grad():
        imgs = sam.preprocess(imgs)
        features = sam.image_encoder(imgs)
        features = torch.nn.functional.avg_pool2d(features, kernel_size=4, stride=4, padding=0)
        features = features.flatten(start_dim=-2)
    return features


def process_video(video_path, sam, transform, batch_size, device):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    frame_batch = []
    feat_chunks = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = transform.apply_image(frame)
        frame_t = torch.as_tensor(frame)
        frame_t = frame_t.permute(2, 0, 1).contiguous()
        frame_batch.append(frame_t)

        if len(frame_batch) >= batch_size:
            feats = encode_frames_with_sam(frame_batch, sam, device)
            feat_chunks.append(feats.cpu())
            frame_batch = []

    cap.release()

    if frame_batch:
        feats = encode_frames_with_sam(frame_batch, sam, device)
        feat_chunks.append(feats.cpu())

    if len(feat_chunks) == 0:
        return None

    all_feats = torch.cat(feat_chunks, dim=0)
    return all_feats


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path",
                        default=os.getenv("OXE_SEMANTIC_DIR", "outputs/oxe_semantic/berkeley_autolab_ur5/image"),
                        type=str)
    parser.add_argument("--data_root",
                        default=os.getenv("OXE_VIDEO_DIR", "outputs/oxe_video/berkeley_autolab_ur5/image"),
                        type=str)
    parser.add_argument("--split",
                        default="validation",
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

    if dist_utils.is_dist():
        dist_utils.ddp_setup()
        rank, world_size = dist_utils.get_rank(), dist_utils.get_world_size()
    else:
        rank, world_size = 0, 1
    print('rank: ', rank, 'world_size: ', world_size)

    if rank == 0:
        os.makedirs(args.save_path, exist_ok=True)
    data_root = args.data_root

    model_type = '_'.join(os.path.basename(args.checkpoint).split('_')[1:3])
    sam = sam_model_registry[model_type](checkpoint=args.checkpoint)
    device = 'cuda'
    sam.to(device)

    transform = ResizeLongestSide(sam.image_encoder.img_size)

    mp4_pattern = os.path.join(data_root, '*.mp4')
    mp4_files = glob.glob(mp4_pattern)
    if len(mp4_files) == 0:
        raise RuntimeError(f"No mp4 files found under {mp4_pattern}")
    mp4_files = natsorted(mp4_files)
    mp4_files = mp4_files[rank::world_size]
    print(f"[Rank {rank}] Number of videos to process: {len(mp4_files)}")

    for video_path in tqdm(mp4_files, desc=f"[Rank {rank}]"):
        base = os.path.basename(video_path)
        name, _ = os.path.splitext(base)
        out_path = os.path.join(args.save_path, f"{name}.pt")

        if (not args.override_exist_files) and os.path.exists(out_path):
            continue

        feats = process_video(video_path, sam, transform, args.batch_size, device)
        if feats is None:
            continue

        torch.save(feats.to(torch.bfloat16), out_path)

    if dist_utils.is_dist():
        dist_utils.ddp_cleanup()
