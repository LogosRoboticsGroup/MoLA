import os
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from functools import partial

from natsort import natsorted
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from PIL import Image
import cv2
import imageio

import dist_utils
from cotracker.utils.visualizer import Visualizer
from cotracker.predictor import CoTrackerPredictor


def visualize_optical_flow(flow: np.ndarray, convert_to_bgr=False) -> np.ndarray:
    h, w = flow.shape[:2]
    flow_map = np.zeros((h, w, 3), dtype=np.uint8)

    dx = flow[..., 0]
    dy = flow[..., 1]
    magnitude, angle = cv2.cartToPolar(dx, dy, angleInDegrees=True)

    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[..., 0] = (angle / 2).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = np.clip((magnitude * 8), 0, 255).astype(np.uint8)

    flow_map = cv2.cvtColor(
        hsv,
        cv2.COLOR_HSV2BGR if convert_to_bgr else cv2.COLOR_HSV2RGB,
    )
    return flow_map


def load_episode(j, image_key="rgb_static"):
    ep = np.load(os.path.join(data_root, f"episode_{j:0{n_digits}d}.npz"))
    img = ep[image_key]
    img = Image.fromarray(img).resize((224, 224))
    return img


def save_track_label(j, trk, vis):
    file_path = os.path.join(args.save_path, f"{j}.npz")
    if os.path.exists(file_path):
        print(f"File {file_path} already exists, skipping...")
        return

    with open(os.path.join(args.save_path, f"{j}.npz"), "wb") as f:
        np.savez_compressed(
            f,
            tracks=trk,
            visibility=vis,
        )


def get_points_on_a_grid(patch_size, image_size, device):
    if isinstance(patch_size, int):
        patch_size = (patch_size, patch_size)

    H, W = image_size
    ph, pw = patch_size

    assert H % ph == 0 and W % pw == 0

    y_centers = np.arange(ph // 2, H, ph)
    x_centers = np.arange(pw // 2, W, pw)

    xv, yv = np.meshgrid(x_centers, y_centers)
    centers = np.stack([xv, yv], axis=-1).reshape(-1, 2)

    return torch.from_numpy(centers).to(device)


class CalvinDataset(Dataset):

    def __init__(
        self,
        data_root,
        frame_gap,
        grid_query_frame,
        image_key="rgb_static",
        override_exist_files=False,
        except_lang=False,
    ):
        super().__init__()

        self.data_root = data_root
        self.frame_gap = frame_gap
        self.grid_query_frame = grid_query_frame
        self.image_key = image_key

        self.lang = {"info": {"indx": None}}

        episode_files = natsorted(
            [
                f
                for f in os.listdir(self.data_root)
                if f.startswith("episode_") and f.endswith(".npz")
            ]
        )

        episode_ids = []

        for f in episode_files:
            try:
                num_str = f[len("episode_") : -len(".npz")]
                episode_ids.append(int(num_str))
            except Exception:
                continue

        episode_ids = natsorted(episode_ids)

        ep_start_end_ids = []

        if len(episode_ids) > 0:
            start = episode_ids[0]
            prev = episode_ids[0]

            for eid in episode_ids[1:]:
                if eid == prev + 1:
                    prev = eid
                else:
                    ep_start_end_ids.append((start, prev))
                    start = eid
                    prev = eid

            ep_start_end_ids.append((start, prev))

        chunk_size = 1000
        ep_start_end_ids_chunked = []

        for s, e in ep_start_end_ids:
            cur = s

            while cur <= e:
                ce = min(cur + chunk_size - 1, e)
                ep_start_end_ids_chunked.append((cur, ce))
                cur = ce + 1

        self.lang["info"]["indx"] = ep_start_end_ids_chunked
        self.override_exist_files = override_exist_files

    def __len__(self):
        return len(self.lang["info"]["indx"])

    def __getitem__(self, idx):
        start_idx, end_idx = self.lang["info"]["indx"][idx]

        if not self.override_exist_files:
            flag = True

            for j in range(start_idx, end_idx + 1):
                if f"{j}.npz" not in all_exists or (f"{j}.npz" in seq_to_process):
                    flag = False
                    break

            if flag:
                return None

        video = []

        _load_episode = partial(load_episode, image_key=self.image_key)
        video = list(loader.map(_load_episode, range(start_idx, end_idx + 1)))

        video = np.stack(video)
        video = torch.from_numpy(video).permute(0, 3, 1, 2).float()

        if video.shape[0] < self.frame_gap + 1:
            video = torch.zeros((0, 2, *video.shape[1:]))

        else:
            video = video.unfold(0, self.frame_gap + 1, 1)
            video = video[..., [0, -1]]
            video = video.permute(0, 4, 1, 2, 3).contiguous()

        queries = torch.cat(
            [
                torch.ones_like(grid_pts[:, :, :1]) * self.grid_query_frame,
                grid_pts,
            ],
            dim=2,
        ).repeat(video.shape[0], 1, 1)

        ret = dict(
            video=video,
            queries=queries,
            start_idx=start_idx,
            end_idx=end_idx,
        )

        return ret

    @staticmethod
    def collect_fn(batches):
        assert len(batches) == 1
        return batches[0]


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--save_path",
        default=os.getenv("OXE_FLOW_DIR", "outputs/oxe_flow/berkeley_autolab_ur5/image"),
        type=str,
    )

    parser.add_argument(
        "--data_root",
        default=os.getenv("OXE_VIDEO_DIR", "outputs/oxe_video/berkeley_autolab_ur5/image"),
        type=str,
    )

    parser.add_argument(
        "--split",
        default="training",
        type=str,
    )

    parser.add_argument(
        "--image_key",
        default="rgb_static",
        type=str,
    )

    parser.add_argument(
        "--checkpoint",
        default=os.getenv("COTRACKER_CKPT"),
        help="CoTracker model parameters",
    )

    parser.add_argument("--patch_size", type=int, default=8)

    parser.add_argument(
        "--grid_query_frame",
        type=int,
        default=0,
        help="Compute dense and grid tracks starting from this frame",
    )

    parser.add_argument(
        "--backward_tracking",
        action="store_true",
        help="Compute tracks in both directions, not only forward",
    )

    parser.add_argument(
        "--use_v2_model",
        action="store_true",
        help="Pass it if you wish to use CoTracker2, CoTracker++ is the default now",
    )

    parser.add_argument(
        "--offline",
        action="store_true",
        default=True,
        help="Pass it if you would like to use the offline model, in case of online don't pass it",
    )

    parser.add_argument("--start_idx", default=0, type=int)

    parser.add_argument("--end_idx", default=100000000, type=int)

    parser.add_argument("--seq_to_process", type=str, default=None)

    parser.add_argument(
        "--override_exist_files",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--frame_gap",
        default=3,
        type=int,
    )

    parser.add_argument(
        "--except_lang",
        action="store_true",
        default=False,
    )

    args = parser.parse_args()

    if dist_utils.is_dist():
        dist_utils.ddp_setup()
        rank, world_size = dist_utils.get_rank(), dist_utils.get_world_size()
    else:
        rank, world_size = 0, 1

    print(f"Rank: {rank}, World size: {world_size}")

    if rank == 0:
        os.makedirs(args.save_path, exist_ok=True)

    data_root = args.data_root

    if args.checkpoint is not None:

        if args.use_v2_model:
            model = CoTrackerPredictor(
                checkpoint=args.checkpoint,
                v2=args.use_v2_model,
            )

        else:

            if args.offline:
                window_len = 60
            else:
                window_len = 16

            model = CoTrackerPredictor(
                checkpoint=args.checkpoint,
                v2=args.use_v2_model,
                offline=args.offline,
                window_len=window_len,
            )

    else:
        model = torch.hub.load(
            "facebookresearch/co-tracker",
            "cotracker3_offline",
        )

    model = model.to("cuda")
    model.eval()

    H, W = 224, 224

    grid_pts = (
        get_points_on_a_grid(args.patch_size, [H, W], device="cpu")
        .float()
        .unsqueeze(0)
    )

    grid_size = H // args.patch_size

    mp4_files = sorted(
        [
            f
            for f in os.listdir(data_root)
            if f.lower().endswith(".mp4")
        ]
    )

    if len(mp4_files) == 0:
        print(f"No mp4 files found in {data_root}")

        if dist_utils.is_dist():
            dist_utils.ddp_cleanup()

        raise SystemExit(0)

    if args.seq_to_process is not None:

        with open(args.seq_to_process, "r") as f:
            seq_to_process = [item.strip() for item in f if item.strip()]

        seq_to_process = set(seq_to_process)

    else:
        seq_to_process = None

    existing_outputs = set(
        entry.name
        for entry in os.scandir(args.save_path)
        if entry.is_file()
    )

    selected_files = []

    for idx, fname in enumerate(mp4_files):

        if idx < args.start_idx or idx > args.end_idx:
            continue

        if seq_to_process is not None:
            base = os.path.splitext(fname)[0]

            if fname not in seq_to_process and base not in seq_to_process:
                continue

        selected_files.append(fname)

    files_for_rank = selected_files[rank::world_size]

    print(f"Rank {rank}: processing {len(files_for_rank)} videos")

    base_queries = torch.cat(
        [
            torch.ones_like(grid_pts[:, :, :1]) * args.grid_query_frame,
            grid_pts,
        ],
        dim=2,
    )

    batch_size = 32

    def read_video_frames(video_path, size=(224, 224)):
        reader = imageio.get_reader(video_path)

        frames = []

        for frame in reader:
            img = Image.fromarray(frame).resize(size)
            frames.append(np.array(img))

        reader.close()

        if len(frames) == 0:
            return None

        video_np = np.stack(frames)

        video = torch.from_numpy(video_np).permute(0, 3, 1, 2).float()

        return video

    for fname in tqdm(files_for_rank):

        input_path = os.path.join(data_root, fname)
        base_name = os.path.splitext(fname)[0]

        output_name = base_name + ".pt"
        output_path = os.path.join(args.save_path, output_name)

        if (
            (not args.override_exist_files)
            and (output_name in existing_outputs or os.path.exists(output_path))
        ):
            print(f"File {output_name} already exists, skipping...")
            continue

        video = read_video_frames(input_path, size=(H, W))

        if video is None:
            print(f"Warning: {fname} has no frames, skipping...")
            continue

        num_frames = video.shape[0]
        num_points = grid_pts.shape[1]

        all_tracks = torch.zeros((num_frames, num_points, 2), dtype=torch.float32)
        all_visibility = torch.ones((num_frames, num_points), dtype=torch.bool)

        if num_frames <= args.frame_gap:

            torch.save(
                {
                    "tracks": all_tracks,
                    "visibility": all_visibility,
                },
                output_path,
            )

            continue

        video_pairs = video.unfold(0, args.frame_gap + 1, 1)
        video_pairs = video_pairs[..., [0, -1]]
        video_pairs = video_pairs.permute(0, 4, 1, 2, 3).contiguous()

        num_pairs = video_pairs.shape[0]

        pred_tracks_list = []
        pred_visibility_list = []

        for start in range(0, num_pairs, batch_size):

            end_idx = min(start + batch_size, num_pairs)

            video_batch = video_pairs[start:end_idx].cuda(non_blocking=True)

            queries_batch = base_queries.repeat(end_idx - start, 1, 1).cuda(
                non_blocking=True
            )

            with torch.no_grad():

                pred_tracks_batch, pred_visibility_batch = model(
                    video_batch,
                    queries=queries_batch,
                    grid_size=grid_size,
                    backward_tracking=args.backward_tracking,
                )

            pred_tracks_list.append(pred_tracks_batch)
            pred_visibility_list.append(pred_visibility_batch)

        if len(pred_tracks_list) > 0:

            pred_tracks = torch.cat(pred_tracks_list, dim=0)
            pred_visibility = torch.cat(pred_visibility_list, dim=0)

            pred_tracks_delta = (
                pred_tracks[:, 1:2, :, :] - pred_tracks[:, 0:1, :, :]
            ).squeeze(1)

            all_tracks[:num_pairs] = pred_tracks_delta.cpu()
            all_visibility[:num_pairs] = pred_visibility[:, 1, :].cpu()

        torch.save(
            {
                "tracks": all_tracks,
                "visibility": all_visibility,
            },
            output_path,
        )

    if dist_utils.is_dist():
        dist_utils.ddp_cleanup()
