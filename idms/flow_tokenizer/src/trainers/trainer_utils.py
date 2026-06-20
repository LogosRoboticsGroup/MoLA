import torch.nn.functional as F
import math
import cv2
from PIL import Image, ImageFont, ImageDraw
import os
import torchvision.transforms as T
import numpy as np
import torch

def visualize_latent_motion_reconstruction(
    initial_frame,
    next_frame,
    recons_next_frame,
    latent_motion_ids,
    path
):
    c, h, w = initial_frame.shape
    h = h + 30
    initial_frame = T.ToPILImage()(initial_frame)
    next_frame = T.ToPILImage()(next_frame)
    recons_next_frame = T.ToPILImage()(recons_next_frame)
    latent_motion_ids = latent_motion_ids.numpy().tolist()

    compare_img = Image.new('RGB', size=(3*w, h))
    draw_compare_img = ImageDraw.Draw(compare_img)

    compare_img.paste(initial_frame, box=(0, 0))
    compare_img.paste(next_frame, box=(w, 0))
    compare_img.paste(recons_next_frame, box=(2*w, 0))

    font_path = os.path.join(cv2.__path__[0],'qt','fonts','DejaVuSans.ttf')
    font = ImageFont.truetype(font_path, size=12)
    draw_compare_img.text((w, h-20), f"{latent_motion_ids}", font=font, fill=(0, 255, 0))
    compare_img.save(path)

def visualize_latent_motion_reconstruction_flow(
    initial_flow,
    next_flow,
    recons_next_flow,
    latent_motion_ids,
    path
):
    initial_flow = initial_flow.permute(1, 2, 0)
    next_flow = next_flow.permute(1, 2, 0)
    recons_next_flow = recons_next_flow.permute(1, 2, 0)
    h, w, _ = initial_flow.shape
    h = h + 30
    latent_motion_ids = latent_motion_ids.cpu().numpy().tolist()

    def visualize_optical_flow(flow: np.ndarray, convert_to_bgr=False) -> np.ndarray:
        if isinstance(flow, torch.Tensor):
            flow = flow.cpu().numpy()

        h, w = flow.shape[:2]
        flow_map = np.zeros((h, w, 3), dtype=np.uint8)

        dx = flow[..., 0]
        dy = flow[..., 1]
        magnitude, angle = cv2.cartToPolar(dx, dy, angleInDegrees=True)

        hsv = np.zeros((h, w, 3), dtype=np.uint8)
        hsv[..., 0] = (angle / 2).astype(np.uint8)
        hsv[..., 1] = 255
        hsv[..., 2] = np.clip((magnitude * 8), 0, 255).astype(np.uint8)
        flow_map = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR if convert_to_bgr else cv2.COLOR_HSV2RGB)
        return flow_map
    
    initial_flow = visualize_optical_flow(initial_flow)
    next_flow = visualize_optical_flow(next_flow)
    recons_next_flow = visualize_optical_flow(recons_next_flow)

    compare_img = Image.new('RGB', size=(3 * w, h))
    draw_compare_img = ImageDraw.Draw(compare_img)

    initial_flow = Image.fromarray(initial_flow)
    next_flow = Image.fromarray(next_flow)
    recons_next_flow = Image.fromarray(recons_next_flow)

    compare_img.paste(initial_flow, box=(0, 0))
    compare_img.paste(next_flow, box=(w, 0))
    compare_img.paste(recons_next_flow, box=(2 * w, 0))

    try:
        font_path = os.path.join(cv2.__path__[0], 'qt', 'fonts', 'DejaVuSans.ttf')
        font = ImageFont.truetype(font_path, size=2)
    except:
        font = ImageFont.load_default()

    draw_compare_img.text((w, h - 20), f"{latent_motion_ids}", font=font, fill=(255, 255, 255))

    compare_img.save(path)
