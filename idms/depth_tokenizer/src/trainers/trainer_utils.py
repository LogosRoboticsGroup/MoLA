import torch.nn.functional as F
import math
import cv2
from PIL import Image, ImageFont, ImageDraw
import os
import torchvision.transforms as T
import numpy as np

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

def visualize_latent_motion_reconstruction_depth(
    initial_depth,
    next_depth,
    recons_next_depth,
    latent_motion_ids,
    path,
):
    c, h, w = initial_depth.shape
    h = h + 30
    latent_motion_ids = latent_motion_ids.cpu().numpy().tolist()

    def tensor_to_colormap_pil(depth_tensor):
        depth_np = depth_tensor.squeeze().cpu().numpy()
        depth_normalized = cv2.normalize(
            depth_np, None, 0, 255, cv2.NORM_MINMAX
        )
        depth_uint8 = depth_normalized.astype(np.uint8)
        depth_colormap = cv2.applyColorMap(
            depth_uint8, cv2.COLORMAP_JET
        )
        depth_colormap = cv2.cvtColor(
            depth_colormap, cv2.COLOR_BGR2RGB
        )
        return Image.fromarray(depth_colormap)

    initial_depth = tensor_to_colormap_pil(initial_depth)
    next_depth = tensor_to_colormap_pil(next_depth)
    recons_next_depth = tensor_to_colormap_pil(recons_next_depth)

    compare_img = Image.new("RGB", size=(3 * w, h))
    draw_compare_img = ImageDraw.Draw(compare_img)

    compare_img.paste(initial_depth, box=(0, 0))
    compare_img.paste(next_depth, box=(w, 0))
    compare_img.paste(recons_next_depth, box=(2 * w, 0))

    try:
        font_path = os.path.join(
            cv2.__path__[0], "qt", "fonts", "DejaVuSans.ttf"
        )
        font = ImageFont.truetype(font_path, size=12)
    except:
        font = ImageFont.load_default()

    draw_compare_img.text(
        (w, h - 20),
        f"{latent_motion_ids}",
        font=font,
        fill=(255, 255, 255),
    )

    compare_img.save(path)
