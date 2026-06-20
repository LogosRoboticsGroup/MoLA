
import os
import random
import numpy as np
import torch
import wandb
import clip
from torch.nn.parallel import DistributedDataParallel as DDP
from utils.distributed_utils import init_distributed_device, world_info_from_env

from utils.eval_utils_libero import eval_one_epoch_libero_ddp

# try:
#     from utils.eval_utils_libero import eval_one_epoch_libero_ddp as eval_one_epoch_calvin_ddp
# except:
#     pass 
# from utils.eval_utils_libero import eval_one_epoch_libero_ddp as eval_one_epoch_calvin_ddp
from torch.distributed.elastic.multiprocessing.errors import record
# from utils.arguments_utils import get_args_and_cfg
from utils.arguments_utils import get_parser

import hydra
from hydra import compose, initialize

def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)

@record
def main():
    parser = get_parser(is_eval=True)
    parser.add_argument("--video_model_path", type=str, default=os.getenv("VIDEO_MODEL_PATH", "stabilityai/stable-video-diffusion-img2vid"))
    parser.add_argument("--clip_model_path", type=str, default=os.getenv("CLIP_MODEL_PATH", "openai/clip-vit-base-patch32"))
    args = parser.parse_args()
    args.local_rank, args.rank, args.world_size = world_info_from_env()
    device_id = init_distributed_device(args)
    print("device_id: ", device_id)
    random_seed(args.seed)

    # model
    with initialize(config_path="policy_conf", job_name="calvin_evaluate_all.yaml"):
        cfg = compose(config_name="calvin_evaluate_all.yaml")
    cfg.model.pretrained_model_path = args.video_model_path
    cfg.model.text_encoder_path = args.clip_model_path
    model = hydra.utils.instantiate(cfg.model)
    model.num_sampling_steps = cfg.num_sampling_steps
    model.sampler_type = cfg.sampler_type
    model.multistep = cfg.multistep
    if cfg.sigma_min is not None:
        model.sigma_min = cfg.sigma_min
    if cfg.sigma_max is not None:
        model.sigma_max = cfg.sigma_max
    if cfg.noise_scheduler is not None:
        model.noise_scheduler = cfg.noise_scheduler

    random_seed(args.seed, args.rank)
    print(f"Start running training on rank {args.rank}.")

    device_id = args.rank % torch.cuda.device_count()
    model = model.to(device_id)
  
    ddp_model = DDP(model, device_ids=[device_id], find_unused_parameters=True)

    if args.resume_from_checkpoint is not None:
        if args.rank == 0:
            print(f"Loading checkpoint from {args.resume_from_checkpoint}")
        state_dict = torch.load(args.resume_from_checkpoint, map_location='cpu')
        missing_keys, unexpected_keys = ddp_model.module.load_state_dict(
            state_dict['model']
        )


    ddp_model.module.freeze()
    ddp_model.module.process_device()
    ddp_model.eval()

    image_processor = clip.clip._transform(n_px=256)

    eval_one_epoch_libero_ddp(
        args=args,
        model=ddp_model,
        image_processor=image_processor,
    )

if __name__ == "__main__":
    os.environ["NCCL_BLOCKING_WAIT"] = "0"
    main()
