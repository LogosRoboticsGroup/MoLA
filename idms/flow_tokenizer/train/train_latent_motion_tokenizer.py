import pyrootutils
pyrootutils.setup_root(__file__, indicator='.project-root', pythonpath=True, dotenv=True)
import argparse
import json
from pathlib import Path
from torch.utils.data import DataLoader
import omegaconf
import hydra
from functools import partial
from transformers import AutoTokenizer
from common.models.model_utils import load_model
from common.processors.preprocessor_utils import get_rgb_preprocessor
from flow_tokenizer.src.trainers.latent_motion_tokenizer_trainer import LatentMotionTokenizer_Trainer
from torch.utils.data import DataLoader
from functools import partial
from common.data.data_utils import load_dataset

def _resolve_config_refs(cfg, config_path):
    config_dir = Path(config_path).resolve().parent
    for key in ("latent_motion_tokenizer_config_path", "dataset_config_path"):
        path = Path(cfg[key])
        if not path.is_absolute():
            cfg[key] = str((config_dir / path).resolve())
    save_path = Path(cfg["training_config"]["save_path"])
    if not save_path.is_absolute():
        cfg["training_config"]["save_path"] = str((config_dir / save_path).resolve())
    return cfg

def main(cfg):
    # Prepare Latent Motion Tokenizer
    latent_motion_tokenizer_config_path = cfg['latent_motion_tokenizer_config_path']
    print(f"initializing Latent Motion Tokenizer from {latent_motion_tokenizer_config_path} ...")
    latent_motion_tokenizer_config = omegaconf.OmegaConf.load(latent_motion_tokenizer_config_path)
    latent_motion_tokenizer = hydra.utils.instantiate(latent_motion_tokenizer_config)
    latent_motion_tokenizer.config = latent_motion_tokenizer_config

    # Prepare rgb_processor
    rgb_preprocessor = get_rgb_preprocessor(**cfg['rgb_preprocessor_config'])

    # Preprepare Dataloaders
    dataset_config_path = cfg['dataset_config_path']
    extra_data_config = {
        'sequence_length': 1,
        'do_extract_future_frames': True,
        'do_extract_action': False
    }
    train_dataset, eval_dataset = load_dataset(dataset_config_path, extra_data_config, "flow")
    dataloader_cls = partial(
        DataLoader, 
        pin_memory=True, # Accelerate data reading
        shuffle=True,
        persistent_workers=True,
        num_workers=cfg['dataloader_config']['workers_per_gpu'],
        batch_size=cfg['dataloader_config']['bs_per_gpu'],
        prefetch_factor= cfg['dataloader_config']['prefetch_factor']
    )
    train_dataloader = dataloader_cls(train_dataset)
    eval_dataloader = dataloader_cls(eval_dataset)
    
    # Prepare Trainer
    trainer = LatentMotionTokenizer_Trainer(
        latent_motion_tokenizer=latent_motion_tokenizer,
        rgb_preprocessor=rgb_preprocessor,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        bs_per_gpu=cfg['dataloader_config']['bs_per_gpu'],
        **cfg['training_config']
    )

    # Start Training
    trainer.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default="../configs/train/calvin.yaml")
    args = parser.parse_args()

    cfg = omegaconf.OmegaConf.load(args.config_path)
    cfg = _resolve_config_refs(cfg, args.config_path)
    main(cfg)

    
