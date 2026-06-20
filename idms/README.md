# Mixture of Inverse Dynamics Models

This directory contains the semantic, depth, and flow inverse dynamics tokenizers used by MoLA.

## Installation

```bash
cd idms
conda create -n mola-idm python=3.8
conda activate mola-idm
pip install -r requirements.txt
```

## Data Paths

Set the paths for the dataset and modality supervision before training:

Dataset Downloads:

- CALVIN ABC-D can be downloaded with the official [CALVIN dataset script](https://github.com/mees/calvin/blob/main/dataset/README.md).
- LIBERO hdf5 datasets are available at [Hugging Face: `yifengzhu-hf/LIBERO-datasets`](https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets/tree/main)

Checkpoint downloads:

- Depth Anything V2 ViT-B: [Depth-Anything-V2-Base](https://huggingface.co/depth-anything/Depth-Anything-V2-Base/tree/main)
- CoTracker3 scaled offline: [facebook/cotracker3](https://huggingface.co/facebook/cotracker3/tree/main)
- SAM ViT-B: [Segment Anything model checkpoints](https://github.com/facebookresearch/segment-anything#model-checkpoints)
- MAE image encoder for tokenizer training: [facebook/vit-mae-large](https://huggingface.co/facebook/vit-mae-large/tree/main)

If the machine cannot access Hugging Face during training, download `facebook/vit-mae-large` first and point the tokenizer configs to the local directory:

```bash
VIT_MAE_MODEL_PATH=/path/to/vit-mae-large
sed -i "s#pretrained_model_name_or_path: \"facebook/vit-mae-large\"#pretrained_model_name_or_path: \"${VIT_MAE_MODEL_PATH}\"#g" \
  depth_tokenizer/configs/models/vq.yaml \
  flow_tokenizer/configs/models/vq.yaml \
  semantic_tokenizer/configs/models/vq.yaml
```

## Preprocessing

Convert LIBERO to the CALVIN-style episode format:

```bash
export LIBERO_CALVIN_STYLE_DIR=/path/to/libero_calvin_style/libero_90
python data_process/libero2calvin_style.py \
  --src_dir /path/to/LIBERO-datasets/libero_90 \
  --tgt_dir "$LIBERO_CALVIN_STYLE_DIR"
```

Extract flow supervision with CoTracker:

```bash
git clone https://github.com/facebookresearch/co-tracker.git
cp data_process/{dist_utils,flow_extractor,flow_extractor_oxe}.py co-tracker/
cd co-tracker
```

For recent PyTorch versions, replace `.view` with `.reshape` in CoTracker to avoid non-contiguous tensor errors:

```python
# co-tracker/cotracker/models/core/cotracker/cotracker3_offline.py
- coords_init = coords.view(B * T, N, 2)
+ coords_init = coords.reshape(B * T, N, 2)
```

```bash
export CALVIN_DATASET_DIR=/path/to/calvin/task_ABC_D
export CALVIN_FLOW_DIR=/path/to/calvin_flow/rgb_static
export COTRACKER_CKPT=/path/to/cotracker3_scaled_offline.pth
torchrun --nproc_per_node=8 flow_extractor.py \
  --data_root "$CALVIN_DATASET_DIR" \
  --save_path "$CALVIN_FLOW_DIR" \
  --checkpoint "$COTRACKER_CKPT"
```

Extract semantic supervision with SAM:

```bash
export CALVIN_DATASET_DIR=/path/to/calvin/task_ABC_D
export CALVIN_SEMANTIC_DIR=/path/to/calvin_semantic/rgb_static
export SAM_CKPT=/path/to/sam_vit_b_01ec64.pth
git clone https://github.com/facebookresearch/segment-anything.git
cp data_process/{dist_utils,semantic_extractor,semantic_extractor_oxe}.py segment-anything/
cd segment-anything
torchrun --nproc_per_node=8 semantic_extractor.py \
  --data_root "$CALVIN_DATASET_DIR" \
  --save_path "$CALVIN_SEMANTIC_DIR" \
  --checkpoint "$SAM_CKPT"
```

## Training

CALVIN:

```bash
export CALVIN_DATASET_DIR=/path/to/calvin/task_ABC_D
export DEPTH_ANYTHING_V2_CKPT=/path/to/depth_anything_v2_vitb.pth
bash scripts/train_depth_tokenizer_on_calvin.sh

export CALVIN_DATASET_DIR=/path/to/calvin/task_ABC_D
export CALVIN_FLOW_DIR=/path/to/calvin_flow/rgb_static
bash scripts/train_flow_tokenizer_on_calvin.sh

export CALVIN_DATASET_DIR=/path/to/calvin/
export CALVIN_SEMANTIC_DIR=/path/to/calvin_semantic/rgb_static
bash scripts/train_semantic_tokenizer_on_calvin.sh
```

LIBERO:

```bash
export LIBERO_CALVIN_STYLE_DIR=/path/to/libero_calvin_style/libero_90
export DEPTH_ANYTHING_V2_CKPT=/path/to/depth_anything_v2_vitb.pth
bash scripts/train_depth_tokenizer_on_libero.sh

export LIBERO_CALVIN_STYLE_DIR=/path/to/libero_calvin_style/libero_90
export LIBERO_FLOW_DIR=/path/to/libero_flow/rgb_static
bash scripts/train_flow_tokenizer_on_libero.sh

export LIBERO_CALVIN_STYLE_DIR=/path/to/libero_calvin_style/libero_90
export LIBERO_SEMANTIC_DIR=/path/to/libero_semantic/rgb_static
bash scripts/train_semantic_tokenizer_on_libero.sh
```
