import os
import re
import numpy as np
import h5py
from pathlib import Path
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm


def extract_task_language(file_name: str):
    text = os.path.basename(file_name)
    text = re.sub(r'\.hdf5$', '', text, flags=re.IGNORECASE)
    text = text.replace('_', ' ')
    text = re.sub(r'(KITCHEN|LIVING\s*ROOM|STUDY)\s+SCENE\s*\d+\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s*demo$', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text).strip().lower()
    return text


def process_demo(h5_path, demo_idx, language, start_global_idx, split_dir):
    ep_start_end_ids = []
    lang_text = []
    lang_emb = []

    with h5py.File(h5_path, "r") as f:
        demo_data = f["data"]
        obs_static = np.array(demo_data[f"demo_{demo_idx}"]["obs"]["agentview_rgb"])
        obs_static = np.flip(obs_static, axis=1)
        obs_gripper = np.array(demo_data[f"demo_{demo_idx}"]["obs"]["eye_in_hand_rgb"])
        robot_obs = np.array(demo_data[f"demo_{demo_idx}"]["obs"]["joint_states"])
        scene_obs = np.array(demo_data[f"demo_{demo_idx}"]["obs"]["joint_states"])
        actions = np.array(demo_data[f"demo_{demo_idx}"]["actions"])
        T = obs_static.shape[0]

        idx = start_global_idx
        for t in range(T):
            npz_data = {
                "rgb_static": obs_static[t],
                "rgb_gripper": obs_gripper[t],
                "robot_obs": robot_obs[t],
                "scene_obs": scene_obs[t],
                "rel_actions": actions[t],
            }
            save_path = split_dir / f"episode_{idx:07d}.npz"
            np.savez_compressed(save_path, **npz_data)
            idx += 1

        ep_start_end_ids.append([start_global_idx, idx - 1])
        lang_text.append(language)
        lang_emb.append(None)

    return ep_start_end_ids, lang_text, lang_emb, idx - start_global_idx


class LiberoToCalvinConverter:
    def __init__(self, src_dir: str, tgt_dir: str, train_ratio: float = 0.9, seed: int = 42, num_workers: int = 4):
        self.src_dir = Path(src_dir)
        self.tgt_dir = Path(tgt_dir)
        self.train_ratio = train_ratio
        self.seed = seed
        self.num_workers = num_workers
        self.tgt_dir.mkdir(parents=True, exist_ok=True)

    def run(self):
        all_demos = []
        languages_list = []
        for h5_path in sorted(self.src_dir.iterdir()):
            if not h5_path.name.endswith(".hdf5"):
                continue
            language = extract_task_language(h5_path.name)
            with h5py.File(h5_path, "r") as f:
                demo_data = f["data"]
                for demo_idx in range(len(demo_data)):
                    all_demos.append((h5_path, demo_idx, language))
                    languages_list.append(language)

        lang_txt_path = self.tgt_dir / "languages.txt"
        with open(lang_txt_path, "w", encoding="utf-8") as f:
            for lang in languages_list:
                f.write(lang + "\n")
        print(f"Saved language list to {lang_txt_path}")

        random.seed(self.seed)
        random.shuffle(all_demos)

        split_idx = int(len(all_demos) * self.train_ratio)
        train_demos = all_demos[:split_idx]
        val_demos = all_demos[split_idx:]

        print(f"Total demos: {len(all_demos)}")
        print(f"Training demos: {len(train_demos)}, Validation demos: {len(val_demos)}")

        self._process_split(train_demos, self.tgt_dir / "training")
        self._process_split(val_demos, self.tgt_dir / "validation")

        print("✅ Conversion to training/validation finished.")

    def _process_split(self, demos, split_dir: Path):
        split_dir.mkdir(parents=True, exist_ok=True)

        global_step_idx = 0
        ep_start_end_ids = []
        lang_text = []
        lang_emb = []

        demo_start_idx = []
        for h5_path, demo_idx, language in tqdm(demos, desc=f"Indexing {split_dir.name} demos"):
            with h5py.File(h5_path, "r") as f:
                T = f["data"][f"demo_{demo_idx}"]["obs"]["agentview_rgb"].shape[0]
            demo_start_idx.append(global_step_idx)
            global_step_idx += T

        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [
                executor.submit(process_demo, h5_path, demo_idx, language, demo_start_idx[i], split_dir)
                for i, (h5_path, demo_idx, language) in enumerate(demos)
            ]

            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing {split_dir.name} demos"):
                ep_ids, txt, emb, _ = future.result()
                ep_start_end_ids.extend(ep_ids)
                lang_text.extend(txt)
                lang_emb.extend(emb)

        np.save(split_dir / "ep_start_end_ids.npy", np.array(ep_start_end_ids, dtype=np.int64))

        auto_lang_ann = {
            "info": {"indx": np.array(ep_start_end_ids, dtype=np.int64)},
            "language": {"ann": np.array(lang_text, dtype=object),
                         "emb": np.array(lang_emb, dtype=object)}
        }
        np.save(split_dir / "auto_lang_ann.npy", auto_lang_ann)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir", default=os.getenv("LIBERO_SOURCE_DIR", "/path/to/LIBERO-datasets/libero_90"))
    parser.add_argument("--tgt_dir", default=os.getenv("LIBERO_CALVIN_STYLE_DIR", "outputs/libero_calvin_style/libero_90"))
    parser.add_argument("--train_ratio", type=float, default=0.95)
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()

    converter = LiberoToCalvinConverter(args.src_dir, args.tgt_dir, train_ratio=args.train_ratio, num_workers=args.num_workers)
    converter.run()
