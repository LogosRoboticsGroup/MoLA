import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ["PYOPENGL_PLATFORM"] = "osmesa"
os.environ['MUJOCO_GL'] = 'osmesa'

from pathlib import Path
import copy
import io
import distutils.dir_util
import numpy as np
import time
import torch
from torch.distributed import gather
from collections import deque
import functools
from scipy.spatial.transform import Rotation as R
from tqdm.auto import tqdm

from utils.data_utils import preprocess_image, preprocess_text_calvin
from utils.train_utils import get_cast_dtype

# libero
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
from PIL import Image
from pdb import set_trace 

import imageio

def quaternion_to_euler(q):
    rot = R.from_quat(q)
    euler = rot.as_euler('xyz', degrees=False)
    
    return euler

benchmark_map = {
    "libero_10": "LIBERO_10",
    "libero_spatial": "LIBERO_SPATIAL",
    "libero_object": "LIBERO_OBJECT",
    "libero_goal": "LIBERO_GOAL",
}

class ModelWrapper:
    def __init__(self, model, image_processor, cast_dtype, history_len=10, 
                use_ensembling=False, ensembling_temp=0.01, libero_eval_max_steps=600, action_pred_steps=3, gripper_width=False):
        super().__init__()
        self.model = model
        self.cast_type = cast_dtype
        self.image_process_fn = functools.partial(preprocess_image, image_processor=image_processor)
        self.action_hist_queue = []
        self.history_len = history_len
        self.libero_eval_max_steps = libero_eval_max_steps
        self.action_pred_steps = action_pred_steps
        self.device = "cuda"
        self.use_ensembling = use_ensembling
        self.ensembling_temp = ensembling_temp
        self.img_queue = deque(maxlen=history_len)
        self.gripper_queue = deque(maxlen=history_len)
        self.state_queue = deque(maxlen=history_len)
        self.mask_queue = deque(maxlen=history_len)
        self.text_queue = deque(maxlen=history_len)
        self.act_queue = deque(maxlen=history_len-1)
        self.cnt = 0
        self.gripper_width = gripper_width
        if self.use_ensembling:
            self.all_time_actions = torch.zeros(
                    [
                        self.libero_eval_max_steps,
                        self.libero_eval_max_steps + self.action_pred_steps,
                        7,
                    ]
                ).to(self.device)

    def reset(self):
        self.model.module.reset()
        self.img_queue = deque(maxlen=self.history_len)
        self.gripper_queue = deque(maxlen=self.history_len)
        self.state_queue = deque(maxlen=self.history_len)
        self.mask_queue = deque(maxlen=self.history_len)
        self.text_queue = deque(maxlen=self.history_len)
        self.act_queue = deque(maxlen=self.history_len-1)
        self.gripper_state = np.array([-1.0])
        if self.use_ensembling:
            self.all_time_actions = torch.zeros(
                    [
                        self.libero_eval_max_steps,
                        self.libero_eval_max_steps + self.action_pred_steps,
                        7,
                    ]
                ).to(self.device)

        self.cnt += 1

    def step(self, obs, goal, timestep):
        # preprocess image
        image = obs["agentview_image"]
        image = Image.fromarray(image)
        image_x = self.image_process_fn([image])
        # expand image dimension
        image_x = image_x.unsqueeze(1).to(dtype=self.cast_type)
        image_x = torch.flip(image_x, dims=[3])

        gripper = obs["robot0_eye_in_hand_image"]
        gripper = Image.fromarray(gripper)
        gripper = self.image_process_fn([gripper])
        # expand image dimension
        gripper = gripper.unsqueeze(1).to(dtype=self.cast_type) 

        with torch.no_grad():
            device = 'cuda'
            image_x = image_x.to(device)
            gripper = gripper.to(device)

            action = self.model.module.libero_step(
                image_primary=image_x,
                image_wrist=gripper,
                text=goal,
            )
            action[-1] = action[-1] > 0.0
            action[-1] = (action[-1] - 0.5) * 2
            action = action.detach().cpu().numpy()

        return action

def evaluate_libero_task(task, env, obs, args, model):
    steps = 0
    success = 0
    frames = []
    os.makedirs("videos", exist_ok=True)
    camera_key="agentview_image"
    model.reset()
    goal = task.language

    with torch.no_grad():
        while steps < args.libero_eval_max_steps:
            if camera_key in obs:
                img = obs[camera_key]

                if isinstance(img, torch.Tensor):
                    img = img.detach().cpu().numpy()
                if img.dtype != np.uint8:
                    img = (img * 255).clip(0, 255).astype(np.uint8)
                img = np.flipud(img)
                img = np.fliplr(img)
                frames.append(img)

            action = model.step(obs, goal, steps)

            steps += 1
            obs, reward, done, info = env.step(action)

            if done:
                success = 1
                break

    env.close()

    # video_path = os.path.join(
    #     "videos",
    #     f"{task.name}_success_{success}.mp4"
    # )
    # imageio.mimsave(video_path, frames, fps=20)

    return success

def evaluate_policy_ddp(args, model):
    pass 
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.finetune_type]()
    device_num = int(torch.distributed.get_world_size())
    device_id = torch.distributed.get_rank()
    results = []
    if "libero" in args.finetune_type:
        global num_eval_episodes 
        global task_num
        num_eval_episodes = 20
        task_num = 10
            
        NUM_SEQUENCES = num_eval_episodes * task_num 
        eval_sequences = list(range(NUM_SEQUENCES))
        assert NUM_SEQUENCES % device_num == 0
        interval_len = int(NUM_SEQUENCES // device_num)
        eval_sequences = eval_sequences[device_id*interval_len:min((device_id+1)*interval_len, NUM_SEQUENCES)]
        eval_sequences = tqdm(eval_sequences)
    else:
        raise NotImplementedError
    for eval_id in eval_sequences:
        task_id = eval_id // num_eval_episodes
        exp_id = eval_id % num_eval_episodes 
        task = task_suite.get_task(task_id)
        task_name = task.name
        task_description = task.language
        task_bddl_file = os.path.join(f"{args.libero_path}/libero/libero/bddl_files", task.problem_folder, task.bddl_file)
        env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": 256,
        "camera_widths": 256,
        "render_gpu_device_id":device_id
        }
        print("device_id :", device_id)
        env = OffScreenRenderEnv(**env_args)
        env.task_id = task_id
        env.task_name = task_name
        env.task_suite_name = args.finetune_type
        env.reset()
        env.seed(args.seed)

        # set initial state
        init_states_path = os.path.join(
            f"{args.libero_path}/libero/libero/init_files", task.problem_folder, task.init_states_file
        )
        init_states = torch.load(init_states_path)
        init_state = init_states[exp_id]
        obs = env.set_init_state(init_state)

        for _ in range(5):  # simulate the physics without any actions
            env.step(np.zeros(7))

        result = evaluate_libero_task(task, env, obs, args, model)
        results.append(result) 
        print("rank", torch.distributed.get_rank(), "results :", results)
    
    def merge_multi_list(res):
        tmp = []
        for l in res:
            tmp.extend(l)
        return tmp

    def extract_iter_from_tqdm(tqdm_iter):
        return [_ for _ in tqdm_iter]

    eval_sequences = extract_iter_from_tqdm(eval_sequences)
    res_tup = [(res, eval_seq) for res, eval_seq in zip(results, eval_sequences)]
    all_res_tup = [copy.deepcopy(res_tup) for _ in range(device_num)] if torch.distributed.get_rank() == 0 else None
    torch.distributed.gather_object(res_tup, all_res_tup, dst=0)

    if torch.distributed.get_rank() == 0:
        res_tup_list = merge_multi_list(all_res_tup)
        res_tup_list.sort(key=lambda x: x[1])
        print_and_save(res_tup_list, task_suite)

def print_and_save(result_list, task_suite):
    for j in range(task_num):
        this_result_list = result_list[j * num_eval_episodes: (j + 1) * num_eval_episodes]
        print("this_result_list :", this_result_list)
        this_result_list = np.array(this_result_list)
        avg_success = np.mean(this_result_list, axis=0)[0]
        task = task_suite.get_task(j)
        task_name = task.name
        print(f"Success rates for task {j} {task_name}:")
        print(f"{avg_success * 100:.1f}%")

def eval_one_epoch_libero_ddp(args, model, image_processor):
    cast_dtype = get_cast_dtype(args.precision)
    hist_len = args.sequence_length
    wrapped_model = ModelWrapper(
                        model, 
                        image_processor, 
                        cast_dtype, 
                        history_len=hist_len, 
                        use_ensembling=args.eval_libero_ensembling,
                        ensembling_temp=args.ensembling_temp,
                        libero_eval_max_steps=args.libero_eval_max_steps,
                        action_pred_steps = args.action_pred_steps,
                        gripper_width=args.gripper_width)
    evaluate_policy_ddp(args, wrapped_model)