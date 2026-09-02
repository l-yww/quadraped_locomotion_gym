
import math
import numpy as np
# import mujoco, mujoco_viewer
# from tqdm import tqdm
from collections import deque

import torch
import signal

import time
import threading
import os
import sys
from collections import deque
import numpy as np
import gc
from datetime import datetime

lock = threading.Lock()
time_step = 0.01 # 0.01 = 100hz

action_cmd_queue = deque(maxlen=15)
action_cmd_queue.append([0] * 12)
previous_ = None

params = {
    # 控制指令
    "vx": [-0.4, 0.4],
    "vy": [-0.4, 0.4],
    "height": [0.25, 0.38],
    "vx_offset": -0.0, # vx速度偏置
    "vy_offset": 0.0, # vy速度偏置

    # 手臂控制
    "arm_state": True, # 观测是否拿手臂信息
    "arm_compensate": False, #arm_state True时生效，是否根据机械臂位置控制Vx
    
    # 高程图相关
    "map_obs": False, # True 时高程图加入obs 以下参数
    "map_input_policy": False, # 为true时，作为模型输入，先map，后obs
    "map_zero": False, # True 时，高程图全0 ,不启动高程图时，也默认全0
    
    "map_size": 441, # 高程图的维度
    "odom_in_obs": False, 

    "frame_stack": 66, # 历史帧
    "pos_scale": 0.25, 
    "speed_scale": 2,
}

class cmd:
    vx = 0.0
    dyaw = 0.0
    height = 0.33

def run_policy(policy):
    #仿真步长是1000Hz，但是上游数据是100Hz
    action = np.zeros(4, dtype=np.double)
    obs_size = 16

    hist_obs = deque()
    for _ in range (params['frame_stack']):
        hist_obs.append(np.zeros([1, obs_size], dtype=np.double))
    
    while(1):
        proc_start_time = time.time()
        #需要根据输入配置更改每帧的观察值的维度
        commands =  np.zeros([1, 3], dtype=np.float64)
        commands[0, 0] = cmd.vx * 10.
        commands[0, 1] = cmd.dyaw * 2.
        commands[0, 2] = cmd.height * 5.
        #将数据放入obs
        obs =  np.zeros([1, obs_size], dtype=np.float64)
        obs[0, 0] = 0
        obs[0, 1] = 0
        obs[0, 2:6] = 0
        obs[0, 6:10] = action
        obs[0, 10:13] = 0
        obs[0, 13:16] = 0

        hist_obs.append(obs)
        hist_obs.popleft()

        policy_input = np.zeros([1, int(obs_size * params['frame_stack'])], dtype=np.float32)
        
        for i in range(params['frame_stack']):
            policy_input[0, i * obs_size : (i + 1) * obs_size] = hist_obs[i][0, :]
        # device = "cuda:0"
        all_obs_tensor = torch.tensor(policy_input.copy(), dtype=torch.float32)
        cmd_tensor = torch.tensor(commands.copy(), dtype=torch.float32)
        with torch.no_grad():
            action[:] = policy(all_obs_tensor, cmd_tensor)[0].cpu().numpy()[:4]
        print('action', action)
        elapsed_time = time.time() - proc_start_time
        time.sleep(max(0, time_step - elapsed_time))    # 控制每次循环为time_step = 0.01s

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Deployment script.')
    parser.add_argument('--load_model', type=str, required=True,
                        help='Run to load from.')
    args = parser.parse_args()
    if not os.path.isfile(args.load_model):
        print(f"Error: Model file {args.load_model} does not exist.")
        sys.exit(1)
    try:
        #python scripts/sim2sim.py --load_model /path/to/export/model.pt
        policy = torch.jit.load(args.load_model)
    except Exception as e:
        print(f"Failed to load model: {e}")
        sys.exit(1)
    
    run_policy(policy)

