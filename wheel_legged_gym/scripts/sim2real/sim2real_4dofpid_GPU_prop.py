
import math
import numpy as np
# import mujoco, mujoco_viewer
# from tqdm import tqdm
from collections import deque
# from humanoid import LEGGED_GYM_ROOT_DIR
from env.cowa_dual_config import CowaCfg_DUAL
#from env.cowa_param_config import CowaCfg_FIX
import torch
import signal
#from env_rl.communication_with_robot import ObservationNode, RLPublish
from env.communication_4dofpid import ObservationNode, RLPublish, PIDController
import logging
from env.log import Log
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


def CorrectYaw(current_yaw):
    global previous_
    if previous_ is None:
        previous_ = current_yaw
        return current_yaw

    diff = current_yaw - previous_
    while diff > math.pi:
        diff -= 2* math.pi
    while diff < -math.pi:
        diff += 2 * math.pi
    
    corrent_yaw = previous_ + diff
    previous_ = corrent_yaw

    return corrent_yaw

def QuaternionMultiply(q1, q2):
    """
    Multiply two quaternions.

    :param q1: First quaternion [w, x, y, z].
    :param q2: Second quaternion [w, x, y, z].
    :return: Resulting quaternion [w, x, y, z].
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return w, x, y, z
  
def Quaternion2Euler(w,x,y,z):
    """
    Convert a quaternion to Euler angles (roll, pitch, yaw).

    :param w: Real part of the quaternion.
    :param x: i-component of the quaternion.
    :param y: j-component of the quaternion.
    :param z: k-component of the quaternion.
    :return: Tuple (roll, pitch, yaw) in radians.
    """
    # Roll (x-axis rotation)

    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll_x = math.atan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch_y = math.copysign(math.pi / 2, sinp)  # Use 90 degrees if out of range
    else:
        pitch_y = math.asin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw_z = math.atan2(siny_cosp, cosy_cosp)
    yaw_z = CorrectYaw(yaw_z)

    return np.array([roll_x,pitch_y,yaw_z])
    #return np.array([0,0,0])

def write_imu2file(timestamp,r,p,y,gap):
    with open('imu_data.txt','a') as f:
        f.write(str(timestamp) + ' ' + str(r) + ' ' + str(p) + ' ' + str(y) + ' ' + str(gap) + '\n')

def run_policy(policy, cfg, observation_data):
    #仿真步长是1000Hz，但是上游数据是100Hz
    publish_data.PublishAction([0]*6)
    #action = np.zeros(cfg.env.num_actions, dtype=np.double)
    action = np.zeros(4, dtype=np.double)
    print('init action: ',action)
    #obs_size = cfg.env.num_single_obs
    obs_size = 16
    if params['odom_in_obs']: obs_size = obs_size + 2
    if params['map_obs']: obs_size = obs_size + params['map_size']
    hist_obs = deque()
    for _ in range (params['frame_stack']):
        hist_obs.append(np.zeros([1, obs_size], dtype=np.double))
    
    global action_cmd
    action_cmd = [0] * 13
    #这里仿真是1000Hz的，但是上游数据是100Hz的
    print_count = 0
    last_time = time.time()
    imu_yaw_time = time.time()
    info_ok_flag = False
    init_imu_flag = False
    init_odom_flag = False
    imu_init_yaw = 0.
    last_cmd_vx = None
    odom_pos_data = [0.,0.]
    debug_odom_raw = 0.
    heading = 0.0
    #vx_pid = PIDController(0.5, 0.0018, 0.13) # 1.0, 0.001, 0.13
    vx_pid = PIDController(0.5, 0.0005, 0.13) # 1.0, 0.001, 0.13
    dyaw_pid = PIDController(0.2, 0.0, 0.01)
    # TODO 根据imu yaw角漂移速度，做补偿
    while(1):
        proc_start_time = time.time()
        imu_data = observation_data.GetObservation()['imu']
        leg_state = observation_data.GetObservation()['leg_info']
        if (imu_data == None or leg_state == None ):
            if not info_ok_flag:
                for i in range(10):
                    print("imu or leg info is None !!!!!!")
                    info_ok_flag = True
            continue
            
        # 控制指令处理
        motion_cmd = observation_data.GetObservation()['motion']
        cmd.vx = np.clip(motion_cmd['speed']+params['vx_offset'],params["vx"][0],params['vx'][1])
        cmd.dyaw = np.clip(motion_cmd['steer']+params['vy_offset'],params["vy"][0],params['vy'][1])
        cmd.height = np.clip(motion_cmd['height'],params["height"][0],params["height"][1])

        # imu 信息处理
        eu_ang = Quaternion2Euler(imu_data.transform[3],imu_data.transform[4],imu_data.transform[5],imu_data.transform[6])
        if not init_imu_flag: 
            imu_init_yaw = eu_ang[2]
            #heading =  imu_init_yaw
            init_imu_flag = True
        eu_ang[2] = eu_ang[2] - imu_init_yaw
        eu_ang[2] += (time.time()-imu_yaw_time) * 0.0005
        #eu_ang[2] -= (time.time()-imu_yaw_time) * 0.05
        # print("r:{} p:{} y:{}".format(eu_ang[0],eu_ang[1],eu_ang[2]))
        angular_velocity = np.array([imu_data.angular.x, imu_data.angular.y, imu_data.angular.z]).astype(np.double)  # 单位: rad/s  

        # 腿部信息处理
        q = np.array(leg_state.joint_state.pos).astype(np.double)
        dq = np.array(leg_state.joint_state.speed).astype(np.double)
        #q = np.delete(q,[3,7]) # 删除驻足信息
        #dq = np.delete(dq,[3,7]) # 删除驻足信息

        # 机械臂信息处理
        if params['arm_state']:
            arm_state = observation_data.GetObservation()['arm_info']
            q_arm = np.array(arm_state.joint_state.pos).astype(np.double)
            dq_arm = np.array(arm_state.joint_state.speed).astype(np.double)
            q_arm[4:] = 0.0
            dq_arm[4:] = 0.0
            # q_arm[2] = q_arm[2] - (-2.25)
            # q_arm[3] = q_arm[3] - 3.14
            # 平衡机械臂
            if params['arm_compensate']:
                if q_arm[2] < -1.57: cmd.vx += -0.108 #-0.08 # -0.105
                elif q_arm[2] >= -1.57 and q_arm[2] < -0.8: cmd.vx += -0.11 # -0.11 -0.13
                elif q_arm[2] >= -0.8 and q_arm[2] < 0: cmd.vx += -0.13 # -0.13 -0.15
                elif q_arm[2] > 0.0: cmd.vx += -0.13 # -0.16 水泥地0.14 -0.16

        # 里程计处理
        pos_state = observation_data.GetObservation()['pos']
        current_pos = np.array([pos_state['x'],pos_state['y']]).astype(np.float32)
        if not init_odom_flag:
            odom_pos_data[0] = pos_state['x']
            odom_pos_data[1] = pos_state['y']
            init_odom_flag = True
        # if last_cmd_vx is None: last_cmd_vx = cmd.vx
        # elif abs(last_cmd_vx) > 1e-6 and abs(cmd.vx) < 1e-6:
        #     # 上一帧速度指令不为0，当前帧为0时候，即松开控制按键，记录当前位置
        #     odom_pos_data[0] = pos_state['x']
        #     odom_pos_data[1] = pos_state['y']
        # elif abs(last_cmd_vx) > 1e-6 and abs(cmd.vx) > 1e-6:
        #     odom_pos_data = [0.,0.]
        if abs(motion_cmd['speed']) > 0.09 or abs(motion_cmd['steer']) > 0.05:
            # 只要有速度下发，就一直更新为止记录，速度停发时候，记录最后一帧位置
            odom_pos_data[0] = pos_state['x']
            odom_pos_data[1] = pos_state['y']
            heading = eu_ang[2] # 拿imu的yaw角  
        # pid计算Vx
        last_cmd_vx = cmd.vx
        obs_odom_pos = np.array(odom_pos_data).astype(np.double)
        odom_gap = current_pos - obs_odom_pos
        debug_odom_raw = odom_gap[0]
        if abs(odom_gap[0]) < 0.15: odom_gap[0] = 0.0
        cmd.vx += vx_pid.Update(odom_gap[0])
        # pid计算Vy(dyaw)
        heading_error = eu_ang[2] - heading
        if abs(heading_error) < 0.15: heading_error = 0.0
        cmd.dyaw += dyaw_pid.Update(heading_error)
        # cmd.dyaw -= dyaw_pid.Update(odom_gap[1]) # 应该跟heading

        # 高程图处理
        map_raw = observation_data.GetObservation()['map']
        if params['map_zero']: map_raw = np.array([0]*params['map_size']).astype(np.float32)
        map_data = np.array(map_raw).astype(np.float32)
      
        #需要根据输入配置更改每帧的观察值的维度
        commands =  np.zeros([1, 3], dtype=np.float64)
        commands[0, 0] = cmd.vx * cfg.normalization.obs_scales.lin_vel
        commands[0, 1] = cmd.dyaw * cfg.normalization.obs_scales.ang_vel
        commands[0, 2] = cmd.height * cfg.normalization.obs_scales.height_measurements
        #将数据放入obs
        obs =  np.zeros([1, obs_size], dtype=np.float64)
        obs[0, 0] = q[0] * cfg.normalization.obs_scales.dof_pos
        obs[0, 1] = q[2] * cfg.normalization.obs_scales.dof_pos
        obs[0, 2:6] = dq * cfg.normalization.obs_scales.dof_vel
        obs[0, 6:10] = action
        obs[0, 10:13] = angular_velocity * cfg.normalization.obs_scales.ang_vel
        obs[0, 13:16] = eu_ang * cfg.normalization.obs_scales.quat
        
        #obs[0, 0] = cmd.vx * cfg.normalization.obs_scales.lin_vel
        #obs[0, 1] = cmd.dyaw * cfg.normalization.obs_scales.ang_vel
        #obs[0, 2] = cmd.height * cfg.normalization.obs_scales.height_measurements
        #obs[0, 3] = q[0] * cfg.normalization.obs_scales.dof_pos
        #obs[0, 4] = q[1] * cfg.normalization.obs_scales.dof_pos
        #obs[0, 5] = q[3] * cfg.normalization.obs_scales.dof_pos
        #obs[0, 6] = q[4] * cfg.normalization.obs_scales.dof_pos
        #obs[0, 7:13] = dq * cfg.normalization.obs_scales.dof_vel
        #obs[0, 13:19] = action
        #obs[0, 19:22] = angular_velocity * cfg.normalization.obs_scales.ang_vel
        #obs[0, 22:25] = eu_ang * cfg.normalization.obs_scales.quat
        #if params['odom_in_obs']: obs[0, 25:27] = current_pos - obs_odom_pos
        #if params['map_obs']: obs[0, 25:25+params['map_size']] = np.array(map_data * cfg.normalization.obs_scales.height_measurements, dtype=np.float32) # 146
        obs = np.clip(obs, -cfg.normalization.clip_observations, cfg.normalization.clip_observations)

        hist_obs.append(obs)
        hist_obs.popleft()

        policy_input = np.zeros([1, int(obs_size * params['frame_stack'])], dtype=np.float32)
        
        for i in range(params['frame_stack']):
            policy_input[0, i * obs_size : (i + 1) * obs_size] = hist_obs[i][0, :]
        #inference delay
        # input = np.array([map , policy_input])
        #input_map_tensor = torch.tensor(map_data * cfg.normalization.obs_scales.height_measurements).float()
        #input_policy_tensor = torch.tensor(policy_input).float()
        #input_obs_tensor = torch.tensor(obs).float()
        scaled_map = np.array(map_data * cfg.normalization.obs_scales.height_measurements, dtype=np.float32)
        # 将所有需要输入的数据拼接成一个大的观测向量
        
        all_obs = np.concatenate([
            scaled_map.flatten(),
            policy_input.flatten(),
        ], axis=0)

        # 添加 batch 维度 (1, N)，然后转为 torch tensor
        if params['map_input_policy']:
            all_obs_tensor = torch.tensor(all_obs[None, :].tolist(), dtype=torch.float32)
        else:
            # all_obs_tensor = torch.tensor(policy_input.tolist(), dtype=torch.float32)
        #action[:] = policy(all_obs_tensor)[0].detach().numpy()[:6]
        #indices = [3,4,5,9,10,11]
        #print("obs size:",sys.getsizeof(all_obs_tensor))
        #with torch.no_grad():
            all_obs_tensor = torch.tensor(policy_input.copy(), dtype=torch.float32, device=device)
        cmd_tensor = torch.tensor(commands.copy(), dtype=torch.float32, device=device)
        with torch.no_grad():
            action[:] = policy(all_obs_tensor, cmd_tensor)[0].cpu().numpy()[:4]
        # action[:] = policy(all_obs_tensor)[0].detach().numpy()[:4]
        action = np.clip(action, -10, 10)
        indices = [3,5,9,11]
        for i,idx in enumerate(indices):
            action_cmd[idx] = action[i]
        action_scale = action_cmd

        action_scale[3] = params['pos_scale'] * action_scale[3] # 1.5
        #action_scale[4] = params['pos_scale'] * action_scale[4]
        action_scale[5] = params['speed_scale'] * action_scale[5]
        action_scale[9] = params['pos_scale'] * action_scale[9]
        #action_scale[10] = params['pos_scale'] * action_scale[10]
        action_scale[11] = params['speed_scale'] * action_scale[11]

        debug_dict = {
            "heading" : heading,
            "yaw" : eu_ang[2],
            "heading_error" : heading_error,
            "dyaw_cmd" : motion_cmd['steer'],
            "vx_cmd" : motion_cmd['speed'],
            "vx_output" : cmd.vx,
            "dyaw_output" : cmd.dyaw,
            "odom_gap": odom_gap[0],
            "odom_gap_raw": debug_odom_raw,
        }
        observation_data.PublishObservation(obs[0], debug_dict)
        observation_data.PublishBaseInfo()
        #observation_data.PublushObservation(obs[0])
        publish_data.PublishAction(action_scale)
        #TODO 增加publish pos cmd

        # debug info
        if (print_count < 50):
            print('-----------action----------')
            print('action: {} count: {}'.format(action,print_count))
            # print("obs:",obs,print_count)
            #print('------------obs------------')
            #print("cmd: {}".format(obs[0,0:3]))
            #print("q: {}".format(obs[0,3:7]))
            #print("dq: {}".format(obs[0,7:13]))
            #print("action: {}".format(obs[0,13:19]))
            #print("angular_v: {}".format(obs[0,19:22]))
            #print("eu_ang: {}".format(obs[0,22:25]))
            ## print('action_cmd:',action_cmd,print_count)
            # print("map: {}".format(map_data))
            print_count += 1
        if (time.time() - last_time > 0.5):
            print("date: {} vx: {:.3f} vy: {:.3f} vz: {:.3f} cmdx: {:.3f} cmd_yaw: {:.3f} yawinit: {:.4f} yaw: {:.4f} heading_error: {:.4f}"\
                  .format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"),cmd.vx,cmd.dyaw,cmd.height,motion_cmd['speed'],motion_cmd['steer'],imu_init_yaw,eu_ang[2],heading_error))
            #print("vx: {:.3f} vy: {:.3f} vz: {:.3f} yawinit: {:.4f} yaw: {:.4f}".format(cmd.vx,cmd.dyaw,cmd.height,imu_init_yaw,eu_ang[2]))
            #print(':',q_arm[1:])
            print('odom:',odom_gap)
            #print("pos x: {} pos y: {}".format(obs_odom_pos[0],obs_odom_pos[1]))
            last_time = time.time()
        elapsed_time = time.time() - proc_start_time
        # gc.collect()
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
    observation_data = ObservationNode(params["map_size"])
    time.sleep(5)
    publish_data = RLPublish()
    try:
        #python scripts/sim2sim.py --load_model /path/to/export/model.pt
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        policy = torch.jit.load(args.load_model, map_location=device)
        policy.eval()
    except Exception as e:
        print(f"Failed to load model: {e}")
        sys.exit(1)
    
    run_policy(policy, CowaCfg_DUAL(), observation_data)
