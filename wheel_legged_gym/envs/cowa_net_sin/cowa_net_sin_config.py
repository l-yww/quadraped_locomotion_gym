# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2024 Beijing RobotEra TECHNOLOGY CO.,LTD. All rights reserved.


from wheel_legged_gym.envs.cowa.cowa_config import CowaCfg, CowaCfgPPO

class CowaCfg_Net_Sin(CowaCfg):
    """
    用于对齐sin曲线跟随,joint相关属性
    """
    class mode:
        use_net = True

    class env(CowaCfg.env):
        frame_stack = 5
        actor_input_stack = 5
        c_frame_stack = 1
        num_single_obs = 4+6+6+1
        single_num_privileged_obs = 4+6+4+2+6+1
        num_observations = int(frame_stack * num_single_obs)
        num_privileged_obs = int(c_frame_stack * single_num_privileged_obs)

    class asset(CowaCfg.asset):
        file = "{WHEEL_LEGGED_GYM_ROOT_DIR}/resources/robots/cowa_wheel_legged_v2/cowa_wheel_legged_v2/urdf/wheel_v2_6dof_wo_arm.urdf"
        fix_base_link = True
        
    class control(CowaCfg.control):
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 5  # 50Hz 100 Hz
        pos_action_scale = 0.25 
        vel_action_scale = 2.0

        cycle_time = 3.0

    class sim(CowaCfg.sim):
        dt = 0.002  # 200Hz 500 Hz

    class rewards(CowaCfg.rewards):
        class scales:
            #################   tracking rewards  ##################
            ref_hip_pos = 1
            ref_knee_pos = 1
            ref_wheel_vel = 1
            ############## normalized rewards #####################
            dof_vel = -5e-5 #-1e-7
            dof_acc = -5e-7 #-1e-7
            torques = -1e-5 #-1e-7
            torque_limits = -0.05 #-1e-7
            action_rate = -0.03 #-0.001
            action_smoothness = -0.03 #-0.005
            dof_pos_limits = -0.1
            dof_vel_limits = -0.1

class CowaCfgPPO_Net_Sin(CowaCfgPPO):
    seed = 10
    runner_class_name = 'OnPolicyRunner'   # DWLOnPolicyRunner

    class policy(CowaCfgPPO.policy):
        init_noise_std = 1.0
        actor_hidden_dims = [128, 64, 32] # 256, 128, 64
        critic_hidden_dims = [128, 64, 32]

    class algorithm(CowaCfgPPO.algorithm):
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4  # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 5.0e-4  # 5.e-4
        schedule = "adaptive"  # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.0

    class runner(CowaCfgPPO.runner):
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 48  # per iteration
        max_iterations = 1000001  # number of policy updates        #  xxw

        # logging
        save_interval = 50  # Please check for potential savings every `save_interval` iterations.
        experiment_name = ''
        run_name = ''
        # Load and resume
        resume = False
        load_run = -1  # -1 = last run
        checkpoint = -1  # -1 = last saved model
        resume_path = ''  # updated from load_run and chkpt
