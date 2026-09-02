from ..quadruped.legged_robot import *
from wheel_legged_gym.utils.motion_loader import AMPLoader

class LeggedRobotAMP(LeggedRobot):
    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        self.global_counter += 1

        actions = torch.clamp(actions, min = -self.cfg.normalization.clip_actions, max = self.cfg.normalization.clip_actions)
        self.actions = actions
        if self.cfg.control.action_smoothness:
            ratio = self.cfg.control.ratio
            self.actions = ratio * self.actions + (1 - ratio) * self.last_actions

        self.render()
        
        self.pre_physics_step()
           
        for _ in range(self.cfg.control.decimation):
            self.envs_steps_buf += 1
            self.torques = self._compute_torques(actions).view(self.torques.shape)
            # ---------------- 随机 电机编码器 延迟 --------------- #
            if self.cfg.domain_rand.add_dof_lag:
                q = self.dof_pos
                self.dof_lag_buffer[:,:,1:] = self.dof_lag_buffer[:,:,:self.cfg.domain_rand.dof_lag_timesteps_range[1]].clone()
                self.dof_lag_buffer[:,:,0] = q.clone()
                dq = self.dof_vel
                self.dof_vel_lag_buffer[:,:,1:] = self.dof_vel_lag_buffer[:,:,:self.cfg.domain_rand.dof_lag_timesteps_range[1]].clone()
                self.dof_vel_lag_buffer[:,:,0] = dq.clone()
            # ---------------- 随机 IMU 延迟 --------------- #
            if self.cfg.domain_rand.add_imu_lag:
                self.gym.refresh_actor_root_state_tensor(self.sim)
                self.base_quat[:] = self.root_states[:, 3:7]
                self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
                self.base_euler_rpy = get_euler_rpy_tensor(self.base_quat)
                self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
                self.imu_lag_buffer[:,:,1:] = self.imu_lag_buffer[:,:,:self.cfg.domain_rand.imu_lag_timesteps_range[1]].clone()
                if self.cfg.env.projected_gravity == True:
                    self.imu_lag_buffer[:,:,0] = torch.cat((self.base_ang_vel, self.projected_gravity ), 1).clone()
                else:
                    self.imu_lag_buffer[:,:,0] = torch.cat((self.base_ang_vel, self.base_euler_rpy ), 1).clone()   
            # ---------- 下发 Torque, 仿真器步进一步 ----------- #
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.compute_dof_vel()

        
        if 'AMPRunner_HIM' in self.train_cfg.runner_class_name:
            termination_ids, termination_priveleged_obs = self.post_physics_step()
        elif 'HIM' in self.train_cfg.runner_class_name:
            termination_ids, termination_priveleged_obs = self.post_physics_step()
        elif 'VAE' in self.train_cfg.runner_class_name:
            termination_ids, termination_priveleged_obs = self.post_physics_step()
        else:
            self.post_physics_step()
            
        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)

        if self.cfg.depth.use_camera and self.global_counter % self.cfg.depth.update_interval == 0:
            self.extras["depth"] = self.depth_buffer[:, -2]  # have already selected last one
            # interpolation = torch.rand((self.cfg.depth.camera_num_envs, 1, 1), device=self.device)
            # self.extras["depth"] = self.depth_buffer[:, -1] * interpolation + self.depth_buffer[:, -2] * (1-interpolation)
        else:
            self.extras["depth"] = None
        
        if 'AMPRunner_HIM' in self.train_cfg.runner_class_name:
            return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras, self.reset_env_ids, termination_priveleged_obs, self.terminal_amp_states
        elif 'HIM' in self.train_cfg.runner_class_name:    
            return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras, termination_ids, termination_priveleged_obs
        elif 'P3O' in self.train_cfg.runner_class_name:
            return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.cost_buf, self.reset_buf, self.extras
        else:
            return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras, self.reset_env_ids, self.terminal_amp_states
    
    def reset(self):
        """ Reset all robots"""
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        if 'AMPRunner_HIM' in self.train_cfg.runner_class_name:
            obs, privileged_obs, _, _, _, _, _, _ = self.step(torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False))
        else:
            obs, privileged_obs, _, _, _, _, _ = self.step(torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False))
        return obs, privileged_obs
    
    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)
        else:
            env_ids = env_ids.to(device=self.device, dtype=torch.long)

        self.reset_env_ids = env_ids

        if len(env_ids) == 0:
            amp_obs_dim = (
                self.base_lin_vel.shape[1]
                + self.base_ang_vel.shape[1]
                + self.dof_pos.shape[1]
                + self.dof_vel.shape[1]
                + len(self.key_body_indices) * 3
            )
            self.terminal_amp_states = torch.empty(
                (0, amp_obs_dim),
                device=self.device,
                dtype=self.dof_pos.dtype,
            )
            return

        self.terminal_amp_states = self.get_amp_observations()[env_ids].clone()
        
        self._resample_commands(env_ids)
        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length==0):
            self.update_command_curriculum(env_ids)
        
        self._randomize_dof_props(env_ids)
        self.randomize_lag_props(env_ids)
        self._refresh_actor_dof_props(env_ids)  # refresh joint damping/friction/aramture
        # reset robot states
        used_reference_motion = False
        _ = np.random.random()
        if self.cfg.init_state.reference_state_initialization \
            and _ < self.cfg.init_state.reference_state_initialization_prob:
            frames = self.amp_loader.get_full_frame_batch(len(env_ids))
            self._reset_dofs_from_reference_motion(env_ids, frames)
            self._reset_root_states_from_reference_motion(env_ids, frames)
            used_reference_motion = True
        else:
            self._reset_dofs(env_ids)
            self._reset_root_states(env_ids)
            self.gym.refresh_rigid_body_state_tensor(self.sim)


        if self.cfg.domain_rand.randomize_rigids_after_start:
            self._randomize_rigid_body_props(env_ids)
            self.refresh_actor_rigid_shape_props(env_ids)
            self.refresh_actor_rigid_body_props(env_ids)
        
        # reset buffers
        # self.dof_vel[env_ids] = 0.
        if used_reference_motion:
            self.last_dof_pos[env_ids] = self.dof_pos[env_ids]
        else:
            self.last_dof_pos[env_ids] = 0.
        self.last_last_actions[env_ids] = 0.
        self.last_actions[env_ids] = 0.
        self.actions[env_ids] = 0.
        self.last_rigid_state[env_ids] = 0.
        self.last_root_vel[env_ids] = self.root_states[env_ids, 7:13]
        if used_reference_motion:
            self.last_dof_vel_50hz[env_ids] = self.dof_vel[env_ids]
            self.last_dof_vel_200hz[env_ids] = self.dof_vel[env_ids]
        else:
            self.last_dof_vel_50hz[env_ids] = 0.
            self.last_dof_vel_200hz[env_ids] = 0.
        self.dof_acc_50hz[env_ids] = 0.
        self.dof_acc_200hz[env_ids] = 0.

        self.feet_air_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        self.fail_buf[env_ids] = 0
        self.envs_steps_buf[env_ids] = 0

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        if 'P3O' in self.train_cfg.runner_class_name:
            for key in self.cost_episode_sums.keys():
                self.extras["episode"]['cost_'+ key] = torch.mean(self.cost_episode_sums[key][env_ids]) / self.max_episode_length_s
                self.cost_episode_sums[key][env_ids] = 0. 
        # log additional curriculum info
        if self.cfg.terrain.mesh_type == "trimesh":
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf
        # fix reset gravity bug
        self.base_pos_init[env_ids] = self.root_states[env_ids, 0:3]
        self.base_pos[env_ids] = self.root_states[env_ids, 0:3]
        self.base_quat[env_ids] = self.root_states[env_ids, 3:7]
        self.base_euler_rpy = get_euler_rpy_tensor(self.base_quat)
        self.projected_gravity[env_ids] = quat_rotate_inverse(self.base_quat[env_ids], self.gravity_vec[env_ids])
        self.feet_quat = self.rigid_state[:, self.feet_indices, 3:7]
        self.feet_euler_rpy = get_euler_rpy_tensor(self.feet_quat)
    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment, 
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(WHEEL_LEGGED_GYM_ROOT_DIR=WHEEL_LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_joints = self.gym.get_asset_joint_count(robot_asset)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)
        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.body_names_to_idx = self.gym.get_asset_rigid_body_dict(robot_asset)
        print('body_names_to_idx: {}'.format(sorted(list(self.body_names_to_idx.items()), key=lambda x: x[1])))

        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.dof_names_to_idx = self.gym.get_asset_dof_dict(robot_asset)
        print('dof_names_to_idx: {}'.format(sorted(list(self.dof_names_to_idx.items()), key=lambda x: x[1])))

        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        print('self.cfg.asset.',self.cfg.asset)
        self.foot_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        self.foot_nums = len(self.foot_names)
        self.key_body_names = [s for s in body_names if self.cfg.asset.key_body_names in s]
        self.key_body_nums = len(self.key_body_names)
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []
        self.cam_handles = []
        self.base_com = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.init_base_com = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self.device, requires_grad=False)
        self.body_mass = torch.zeros(self.num_envs, 1, dtype=torch.float32, device=self.device, requires_grad=False)
        self.init_body_mass = torch.zeros(self.num_envs, 1, dtype=torch.float32, device=self.device, requires_grad=False)
        

        self._init_custom_buffers__()
        self._randomize_dof_props(torch.arange(self.num_envs, device=self.device))
        self._randomize_rigid_body_props(torch.arange(self.num_envs, device=self.device))

        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(-1., 1., (2,1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)
            
            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, self.cfg.asset.name, i, self.cfg.asset.self_collisions, 0)
            dof_props = self._process_dof_props(dof_props_asset, i)
            
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

            if self.cfg.depth.use_camera:
                self.attach_camera(i, env_handle, actor_handle)

        self._refresh_actor_dof_props(torch.arange(self.num_envs, device=self.device))
        
        self.feet_indices = torch.zeros(len(self.foot_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(self.foot_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], self.foot_names[i])
        self.key_body_indices = torch.zeros(len(self.key_body_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(self.key_body_names)):
            self.key_body_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], self.key_body_names[i])

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])
    def _init_buffers(self):
        super()._init_buffers()
        self.key_body_pos = self._get_key_body_pos()
        self.reset_env_ids = None
        self.terminal_amp_states = None
        if self.cfg.init_state.reference_state_initialization:
            self.amp_loader = AMPLoader(motion_files=self.cfg.env.amp_motion_files, 
                                        device=self.device, 
                                        time_between_frames=self.dt,
                                        num_dof=self.num_actions,
                                        num_key_bodies=len(self.key_body_indices))
    def _get_key_body_pos(self):
        return self.rigid_state.view(self.num_envs, self.num_bodies, 13)[:, self.key_body_indices, 0:3]

    def get_amp_observations(self):  #判别器观测数据
        key_body_pos_relative_to_base = self._get_key_body_pos() - self.base_pos.unsqueeze(1) #脚相对躯体的坐标系位置
        # Use base_lin_vel_w, base_ang_vel_w, dof_pos, dof_vel, key_body_pos_relative_to_base in the observations
        return torch.cat((
            self.base_lin_vel,              # 3
            self.base_ang_vel,             # 3
            self.dof_pos,                   # num_dofs
            self.dof_vel,                   # num_dofs
            key_body_pos_relative_to_base.flatten(start_dim=1), # num_key_bodies * 3
        ), dim=-1)  #42
        
    def _reset_amp_dof_states(self, env_ids, dof_pos, dof_vel):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        self.dof_pos[env_ids] = dof_pos
        self.dof_vel[env_ids] = dof_vel

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
    
    def _reset_amp_root_states(self, env_ids,
                           base_pos, 
                          base_quat, 
                          base_lin_vel_w, 
                          base_ang_vel_w):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        self.root_states[env_ids, 0:3] = base_pos
        self.root_states[env_ids, 3:7] = base_quat
        self.root_states[env_ids, 7:10] = base_lin_vel_w
        self.root_states[env_ids, 10:13] = base_ang_vel_w
        self.base_pos[env_ids] = base_pos
        self.base_quat[env_ids] = base_quat
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        self.base_euler_rpy[env_ids] = get_euler_rpy_tensor(self.base_quat[env_ids])
        self.projected_gravity[env_ids] = quat_rotate_inverse(self.base_quat[env_ids], self.gravity_vec[env_ids])
        self.base_lin_vel[env_ids] = quat_rotate_inverse(self.base_quat[env_ids], self.root_states[env_ids, 7:10])
        self.base_ang_vel[env_ids] = quat_rotate_inverse(self.base_quat[env_ids], self.root_states[env_ids, 10:13])
        self.gym.refresh_rigid_body_state_tensor(self.sim)
    def _reset_dofs_from_reference_motion(self, env_ids, ref_motions=None):
        """Reset the dof positions and velocities of the robots in env_ids to the reference motion at random time steps

        Args:
            env_ids (torch.Tensor): Tensor of shape (num_envs_to_reset,) containing the ids of the envs to reset
        """
        ref_dof_pos = self.amp_loader.get_dof_pos_batch(ref_motions)
        ref_dof_vel = self.amp_loader.get_dof_vel_batch(ref_motions)
        self._reset_amp_dof_states(env_ids, ref_dof_pos, ref_dof_vel)
        
    def _reset_root_states_from_reference_motion(self, env_ids, ref_motions=None):
        """Reset the root positions, orientations, linear and angular velocities of the robots in env_ids to the reference motion at random time steps

        Args:
            env_ids (torch.Tensor): Tensor of shape (num_envs_to_reset,) containing the ids of the envs to reset
        """
        ref_base_pos = self.amp_loader.get_base_pos_batch(ref_motions)
        ref_base_pos[:, 2] = self.base_init_state[2]
        base_pos = ref_base_pos + self.env_origins[env_ids]
        ref_base_rot = self.amp_loader.get_base_rot_batch(ref_motions)
        ref_base_lin_vel = self.amp_loader.get_base_lin_vel_batch(ref_motions)
        ref_base_ang_vel = self.amp_loader.get_base_ang_vel_batch(ref_motions)
        ref_base_lin_vel_w = quat_rotate(ref_base_rot, ref_base_lin_vel)
        ref_base_ang_vel_w = quat_rotate(ref_base_rot, ref_base_ang_vel)
        self._reset_amp_root_states(env_ids, base_pos, ref_base_rot, ref_base_lin_vel_w, ref_base_ang_vel_w)
