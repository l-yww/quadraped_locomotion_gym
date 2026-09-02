"""Functions to specify the symmetry in the observation and action space for Unitree G1 29dof."""

from __future__ import annotations

import torch

@torch.no_grad()
def compute_symmetric_states_k1(
    obs: torch.Tensor | None = None,
    actions: torch.Tensor | None = None,
    critic_obs: torch.Tensor | None = None
):
    """Augments the given observations and actions by applying symmetry transformations.

    This function creates augmented versions of the provided observations and actions by applying
    four symmetrical transformations: original, left-right, front-back, and diagonal. The symmetry
    transformations are beneficial for reinforcement learning tasks by providing additional
    diverse data without requiring additional data collection.

    Args:
        obs: The original observation tensor dictionary. Defaults to None.
        actions: The original actions tensor. Defaults to None.
        critic_obs: The original critic observation tensor. Defaults to None.

    Returns:
        Augmented observations and actions tensors, or None if the respective input was None.
    
    Attention:
        The symmetry functions are designed based on the specific structure of the observations and actions for the BoosterK1 22 dof robot. 
        If the structure of the observations or actions changes, the symmetry functions may need to be updated accordingly.
    
    """
    if obs is not None:
        batch_size = obs.shape[0]
        # since we have 2 different symmetries, we need to augment the batch size by 2
        obs_aug = obs.repeat(2, 1)  # create a new tensor with shape (2 * batch_size, obs_dim)

        # policy observation group
        # -- original
        obs_aug[:batch_size] = obs[:]
        # -- left-right
        obs_aug[batch_size : 2 * batch_size] = _transform_policy_obs_left_right_k1(obs)
    else:
        obs_aug = None

    if actions is not None:
        batch_size = actions.shape[0]
        # since we have 2 different symmetries, we need to augment the batch size by 2
        actions_aug = torch.zeros(batch_size * 2, actions.shape[1], device=actions.device)
        # -- original
        actions_aug[:batch_size] = actions[:]
        # -- left-right
        actions_aug[batch_size : 2 * batch_size] = _transform_actions_left_right(actions)
    else:
        actions_aug = None
    
    if critic_obs is not None:
        batch_size = critic_obs.shape[0]
        critic_obs_aug = critic_obs.repeat(2, 1)  # create a new tensor with shape (2 * batch_size, obs_dim)
        # -- original
        critic_obs_aug[:batch_size] = critic_obs[:]
        # -- left-right
        critic_obs_aug[batch_size : 2 * batch_size] = _transform_critic_obs_left_right_k1(critic_obs)
    else:
        critic_obs_aug = None

    return obs_aug, actions_aug, critic_obs_aug


"""
Symmetry functions for observations.
"""


def _transform_policy_obs_left_right_k1(obs: torch.Tensor) -> torch.Tensor:
    """Apply a left-right symmetry transformation to the observation tensor.

    This function modifies the given observation tensor by applying transformations
    that represent a symmetry with respect to the left-right axis. This includes
    negating certain components of the linear and angular velocities, projected gravity,
    velocity commands, and flipping the joint positions, joint velocities, and last actions
    for the ANYmal robot. Additionally, if height-scan data is present, it is flipped
    along the relevant dimension.

    Args:
        obs: The observation tensor to be transformed.

    Returns:
        The transformed observation tensor with left-right symmetry applied.
    """
    # copy observation tensor
    obs = obs.clone()
    device = obs.device
    joint_num = 22  # K1 22 dof

    HISTORY_LEN = 5
    VEL_CMD_DIM = 3
    PROJ_GRAV_DIM = 3
    ANG_VEL_DIM = 3
    JOINT_POS_DIM = joint_num
    JOINT_VEL_DIM = joint_num
    LAST_ACTIONS_DIM = joint_num

    end_idx = 0
    
    for h in range(HISTORY_LEN):
        # velocity command
        start_idx = end_idx
        end_idx = start_idx + VEL_CMD_DIM
        obs[:, start_idx:end_idx] = obs[:, start_idx:end_idx] * torch.tensor([1, -1, -1], device=device)
        
        # projected gravity
        start_idx = end_idx
        end_idx = start_idx + PROJ_GRAV_DIM
        obs[:, start_idx:end_idx] = obs[:, start_idx:end_idx] * torch.tensor([1, -1, 1], device=device)
        
        # ang vel
        start_idx = end_idx
        end_idx = start_idx + ANG_VEL_DIM
        obs[:, start_idx:end_idx] = obs[:, start_idx:end_idx] * torch.tensor([-1, 1, -1], device=device)

        # joint pos
        start_idx = end_idx
        end_idx = start_idx + JOINT_POS_DIM
        obs[:, start_idx:end_idx] = _switch_k1_22dof_joints_left_right(obs[:, start_idx:end_idx])

        # joint vel
        start_idx = end_idx
        end_idx = start_idx + JOINT_VEL_DIM
        obs[:, start_idx:end_idx] = _switch_k1_22dof_joints_left_right(obs[:, start_idx:end_idx])

        # last actions
        start_idx = end_idx
        end_idx = start_idx + LAST_ACTIONS_DIM
        obs[:, start_idx:end_idx] = _switch_k1_22dof_joints_left_right(obs[:, start_idx:end_idx])

    return obs

"""
Symmetry functions for critic observations.
"""

def _transform_critic_obs_left_right_k1(obs: torch.Tensor) -> torch.Tensor:
    """Apply a left-right symmetry transformation to the observation tensor.

    This function modifies the given observation tensor by applying transformations
    that represent a symmetry with respect to the left-right axis. This includes
    negating certain components of the linear and angular velocities, projected gravity,
    velocity commands, and flipping the joint positions, joint velocities, and last actions
    for the ANYmal robot. Additionally, if height-scan data is present, it is flipped
    along the relevant dimension.

    Args:
        env: The environment instance from which the observation is obtained.
        obs: The observation tensor to be transformed.

    Returns:
        The transformed observation tensor with left-right symmetry applied.
    """
    # copy observation tensor
    obs = obs.clone()
    device = obs.device
    joint_num = 22  # K1 22 dof

    HISTORY_LEN = 5
    VEL_CMD_DIM = 3
    PROJ_GRAV_DIM = 3
    ANG_VEL_DIM = 3
    JOINT_POS_DIM = joint_num
    JOINT_VEL_DIM = joint_num
    LAST_ACTIONS_DIM = joint_num
    LIN_VEL_DIM = 3
    KEY_BODY_POS_DIM = 5 * 3  # Assuming 5 key bodies, each with x, y, z positions

    end_idx = 0
    for h in range(HISTORY_LEN):
        # velocity command
        start_idx = end_idx
        end_idx = start_idx + VEL_CMD_DIM
        obs[:, start_idx:end_idx] = obs[:, start_idx:end_idx] * torch.tensor([1, -1, -1], device=device)
        
        # projected gravity
        start_idx = end_idx
        end_idx = start_idx + PROJ_GRAV_DIM
        obs[:, start_idx:end_idx] = obs[:, start_idx:end_idx] * torch.tensor([1, -1, 1], device=device)
        
        # ang vel
        start_idx = end_idx
        end_idx = start_idx + ANG_VEL_DIM
        obs[:, start_idx:end_idx] = obs[:, start_idx:end_idx] * torch.tensor([-1, 1, -1], device=device)

        # joint pos
        start_idx = end_idx
        end_idx = start_idx + JOINT_POS_DIM
        obs[:, start_idx:end_idx] = _switch_k1_22dof_joints_left_right(obs[:, start_idx:end_idx])

        # joint vel
        start_idx = end_idx
        end_idx = start_idx + JOINT_VEL_DIM
        obs[:, start_idx:end_idx] = _switch_k1_22dof_joints_left_right(obs[:, start_idx:end_idx])

        # last actions
        start_idx = end_idx
        end_idx = start_idx + LAST_ACTIONS_DIM
        obs[:, start_idx:end_idx] = _switch_k1_22dof_joints_left_right(obs[:, start_idx:end_idx])

        # base lin vel
        start_idx = end_idx
        end_idx = start_idx + LIN_VEL_DIM
        obs[:, start_idx:end_idx] = obs[:, start_idx:end_idx] * torch.tensor([1, -1, 1], device=device)
        
        # key body pos
        start_idx = end_idx
        end_idx = start_idx + KEY_BODY_POS_DIM
        obs[:, start_idx:end_idx] = _switch_k1_22dof_key_body_pos_left_right(obs[:, start_idx:end_idx])
        
    return obs

"""
Symmetry functions for privileged observations.
"""

def _transform_privileged_obs_left_right_k1(obs: torch.Tensor) -> torch.Tensor:
    """Apply a left-right symmetry transformation to the privileged observation tensor.

    This function modifies the given privileged observation tensor by applying transformations
    that represent a symmetry with respect to the left-right axis. The specific transformations
    applied depend on the structure of the privileged observations for the BoosterK1 22 dof robot.

    Args:
        obs: The privileged observation tensor to be transformed.

    Returns:
        The transformed privileged observation tensor with left-right symmetry applied.
    """
    # copy observation tensor
    obs = obs.clone()
    device = obs.device

    LIN_VEL_DIM = 3
    KEY_BODY_POS_DIM = 5 * 3  # Assuming 5 key bodies, each with x, y, z positions
    
    end_idx = 0
    
    # base lin vel
    start_idx = end_idx
    end_idx = start_idx + LIN_VEL_DIM
    obs[:, start_idx:end_idx] = obs[:, start_idx:end_idx] * torch.tensor([1, -1, 1], device=device)
    
    # key body pos
    start_idx = end_idx
    end_idx = start_idx + KEY_BODY_POS_DIM
    obs[:, start_idx:end_idx] = _switch_k1_22dof_key_body_pos_left_right(obs[:, start_idx:end_idx])

    return obs

"""
Symmetry functions for actions.
"""


def _transform_actions_left_right(actions: torch.Tensor) -> torch.Tensor:
    """Applies a left-right symmetry transformation to the actions tensor.

    This function modifies the given actions tensor by applying transformations
    that represent a symmetry with respect to the left-right axis. This includes
    flipping the joint positions, joint velocities, and last actions for the
    ANYmal robot.

    Args:
        actions: The actions tensor to be transformed.

    Returns:
        The transformed actions tensor with left-right symmetry applied.
    """
    actions = actions.clone()
    actions[:] = _switch_k1_22dof_joints_left_right(actions[:])
    return actions


"""
Unified joint names for booster k1 22 dof:
 0 - "AAHead_yaw"
 1 - "Head_pitch"
 2 - "ALeft_Shoulder_Pitch" 
 3 - "Left_Shoulder_Roll" 
 4 - "Left_Elbow_Pitch" 
 5 - "Left_Elbow_Yaw" 
 6 - "ARight_Shoulder_Pitch" 
 7 - "Right_Shoulder_Roll" 
 8 - "Right_Elbow_Pitch" 
 9 - "Right_Elbow_Yaw"
 10 - "Left_Hip_Pitch"
 11 - "Left_Hip_Roll" 
 12 - "Left_Hip_Yaw" 
 13 - "Left_Knee_Pitch" 
 14 - "Left_Ankle_Pitch" 
 15 - "Left_Ankle_Roll"
 16 - "Right_Hip_Pitch"
 17 - "Right_Hip_Roll" 
 18 - "Right_Hip_Yaw"
 19 - "Right_Knee_Pitch"
 20 - "Right_Ankle_Pitch"
 21 - "Right_Ankle_Roll"
"""


def _switch_k1_22dof_joints_left_right(joint_data: torch.Tensor) -> torch.Tensor:
    """Applies a left-right symmetry transformation to the joint data tensor."""
    joint_data_switched = torch.zeros_like(joint_data)

    # Indices for left and right joints
    left_indices = [2 ,3, 4, 5, 10, 11, 12, 13, 14, 15]
    right_indices = [6, 7, 8, 9, 16, 17, 18, 19, 20, 21]

    # Indices for roll and yaw joints that need sign flipping
    roll_indices = [3, 7, 11, 15, 17, 21]
    yaw_indices = [0, 5, 9, 12, 18]

    # Copy non-symmetric joints first (head joints in this case)
    joint_data_switched[..., [0, 1]] = joint_data[..., [0, 1]]

    # Swap left and right joints
    joint_data_switched[..., left_indices] = joint_data[..., right_indices]
    joint_data_switched[..., right_indices] = joint_data[..., left_indices]

    # Flip the sign of roll and yaw joints
    joint_data_switched[..., roll_indices] *= -1.0
    joint_data_switched[..., yaw_indices] *= -1.0

    return joint_data_switched

def _switch_k1_22dof_key_body_pos_left_right(key_body_pos: torch.Tensor) -> torch.Tensor:
    """Applies a left-right symmetry transformation to the key body positions tensor."""

    # Key bodies of K1 22 dof are defined as:
    # "Head_2"
    # "left_hand_link", "right_hand_link",
    # "left_foot_link", "right_foot_link",

    key_body_pos_switched = key_body_pos.clone()
    num_key_bodies = 4
    start_idx = 1 # skip head

    for i in range(num_key_bodies // 2):
        left_idx = start_idx + i * 2
        right_idx = start_idx + i * 2 + 1

        # Swap left and right key body positions
        key_body_pos_switched[..., left_idx * 3 : left_idx * 3 + 3] = key_body_pos[
            ..., right_idx * 3 : right_idx * 3 + 3
        ]
        key_body_pos_switched[..., right_idx * 3 : right_idx * 3 + 3] = key_body_pos[
            ..., left_idx * 3 : left_idx * 3 + 3
        ]

        # Flip the y-coordinate to reflect left-right symmetry
        key_body_pos_switched[..., left_idx * 3 + 1] *= -1.0
        key_body_pos_switched[..., right_idx * 3 + 1] *= -1.0

    return key_body_pos_switched

# def _switch_g1_29dof_key_body_pos_left_right(key_body_pos: torch.Tensor) -> torch.Tensor:
#     """Applies a left-right symmetry transformation to the key body positions tensor."""

#     # We assume that the key body are in pair, for example:
#     # "left_ankle_roll_link",
#     # "right_ankle_roll_link",
#     # "left_wrist_yaw_link",
#     # "right_wrist_yaw_link",
#     # "left_shoulder_roll_link",
#     # "right_shoulder_roll_link",

#     key_body_pos_switched = key_body_pos.clone()
#     num_key_bodies = key_body_pos.shape[-1] // 3

#     for i in range(num_key_bodies // 2):
#         left_idx = i * 2
#         right_idx = i * 2 + 1

#         # Swap left and right key body positions
#         key_body_pos_switched[..., left_idx * 3 : left_idx * 3 + 3] = key_body_pos[
#             ..., right_idx * 3 : right_idx * 3 + 3
#         ]
#         key_body_pos_switched[..., right_idx * 3 : right_idx * 3 + 3] = key_body_pos[
#             ..., left_idx * 3 : left_idx * 3 + 3
#         ]

#         # Flip the y-coordinate to reflect left-right symmetry
#         key_body_pos_switched[..., left_idx * 3 + 1] *= -1.0
#         key_body_pos_switched[..., right_idx * 3 + 1] *= -1.0

#     return key_body_pos_switched


# =============================================================================
# Adam Robot Symmetry Functions (23 DOF)
# =============================================================================
"""
Unified joint names for Adam robot (23 dof):
 0 - "hipPitch_Left"
 1 - "hipRoll_Left"      (需翻转)
 2 - "hipYaw_Left"       (需翻转)
 3 - "kneePitch_Left"
 4 - "anklePitch_Left"
 5 - "ankleRoll_Left"     (需翻转)
 6 - "hipPitch_Right"
 7 - "hipRoll_Right"     (需翻转)
 8 - "hipYaw_Right"      (需翻转)
 9 - "kneePitch_Right"
10 - "anklePitch_Right"
11 - "ankleRoll_Right"   (需翻转)
12 - "waistRoll"         (中间，不交换)
13 - "waistPitch"        (中间，不交换)
14 - "waistYaw"          (需翻转)
15 - "shoulderPitch_Left"
16 - "shoulderRoll_Left" (需翻转)
17 - "shoulderYaw_Left"  (需翻转)
18 - "elbow_Left"
19 - "shoulderPitch_Right"
20 - "shoulderRoll_Right" (需翻转)
21 - "shoulderYaw_Right" (需翻转)
22 - "elbow_Right"
"""


@torch.no_grad()
def compute_symmetric_states_adam(
    obs: torch.Tensor | None = None,
    actions: torch.Tensor | None = None,
    critic_obs: torch.Tensor | None = None
):
    """Augments the given observations and actions by applying symmetry transformations for Adam robot.

    Args:
        obs: The original observation tensor. Defaults to None.
        actions: The original actions tensor. Defaults to None.
        critic_obs: The original critic observation tensor. Defaults to None.

    Returns:
        Augmented observations and actions tensors, or None if the respective input was None.
    """
    if obs is not None:
        batch_size = obs.shape[0]
        obs_aug = obs.repeat(2, 1)
        obs_aug[:batch_size] = obs[:]
        obs_aug[batch_size : 2 * batch_size] = _transform_policy_obs_left_right_adam(obs)
    else:
        obs_aug = None

    if actions is not None:
        batch_size = actions.shape[0]
        actions_aug = torch.zeros(batch_size * 2, actions.shape[1], device=actions.device)
        actions_aug[:batch_size] = actions[:]
        actions_aug[batch_size : 2 * batch_size] = _transform_actions_left_right_adam(actions)
    else:
        actions_aug = None

    if critic_obs is not None:
        batch_size = critic_obs.shape[0]
        critic_obs_aug = critic_obs.repeat(2, 1)
        critic_obs_aug[:batch_size] = critic_obs[:]
        critic_obs_aug[batch_size : 2 * batch_size] = _transform_critic_obs_left_right_adam(critic_obs)
    else:
        critic_obs_aug = None

    return obs_aug, actions_aug, critic_obs_aug


def _transform_policy_obs_left_right_adam(obs: torch.Tensor) -> torch.Tensor:
    """Apply a left-right symmetry transformation to the observation tensor for Adam robot."""
    obs = obs.clone()
    device = obs.device
    joint_num = 23  # Adam 23 dof

    HISTORY_LEN = 5
    VEL_CMD_DIM = 3
    PROJ_GRAV_DIM = 3
    ANG_VEL_DIM = 3
    JOINT_POS_DIM = joint_num
    JOINT_VEL_DIM = joint_num
    LAST_ACTIONS_DIM = joint_num

    end_idx = 0

    for h in range(HISTORY_LEN):
        # velocity command: [1, -1, -1]
        start_idx = end_idx
        end_idx = start_idx + VEL_CMD_DIM
        obs[:, start_idx:end_idx] = obs[:, start_idx:end_idx] * torch.tensor([1, -1, -1], device=device)

        # projected gravity: [1, -1, 1]
        start_idx = end_idx
        end_idx = start_idx + PROJ_GRAV_DIM
        obs[:, start_idx:end_idx] = obs[:, start_idx:end_idx] * torch.tensor([1, -1, 1], device=device)

        # ang vel: [-1, 1, -1]
        start_idx = end_idx
        end_idx = start_idx + ANG_VEL_DIM
        obs[:, start_idx:end_idx] = obs[:, start_idx:end_idx] * torch.tensor([-1, 1, -1], device=device)

        # joint pos
        start_idx = end_idx
        end_idx = start_idx + JOINT_POS_DIM
        obs[:, start_idx:end_idx] = _switch_adam_joints_left_right(obs[:, start_idx:end_idx])

        # joint vel
        start_idx = end_idx
        end_idx = start_idx + JOINT_VEL_DIM
        obs[:, start_idx:end_idx] = _switch_adam_joints_left_right(obs[:, start_idx:end_idx])

        # last actions
        start_idx = end_idx
        end_idx = start_idx + LAST_ACTIONS_DIM
        obs[:, start_idx:end_idx] = _switch_adam_joints_left_right(obs[:, start_idx:end_idx])

    return obs


def _transform_critic_obs_left_right_adam(obs: torch.Tensor) -> torch.Tensor:
    """Apply a left-right symmetry transformation to Adam stacked critic observations."""
    obs = obs.clone()

    HISTORY_LEN = 5
    MIN_FRAME_DIM = 3 + 3 + 3 + 23 + 23 + 23 + 3 + 2 * 3
    if obs.shape[1] % HISTORY_LEN == 0 and obs.shape[1] // HISTORY_LEN >= MIN_FRAME_DIM:
        frame_dim = obs.shape[1] // HISTORY_LEN
        for frame_idx in range(HISTORY_LEN):
            start_idx = frame_idx * frame_dim
            end_idx = start_idx + frame_dim
            obs[:, start_idx:end_idx] = _transform_critic_obs_frame_left_right_adam(
                obs[:, start_idx:end_idx]
            )
        return obs

    return _transform_critic_obs_frame_left_right_adam(obs)


def _transform_critic_obs_frame_left_right_adam(obs: torch.Tensor) -> torch.Tensor:
    """Apply Adam left-right symmetry to one critic observation frame."""
    obs = obs.clone()
    device = obs.device
    joint_num = 23  # Adam 23 dof

    VEL_CMD_DIM = 3
    PROJ_GRAV_DIM = 3
    ANG_VEL_DIM = 3
    JOINT_POS_DIM = joint_num
    JOINT_VEL_DIM = joint_num
    LAST_ACTIONS_DIM = joint_num
    LIN_VEL_DIM = 3
    KEY_BODY_POS_DIM = 2*3  # toeLeft, toeRight
    FEET_Z_DIM = 2

    def has_segment(start_idx: int, segment_dim: int) -> bool:
        return start_idx + segment_dim <= obs.shape[1]

    end_idx = 0
    if has_segment(end_idx, VEL_CMD_DIM):
        start_idx = end_idx
        end_idx = start_idx + VEL_CMD_DIM
        obs[:, start_idx:end_idx] = obs[:, start_idx:end_idx] * torch.tensor([1, -1, -1], device=device)

    if has_segment(end_idx, PROJ_GRAV_DIM):
        start_idx = end_idx
        end_idx = start_idx + PROJ_GRAV_DIM
        obs[:, start_idx:end_idx] = obs[:, start_idx:end_idx] * torch.tensor([1, -1, 1], device=device)

    if has_segment(end_idx, ANG_VEL_DIM):
        start_idx = end_idx
        end_idx = start_idx + ANG_VEL_DIM
        obs[:, start_idx:end_idx] = obs[:, start_idx:end_idx] * torch.tensor([-1, 1, -1], device=device)

    if has_segment(end_idx, JOINT_POS_DIM):
        start_idx = end_idx
        end_idx = start_idx + JOINT_POS_DIM
        obs[:, start_idx:end_idx] = _switch_adam_joints_left_right(obs[:, start_idx:end_idx])

    if has_segment(end_idx, JOINT_VEL_DIM):
        start_idx = end_idx
        end_idx = start_idx + JOINT_VEL_DIM
        obs[:, start_idx:end_idx] = _switch_adam_joints_left_right(obs[:, start_idx:end_idx])

    if has_segment(end_idx, LAST_ACTIONS_DIM):
        start_idx = end_idx
        end_idx = start_idx + LAST_ACTIONS_DIM
        obs[:, start_idx:end_idx] = _switch_adam_joints_left_right(obs[:, start_idx:end_idx])

    if has_segment(end_idx, LIN_VEL_DIM):
        start_idx = end_idx
        end_idx = start_idx + LIN_VEL_DIM
        obs[:, start_idx:end_idx] = obs[:, start_idx:end_idx] * torch.tensor([1, -1, 1], device=device)

    if has_segment(end_idx, KEY_BODY_POS_DIM):
        start_idx = end_idx
        end_idx = start_idx + KEY_BODY_POS_DIM
        obs[:, start_idx:end_idx] = _switch_adam_key_body_pos_left_right(obs[:, start_idx:end_idx])

    if has_segment(end_idx, FEET_Z_DIM):
        start_idx = end_idx
        end_idx = start_idx + FEET_Z_DIM
        obs[:, start_idx:end_idx] = _switch_adam_feet_z_left_right(obs[:, start_idx:end_idx])

    return obs


def _transform_privileged_obs_left_right_adam(obs: torch.Tensor) -> torch.Tensor:
    """Apply a left-right symmetry transformation to the privileged observation tensor for Adam robot."""
    obs = obs.clone()
    device = obs.device

    LIN_VEL_DIM = 3
    KEY_BODY_POS_DIM = 8 * 3  # toeLeft, toeRight

    end_idx = 0

    # base lin vel
    start_idx = end_idx
    end_idx = start_idx + LIN_VEL_DIM
    obs[:, start_idx:end_idx] = obs[:, start_idx:end_idx] * torch.tensor([1, -1, 1], device=device)

    # key body pos
    start_idx = end_idx
    end_idx = start_idx + KEY_BODY_POS_DIM
    obs[:, start_idx:end_idx] = _switch_adam_key_body_pos_left_right(obs[:, start_idx:end_idx])

    return obs


def _transform_actions_left_right_adam(actions: torch.Tensor) -> torch.Tensor:
    """Applies a left-right symmetry transformation to the actions tensor for Adam robot."""
    actions = actions.clone()
    actions[:] = _switch_adam_joints_left_right(actions[:])
    return actions


def _switch_adam_joints_left_right(joint_data: torch.Tensor) -> torch.Tensor:
    """Applies a left-right symmetry transformation to the joint data tensor for Adam robot."""
    joint_data_switched = torch.zeros_like(joint_data)

    # Left indices: left leg (0-5) + left arm (15-18)
    left_indices = [0, 1, 2, 3, 4, 5, 15, 16, 17, 18]
    # Right indices: right leg (6-11) + right arm (19-22)
    right_indices = [6, 7, 8, 9, 10, 11, 19, 20, 21, 22]
    # Middle indices (no swap): waist (12, 13, 14)
    middle_indices = [12, 13, 14]

    # Roll joints that need sign flipping
    roll_indices = [1, 5, 7, 11, 16, 20]
    # Yaw joints that need sign flipping
    yaw_indices = [2, 8, 17, 21]

    # Copy middle joints first (waist)
    joint_data_switched[..., middle_indices] = joint_data[..., middle_indices]

    # Swap left and right joints
    joint_data_switched[..., left_indices] = joint_data[..., right_indices]
    joint_data_switched[..., right_indices] = joint_data[..., left_indices]

    # Flip the sign of roll and yaw joints
    joint_data_switched[..., roll_indices] *= -1.0
    joint_data_switched[..., yaw_indices] *= -1.0

    return joint_data_switched


def _switch_adam_key_body_pos_left_right(key_body_pos: torch.Tensor) -> torch.Tensor:
    """Applies a left-right symmetry transformation to the key body positions tensor for Adam robot.

    Adam key bodies: toeLeft(0), toeRight(1)
    """
    key_body_pos_switched = key_body_pos.clone()

    # Swap left and right key body positions: (0,1)
    key_body_pos_switched[..., 0:3] = key_body_pos[..., 3:6]
    key_body_pos_switched[..., 3:6] = key_body_pos[..., 0:3]

    # Flip the y-coordinate to reflect left-right symmetry
    key_body_pos_switched[..., 1] *= -1.0
    key_body_pos_switched[..., 4] *= -1.0

    return key_body_pos_switched


def _switch_adam_feet_z_left_right(feet_z: torch.Tensor) -> torch.Tensor:
    """Swap Adam toe z-height observations: toeLeft <-> toeRight."""
    feet_z_switched = torch.zeros_like(feet_z)
    feet_z_switched[..., 0] = feet_z[..., 1]
    feet_z_switched[..., 1] = feet_z[..., 0]
    return feet_z_switched



def _transform_critic_obs_left_right_d1(obs: torch.Tensor) -> torch.Tensor:
    """Apply left-right mirror symmetry to D1 stacked critic observations."""
    obs = obs.clone()

    QUAD_HIM_PRIV_FRAME_DIM = 266
    if obs.shape[1] % QUAD_HIM_PRIV_FRAME_DIM == 0:
        frame_dim = QUAD_HIM_PRIV_FRAME_DIM
        num_frames = obs.shape[1] // frame_dim
        for frame_idx in range(num_frames):
            start_idx = frame_idx * frame_dim
            end_idx = start_idx + frame_dim
            obs[:, start_idx:end_idx] = _transform_critic_obs_frame_left_right_d1(
                obs[:, start_idx:end_idx]
            )
        return obs

    HISTORY_LEN = 5
    MIN_FRAME_DIM = 3 + 12 + 12 + 12 + 3 + 4 * 3 + 3
    if obs.shape[1] % HISTORY_LEN == 0 and obs.shape[1] // HISTORY_LEN >= MIN_FRAME_DIM:
        frame_dim = obs.shape[1] // HISTORY_LEN
        for frame_idx in range(HISTORY_LEN):
            start_idx = frame_idx * frame_dim
            end_idx = start_idx + frame_dim
            obs[:, start_idx:end_idx] = _transform_critic_obs_frame_left_right_d1(
                obs[:, start_idx:end_idx]
            )
        return obs

    return _transform_critic_obs_frame_left_right_d1(obs)


def _transform_critic_obs_frame_left_right_d1(obs: torch.Tensor) -> torch.Tensor:
    """Apply left-right mirror symmetry to one D1 critic observation frame."""
    # quadruped_arm_him privileged obs layout:
    # [cmd(3), dof_pos(12), dof_vel(12), action(12), ang_vel(3), gravity(3),
    #  friction/payload/inertia(3), motor_strength(12), motor_offset(12),
    #  com(3), heightmap(17*11), base_height(1), base_lin_vel(3)] = 266
    if obs.shape[1] == 266:
        return _transform_quad_him_privileged_obs_frame_left_right_d1(obs)

    obs = obs.clone()
    device = obs.device

    VEL_CMD_DIM = 3
    JOINT_POS_DIM = 12
    JOINT_VEL_DIM = 12
    LAST_ACTIONS_DIM = 12
    ANG_VEL_DIM = 3
    KEY_BODY_POS_DIM = 4 * 3
    PROJ_GRAV_DIM = 3
    SCALAR_DIM = 1
    LIN_VEL_DIM = 3

    def has_segment(start_idx: int, segment_dim: int) -> bool:
        return start_idx + segment_dim <= obs.shape[1]

    end_idx = 0
    if has_segment(end_idx, VEL_CMD_DIM):
        start_idx = end_idx
        end_idx = start_idx + VEL_CMD_DIM
        obs[:, start_idx:end_idx] = obs[:, start_idx:end_idx] * torch.tensor([1, -1, -1], device=device)

    if has_segment(end_idx, JOINT_POS_DIM):
        start_idx = end_idx
        end_idx = start_idx + JOINT_POS_DIM
        obs[:, start_idx:end_idx] = _switch_d1_joints_left_right(obs[:, start_idx:end_idx])

    if has_segment(end_idx, JOINT_VEL_DIM):
        start_idx = end_idx
        end_idx = start_idx + JOINT_VEL_DIM
        obs[:, start_idx:end_idx] = _switch_d1_joints_left_right(obs[:, start_idx:end_idx])

    if has_segment(end_idx, LAST_ACTIONS_DIM):
        start_idx = end_idx
        end_idx = start_idx + LAST_ACTIONS_DIM
        obs[:, start_idx:end_idx] = _switch_d1_joints_left_right(obs[:, start_idx:end_idx])

    if has_segment(end_idx, ANG_VEL_DIM):
        start_idx = end_idx
        end_idx = start_idx + ANG_VEL_DIM
        obs[:, start_idx:end_idx] = obs[:, start_idx:end_idx] * torch.tensor([-1, 1, -1], device=device)

    if has_segment(end_idx, KEY_BODY_POS_DIM):
        start_idx = end_idx
        end_idx = start_idx + KEY_BODY_POS_DIM
        obs[:, start_idx:end_idx] = _switch_d1_key_body_pos_left_right(obs[:, start_idx:end_idx])

    if has_segment(end_idx, PROJ_GRAV_DIM):
        start_idx = end_idx
        end_idx = start_idx + PROJ_GRAV_DIM
        obs[:, start_idx:end_idx] = obs[:, start_idx:end_idx] * torch.tensor([1, -1, 1], device=device)

    # Current AmpD1 critic frame tail: base_height, base_lin_vel.
    if has_segment(end_idx, SCALAR_DIM):
        end_idx += SCALAR_DIM

    if has_segment(end_idx, LIN_VEL_DIM):
        start_idx = end_idx
        end_idx = start_idx + LIN_VEL_DIM
        obs[:, start_idx:end_idx] = obs[:, start_idx:end_idx] * torch.tensor([1, -1, 1], device=device)

    return obs


def _transform_quad_him_privileged_obs_frame_left_right_d1(obs: torch.Tensor) -> torch.Tensor:
    """Mirror quadruped_arm_him privileged obs without corrupting privileged tail."""
    obs = obs.clone()
    device = obs.device

    end_idx = 0

    # cmd: [vx, vy, yaw]
    start_idx = end_idx
    end_idx = start_idx + 3
    obs[:, start_idx:end_idx] = obs[:, start_idx:end_idx] * torch.tensor([1, -1, -1], device=device)

    # dof_pos, dof_vel, action: signed joint-coordinate quantities
    for _ in range(3):
        start_idx = end_idx
        end_idx = start_idx + 12
        obs[:, start_idx:end_idx] = _switch_d1_joints_left_right(obs[:, start_idx:end_idx])

    # base_ang_vel: [wx, wy, wz]
    start_idx = end_idx
    end_idx = start_idx + 3
    obs[:, start_idx:end_idx] = obs[:, start_idx:end_idx] * torch.tensor([-1, 1, -1], device=device)

    # projected_gravity: [gx, gy, gz]
    start_idx = end_idx
    end_idx = start_idx + 3
    obs[:, start_idx:end_idx] = obs[:, start_idx:end_idx] * torch.tensor([1, -1, 1], device=device)

    # friction, payload, inertia: mirror-invariant scalars
    end_idx += 3

    # motor_strength: per-joint scalar multipliers, swap left/right without sign flip
    start_idx = end_idx
    end_idx = start_idx + 12
    obs[:, start_idx:end_idx] = _swap_d1_joints_left_right_no_sign(obs[:, start_idx:end_idx])

    # motor_offset: joint-coordinate offsets, swap left/right with hip sign flip
    start_idx = end_idx
    end_idx = start_idx + 12
    obs[:, start_idx:end_idx] = _switch_d1_joints_left_right(obs[:, start_idx:end_idx])

    # com displacement: [x, y, z]
    start_idx = end_idx
    end_idx = start_idx + 3
    obs[:, start_idx:end_idx] = obs[:, start_idx:end_idx] * torch.tensor([1, -1, 1], device=device)

    # heightmap is flattened from meshgrid(x, y): 17 x-points, 11 y-points.
    start_idx = end_idx
    end_idx = start_idx + 17 * 11
    heightmap = obs[:, start_idx:end_idx].reshape(obs.shape[0], 17, 11)
    obs[:, start_idx:end_idx] = heightmap.flip(dims=[2]).reshape(obs.shape[0], -1)

    # base_height: mirror-invariant scalar
    end_idx += 1

    # base_lin_vel: [vx, vy, vz]
    start_idx = end_idx
    end_idx = start_idx + 3
    obs[:, start_idx:end_idx] = obs[:, start_idx:end_idx] * torch.tensor([1, -1, 1], device=device)

    return obs


def _transform_actions_left_right_d1(actions: torch.Tensor) -> torch.Tensor:
    """Apply left-right mirror symmetry to D1 actions."""
    actions = actions.clone()
    actions[:] = _switch_d1_joints_left_right(actions[:])
    return actions


def _switch_d1_joints_left_right(joint_data: torch.Tensor) -> torch.Tensor:
    """Swap D1 left/right legs so trot pair FL/RR maps to FR/RL."""
    joint_data_switched = torch.zeros_like(joint_data)

    left_indices = [0, 1, 2, 6, 7, 8]
    right_indices = [3, 4, 5, 9, 10, 11]
    hip_indices = [0, 3, 6, 9]

    joint_data_switched[..., left_indices] = joint_data[..., right_indices]
    joint_data_switched[..., right_indices] = joint_data[..., left_indices]
    joint_data_switched[..., hip_indices] *= -1.0

    return joint_data_switched


def _swap_d1_joints_left_right_no_sign(joint_data: torch.Tensor) -> torch.Tensor:
    """Swap D1 left/right per-joint scalar data without changing signs."""
    joint_data_switched = torch.zeros_like(joint_data)

    left_indices = [0, 1, 2, 6, 7, 8]
    right_indices = [3, 4, 5, 9, 10, 11]

    joint_data_switched[..., left_indices] = joint_data[..., right_indices]
    joint_data_switched[..., right_indices] = joint_data[..., left_indices]

    return joint_data_switched


def _switch_d1_key_body_pos_left_right(key_body_pos: torch.Tensor) -> torch.Tensor:
    """Swap D1 feet FL<->FR and RL<->RR, then mirror y."""
    key_body_pos_switched = key_body_pos.clone()

    for left_idx, right_idx in ((0, 1), (2, 3)):
        key_body_pos_switched[..., left_idx * 3 : left_idx * 3 + 3] = key_body_pos[
            ..., right_idx * 3 : right_idx * 3 + 3
        ]
        key_body_pos_switched[..., right_idx * 3 : right_idx * 3 + 3] = key_body_pos[
            ..., left_idx * 3 : left_idx * 3 + 3
        ]
        key_body_pos_switched[..., left_idx * 3 + 1] *= -1.0
        key_body_pos_switched[..., right_idx * 3 + 1] *= -1.0

    return key_body_pos_switched


def _switch_d1_feet_z_left_right(feet_z: torch.Tensor) -> torch.Tensor:
    """Swap D1 foot z observations FL<->FR and RL<->RR."""
    feet_z_switched = torch.zeros_like(feet_z)
    feet_z_switched[..., [0, 2]] = feet_z[..., [1, 3]]
    feet_z_switched[..., [1, 3]] = feet_z[..., [0, 2]]
    return feet_z_switched


# =============================================================================
# D1 Quadruped Symmetry (12 DOF, URDF order, 自适应 HISTORY_LEN)
# =============================================================================
"""
机器狗 d1 (四足 12 DOF, URDF 顺序). 
  - 关节顺序/符号沿用 d1 (URDF: FL三/FR三/RL三/RR三, hip 取反), 已验证正确
  - critic 已自适应 (d1 版本), 直接复用 _transform_critic_obs_left_right_d1

适用 obs 布局 (每帧 single_obs_dim, 默认 45 关时钟):
  [cmd(3) + dof_pos(12) + dof_vel(12) + action(12) + ang_vel(3) + gravity(3)] = 45
"""

@torch.no_grad()
def compute_symmetric_states_d1(
    obs: torch.Tensor | None = None,
    actions: torch.Tensor | None = None,
    critic_obs: torch.Tensor | None = None,
    single_obs_dim: int = 45,
):
    """D1 四足左右镜像对称 (URDF 顺序, HISTORY_LEN 自适应 frame_stack).
    Args:
        single_obs_dim: 单帧 obs 维度 (关时钟=45). HISTORY_LEN = obs.shape[1] // single_obs_dim.
    """
    if obs is not None:
        batch_size = obs.shape[0]
        obs_aug = obs.repeat(2, 1)
        obs_aug[:batch_size] = obs[:]
        obs_aug[batch_size : 2 * batch_size] = _transform_policy_obs_left_right_d1(obs, single_obs_dim)
    else:
        obs_aug = None

    if actions is not None:
        batch_size = actions.shape[0]
        actions_aug = torch.zeros(batch_size * 2, actions.shape[1], device=actions.device)
        actions_aug[:batch_size] = actions[:]
        actions_aug[batch_size : 2 * batch_size] = _transform_actions_left_right_d1(actions)
    else:
        actions_aug = None

    if critic_obs is not None:
        batch_size = critic_obs.shape[0]
        critic_obs_aug = critic_obs.repeat(2, 1)
        critic_obs_aug[:batch_size] = critic_obs[:]
        # critic 已自适应 HISTORY_LEN (d1 版本), 直接复用
        critic_obs_aug[batch_size : 2 * batch_size] = _transform_critic_obs_left_right_d1(critic_obs)
    else:
        critic_obs_aug = None

    return obs_aug, actions_aug, critic_obs_aug


def _transform_policy_obs_left_right_d1(obs: torch.Tensor, single_obs_dim: int = 45) -> torch.Tensor:
    """D1 actor obs 左右镜像 (URDF 顺序, HISTORY_LEN 自适应).

    每帧布局 (single_obs_dim=45):
      [cmd(3) + dof_pos(12) + dof_vel(12) + action(12) + ang_vel(3) + gravity(3)]
    关节 12 维 URDF 顺序: [FL_hip,FL_thigh,FL_calf, FR_hip,FR_thigh,FR_calf,
                           RL_hip,RL_thigh,RL_calf, RR_hip,RR_thigh,RR_calf]
    """
    obs = obs.clone()
    device = obs.device

    HISTORY_LEN = obs.shape[1] // single_obs_dim  # 自适应: amp_d1=5, cowa=30
    VEL_CMD_DIM = 3
    JOINT_DIM = 12  # dof_pos / dof_vel / action 各 12
    ANG_VEL_DIM = 3
    PROJ_GRAV_DIM = 3

    end_idx = 0
    for _ in range(HISTORY_LEN):
        # vel_cmd: [vx, -vy, -yaw]
        # print(HISTORY_LEN)
        start_idx = end_idx
        end_idx = start_idx + VEL_CMD_DIM
        obs[:, start_idx:end_idx] = obs[:, start_idx:end_idx] * torch.tensor([1, -1, -1], device=device)

        # dof_pos / dof_vel / action (URDF 顺序, 复用 d1 关节交换)
        for _ in range(3):
            start_idx = end_idx
            end_idx = start_idx + JOINT_DIM
            obs[:, start_idx:end_idx] = _switch_d1_joints_left_right(obs[:, start_idx:end_idx])

        # ang_vel: [-wx, wy, -wz]
        start_idx = end_idx
        end_idx = start_idx + ANG_VEL_DIM
        obs[:, start_idx:end_idx] = obs[:, start_idx:end_idx] * torch.tensor([-1, 1, -1], device=device)

        # projected_gravity: [gx, -gy, gz]
        start_idx = end_idx
        end_idx = start_idx + PROJ_GRAV_DIM
        obs[:, start_idx:end_idx] = obs[:, start_idx:end_idx] * torch.tensor([1, -1, 1], device=device)

    return obs
