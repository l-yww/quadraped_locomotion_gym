import json
import matplotlib.pyplot as plt

# 步骤1: 加载JSON文件
with open('./data/torque.json', 'r') as file:
    torque_data = json.load(file)  # 这应该是一个双层数组，包含12个子数组   # motor_velocity_data
# torque_mujoco
# torque
with open('./data/motor_velocity_data.json', 'r') as file:
    dof_vel_data = json.load(file)  # 这应该是一个双层数组，包含12个子数组

with open('./data/dof_pos_list.json', 'r') as file:
    dof_pos_data = json.load(file)  # 这应该是一个双层数组，包含12个子数组
# dof_pos_list_mujoco
# dof_pos_list
with open('./data/actions_data.json', 'r') as file:
    actions_data = json.load(file)  # 这应该是一个双层数组，包含12个子数组
# action_data_mujoco
# actions_data
with open('./data/ref_pos_list.json', 'r') as file:
    ref_pos_list = json.load(file)  # 这应该是一个双层数组，包含12个子数组
with open('./data/ref_motor_vel_list.json', 'r') as file:
    ref_motor_vel_list = json.load(file)  # 这应该是一个双层数组，包含12个子数组

# 转置数据，以便按列绘制
torque_data_transposed = list(zip(*torque_data))
dof_vel_data_transposed = list(zip(*dof_vel_data))
dof_pos_data_transposed = list(zip(*dof_pos_data))
ref_pos_data_transposed = list(zip(*ref_pos_list))
ref_vel_data_transposed = list(zip(*ref_motor_vel_list))
actions_data_transposed = list(zip(*actions_data))
# 计算子图的行数和列数
num_columns_vel = max(len(dof_vel_data_transposed), len(ref_vel_data_transposed))
num_rows_vel = (num_columns_vel + 1) // 2  # 确保有足够的行来容纳所有的列

num_columns_pos = max(len(dof_pos_data_transposed), len(ref_pos_data_transposed))
num_rows_pos = (num_columns_pos + 1) // 2  # 确保有足够的行来容纳所有的列

num_columns_actions_data = len(actions_data_transposed)
num_rows_actions_data = (num_columns_actions_data + 1) // 2  # 确保有足够的行来容纳所有的列

# 创建第一个图表，并为vel绘制曲线
plt.figure(figsize=(15, num_rows_vel * 4))  # 调整图表大小以容纳所有子图
for i, column in enumerate(dof_vel_data_transposed[1:2]):
    plt.subplot(3, 2, 1)  # 创建子图
    plt.plot(column, label=f'left vel Curve {i+1}')
    plt.legend()

# for i, column in enumerate(ref_vel_data_transposed[1:2]):
#     plt.plot(column, label=f'left ref vel Curve {i+1}')
#     plt.legend()

for i, column in enumerate(actions_data_transposed[1:2]):
    # plt.subplot(num_rows_pos, 2, i+1)  # 创建子图
    scale_column = []
    for mm in column:
        scale_mm = mm * 10
        scale_column.append(scale_mm)
    plt.plot(scale_column, label=f'left action Curve {i+1}')
    plt.legend()
    plt.title(f'left wheel vel Curve {i+1} - Time Series Data')

for i, column in enumerate(dof_vel_data_transposed[3:4]):
    plt.subplot(3, 2, 2)  # 创建子图
    plt.plot(column, label=f'right vel Curve {i+1}')
    plt.legend()

# for i, column in enumerate(ref_vel_data_transposed[3:4]):
#     plt.plot(column, label=f'right ref vel Curve {i+1}')
#     plt.legend()

for i, column in enumerate(actions_data_transposed[3:4]):
    # plt.subplot(num_rows_pos, 2, i+1)  # 创建子图
    scale_column = []
    for mm in column:
        scale_mm = mm * 10
        scale_column.append(scale_mm)
    plt.plot(scale_column, label=f'action Curve {i+1}')
    plt.legend()
    plt.title(f'right wheel vel Curve {i+1} - Time Series Data')

for i, column in enumerate(dof_vel_data_transposed[0:1]):
    plt.subplot(3, 2, 3)  # 创建子图
    plt.plot(column, label=f'left vel Curve {i+1}')
    plt.legend()
    plt.title(f'left leg hip vel Curve {i+1} - Time Series Data')

for i, column in enumerate(dof_vel_data_transposed[2:3]):
    plt.subplot(3, 2, 4)  # 创建子图
    plt.plot(column, label=f'right vel Curve {i+1}')
    plt.legend()
    plt.title(f'right leg hip vel Curve {i+1} - Time Series Data')

for i, column in enumerate(torque_data_transposed[0:1]):
    plt.subplot(3, 2, 5)  # 创建子图
    plt.plot(column, label=f'left torque Curve {i+1}')
    plt.legend()
    plt.title(f'left leg hip torque Curve {i+1} - Time Series Data')

for i, column in enumerate(torque_data_transposed[2:3]):
    plt.subplot(3, 2, 6)  # 创建子图
    plt.plot(column, label=f'right torque Curve {i+1}')
    plt.legend()
    plt.title(f'right leg hip torque Curve {i+1} - Time Series Data')

plt.suptitle('dof vel Time Series Data')  # 设置总标题
plt.tight_layout()

# 创建第二个图表，并为pos绘制曲线
plt.figure(figsize=(15, num_rows_pos * 4))  # 调整图表大小以容纳所有子图
for i, column in enumerate(dof_pos_data_transposed[0:1]):
    plt.subplot(3, 2, 1)  # 创建子图
    plt.plot(column, label=f'left DOF Pos Curve {i+1}')
    plt.legend()

# for i, column in enumerate(ref_pos_data_transposed[0:1]):
#     plt.plot(column, label=f'left ref dof Pos Curve {i+1}')
#     plt.legend()

for i, column in enumerate(actions_data_transposed[0:1]):
    # plt.subplot(num_rows_pos, 2, i+1)  # 创建子图
    scale_column = []
    for mm in column:
        scale_mm = mm * 1.5
        scale_column.append(scale_mm)
    plt.plot(scale_column, label=f'action Curve {i+1}')
    plt.legend()
    plt.title(f'left hip pos Curve {i+1} - Time Series Data')

for i, column in enumerate(dof_pos_data_transposed[2:3]):
    plt.subplot(3, 2, 2)  # 创建子图
    plt.plot(column, label=f'right DOF Pos Curve {i+1}')
    plt.legend()

# for i, column in enumerate(ref_pos_data_transposed[2:3]):
#     plt.plot(column, label=f'right ref dof Pos Curve {i+1}')
#     plt.legend()

for i, column in enumerate(actions_data_transposed[2:3]):
    # plt.subplot(num_rows_pos, 2, i+1)  # 创建子图
    scale_column = []
    for mm in column:
        scale_mm = mm * 1.5
        scale_column.append(scale_mm)
    plt.plot(scale_column, label=f'action Curve {i+1}')
    plt.legend()
    plt.title(f'right hip pos Curve {i+1} - Time Series Data')

for i, column in enumerate(dof_pos_data_transposed[1:2]):
    plt.subplot(3, 2, 3)  # 创建子图
    plt.plot(column, label=f'left DOF Pos Curve {i+1}')
    plt.legend()
    plt.title(f'left wheel pos Curve {i+1} - Time Series Data')

for i, column in enumerate(dof_pos_data_transposed[3:4]):
    plt.subplot(3, 2, 4)  # 创建子图
    plt.plot(column, label=f'right DOF Pos Curve {i+1}')
    plt.legend()
    plt.title(f'right wheel pos Curve {i+1} - Time Series Data')

for i, column in enumerate(torque_data_transposed[1:2]):
    plt.subplot(3, 2, 5)  # 创建子图
    plt.plot(column, label=f'left torque Curve {i+1}')
    plt.legend()
    plt.title(f'left leg wheel torque Curve {i+1} - Time Series Data')

for i, column in enumerate(torque_data_transposed[3:4]):
    plt.subplot(3, 2, 6)  # 创建子图
    plt.plot(column, label=f'right torque Curve {i+1}')
    plt.legend()
    plt.title(f'right leg wheel torque Curve {i+1} - Time Series Data')

plt.suptitle('DOF Position and Reference Position Time Series Data')  # 设置总标题
plt.tight_layout()
plt.show()  # 显示第二个图表