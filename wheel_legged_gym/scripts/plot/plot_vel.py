import json
import matplotlib.pyplot as plt

# 步骤1: 加载JSON文件
with open('./data/torque.json', 'r') as file:
    torque_data = json.load(file)  # 这应该是一个双层数组，包含12个子数组   # motor_velocity_data
# torque_mujoco
# torque
with open('./data/motor_velocity_data.json', 'r') as file:
# with open('./data/dof_pos_list.json', 'r') as file:
    dof_pos_data = json.load(file)  # 这应该是一个双层数组，包含12个子数组
# dof_pos_list_mujoco
# dof_pos_list
with open('./data/actions_data.json', 'r') as file:
    actions_data = json.load(file)  # 这应该是一个双层数组，包含12个子数组
# action_data_mujoco
# actions_data

# 转置数据，以便按列绘制
torque_data_transposed = list(zip(*torque_data))
dof_pos_data_transposed = list(zip(*dof_pos_data))
actions_data_transposed = list(zip(*actions_data))
# 计算子图的行数和列数
num_columns_torque = len(torque_data_transposed)
num_rows_torque = (num_columns_torque + 1) // 2  # 确保有足够的行来容纳所有的列

num_columns_dof_pos = len(dof_pos_data_transposed)
num_rows_dof_pos = (num_columns_dof_pos + 1) // 2  # 确保有足够的行来容纳所有的列


num_columns_actions_data = len(actions_data_transposed)
num_rows_actions_data = (num_columns_actions_data + 1) // 2  # 确保有足够的行来容纳所有的列

# 创建第一个图表，并为torque_data绘制曲线
plt.figure(figsize=(15, num_rows_torque * 4))  # 调整图表大小以容纳所有子图
for i, column in enumerate(torque_data_transposed):
    plt.subplot(num_rows_torque, 2, i+1)  # 创建子图
    plt.plot(column, label=f'Torque Curve {i+1}')
    plt.legend()
    plt.title(f'Torque Curve {i+1} - Time Series Data')

plt.suptitle('Torque Time Series Data')  # 设置总标题
plt.tight_layout()

# 创建第二个图表，并为dof_pos_data和ref_pos_data绘制曲线
plt.figure(figsize=(15, max(num_rows_dof_pos, num_rows_dof_pos) * 4))  # 调整图表大小以容纳所有子图
for i, column in enumerate(dof_pos_data_transposed):
    plt.subplot(num_rows_dof_pos, 2, i+1)  # 创建子图
    plt.plot(column, label=f'DOF Pos Curve {i+1}')
    plt.legend()
    plt.title(f'DOF Pos Curve {i+1} - Time Series Data')

for i, column in enumerate(actions_data_transposed):
    plt.subplot(num_rows_actions_data, 2, i+1)  # 创建子图
    scale_column = []
    for mm in column:
        scale_mm = mm * 1.5
        scale_column.append(scale_mm)
    plt.plot(scale_column, label=f'Action Data Curve {i+1}')
    plt.legend()
    plt.title(f'Action Data Curve {i+1} - Time Series Data')

plt.suptitle('DOF Position and Reference Position Time Series Data')  # 设置总标题
plt.tight_layout()
plt.show()  # 显示第二个图表
